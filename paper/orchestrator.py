"""The daily weekday loop: staggered entry, clock-based rotation, inverse-vol sizing,
25% vol-target.

Cadence (matches the agreed spec):
  * Build-up: each weekday buy the single highest-ranked eligible name not yet held,
    until the book reaches `top_n` (~30).
  * Rotation: each position runs a 21-trading-day clock. When it's up, KEEP the name if
    it's still within the top-`hold_n` band (~45); otherwise SELL it. Freed slots refill
    at the same ~1 new buy/day pace.
Sizing: each entry takes its share of the REMAINING gap to the NAV-based target (not a fixed
budget/top_n), tilted by inverse 63d volatility (clipped, and rescaled to mean exactly 1), then
scaled by a portfolio vol-target factor (gentle 25% → ~fully invested except in vol spikes) and
finally hard-capped at available cash. Gross therefore lands ON budget rather than up to ~2x it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from paper.state import HOLD_DAYS, PortfolioState, Position
from risk_guard import MarginLimits, RiskLimits, check_order, liquidity_check, price_sane


@dataclass
class PaperConfig:
    budget: float = 100_000.0
    top_n: int = 30
    hold_n: int = 45           # no-trade band
    max_new_buys_per_day: int = 1
    vol_target: float = 0.25   # annualized portfolio vol target
    vol_window: int = 63       # days for per-name realised vol
    inv_vol_clip: tuple[float, float] = (0.5, 2.0)
    hold_days: int = HOLD_DAYS
    # Independent pre-trade guard. It re-derives every order from the budget rather than trusting
    # the sizing above, because a guard that reuses the strategy's arithmetic cannot catch the
    # strategy's own bug — and this file had exactly such a bug (gross reaching ~2x budget) two
    # days before it was wired in. Rejections are LOGGED and skipped, never silent.
    use_risk_guard: bool = True


def _annual_vol(adj: pd.DataFrame, window: int) -> pd.Series:
    r = adj.pct_change(fill_method=None)
    return r.tail(window).std() * np.sqrt(252)


def _normalised_tilts(vol: pd.Series, cfg: PaperConfig) -> pd.Series:
    """Inverse-vol tilts rescaled to average EXACTLY 1.

    The raw tilt clip(median_vol/vol, 0.5, 2.0) does not sum to top_n: the 2.0 cap boosts
    low-vol names while the 0.5 floor cuts high-vol ones, and which dominates depends entirely
    on the vol cross-section of whichever names rank top that day. Simulated over 5,000 random
    30-name books the total ran 0.92x-1.09x of budget (59% of them OVER), with a THEORETICAL
    range of 0.5x-2.0x — and `_gross_scalar` is clipped to <=1 so it can only ever reduce, never
    correct an overshoot. Dividing by the mean pins the sum to budget for any cross-section.

    Rescaling preserves the tilt SHAPE exactly (corr with the raw tilts = 1.0000); it changes
    the level, not the structure.
    """
    if not len(vol):
        return pd.Series(dtype=float)
    ref = float(np.nanmedian(vol.values))
    if not ref or ref != ref:
        return pd.Series(1.0, index=vol.index)
    t = pd.Series(np.clip(ref / vol, *cfg.inv_vol_clip), index=vol.index).replace(
        [np.inf, -np.inf], np.nan).fillna(1.0)
    m = float(np.nanmean(t.values))
    return t / m if (m and m == m and m > 0) else pd.Series(1.0, index=vol.index)


def _slot_usd(state, marks: dict[str, float], fx: dict[str, float],
              cfg: PaperConfig, gross_scalar: float) -> float:
    """Dollars available for ONE new position: the remaining gap split across free slots.

    Sizing every entry at budget/top_n ignores drift — positions are entered on a staggered
    clock and never resized, so winners grow and losers shrink and the realised book wanders.
    Sizing off the REMAINING gap self-corrects at zero extra turnover, since it only re-scales
    an order that was being placed anyway.

    The gap is measured against **NAV**, not `cfg.budget`. Against a fixed budget, a book that
    had drawn down would try to "top up" to the original number — buying with money the account
    no longer has. NAV (cash + positions) is the amount actually available, so the book stays
    ~fully invested and compounds, and `cfg.budget` only sets the STARTING capital.
    The cash constraint is NOT applied here — see `_size_shares`, which applies it after the
    inverse-vol tilt has been multiplied in.
    """
    nav = state.nav(marks, fx)
    invested = state.positions_value_usd(marks, fx)
    slots_free = max(cfg.top_n - len(state.tickers), 1)
    remaining = max(nav * gross_scalar - invested, 0.0)
    slot = remaining / slots_free
    # Cap a single entry at ~2x equal-weight. Without this, a FULL book (held == top_n) collapses
    # slots_free to max(0, 1) = 1, so the entire accumulated-cash gap is aimed at ONE order — which
    # sized every rotation buy at the whole cash balance (~16% of budget), breaching the 15%
    # single-order cap and rejecting the entire ranked list (362 alerts on 2026-08-13). The gap
    # instead deploys ~equal-weight per day; excess cash trickles in over subsequent days, which is
    # fine for a slow monthly book. 2x gives drift room without approaching the cap.
    equal_weight = nav * gross_scalar / max(cfg.top_n, 1)
    return min(slot, 2.0 * equal_weight)


def _size_shares(ticker: str, price_local: float, fx: float, tilts: pd.Series,
                 slot_usd: float, cash_usd: float) -> int:
    """Whole shares for one entry: its share of the remaining gap, inverse-vol tilted.

    The cash cap MUST be applied here, after the tilt. Capping `slot_usd` beforehand is not
    enough: the actual spend is slot x tilt, and tilt runs to ~1.7, so a single order could
    still overdraw. Long-only and funded, so cash is a hard constraint, not a preference.
    """
    tilt = float(tilts.get(ticker, 1.0))
    if tilt != tilt or tilt <= 0:
        tilt = 1.0
    # Floor the target at 0. `slot_usd` is normally non-negative, but it tracks NAV through
    # `2 * equal_weight`, so an account at NEGATIVE equity produces a negative slot and
    # `int(-1000 // 10)` is -100 — a BUY for -100 shares. check_order does reject a non-positive
    # quantity, but the sizer must not emit one in the first place: the guard is the second line
    # of defence, not the only one, and it can be disabled via cfg.use_risk_guard.
    target_usd = min(slot_usd * tilt, max(float(cash_usd), 0.0))
    denom = price_local * fx
    # ONE non-positive guard, deliberately. `slot_usd` is normally >= 0 but it tracks NAV via
    # `2 * equal_weight` in _slot_usd, so an account at NEGATIVE equity yields a negative slot and
    # `int(-1000 // 10)` is -100 — a BUY order for -100 shares. check_order does reject a
    # non-positive quantity, but the sizer must not emit one: the guard is the second line of
    # defence, not the only one, and it can be switched off via cfg.use_risk_guard.
    # Kept as a single check rather than belt-and-braces: overlapping guards that no test can
    # tell apart survive mutation testing and give false confidence.
    if target_usd <= 0 or not denom or denom <= 0 or denom != denom:
        return 0
    return int(target_usd // denom)


def _gross_scalar(vol: pd.Series, cfg: PaperConfig) -> float:
    """Portfolio vol-target factor (approx): scale gross so est book vol ≈ target.
    Book vol ≈ median constituent vol × ~0.6 diversification. Clipped to ≤1 (no leverage)."""
    med = float(np.nanmedian(vol.values)) if len(vol) else np.nan
    if not med or med != med:
        return 1.0
    est_book_vol = med * 0.6
    return float(np.clip(cfg.vol_target / est_book_vol, 0.0, 1.0))


def run_daily(state: PortfolioState, ranking: pd.Series, panels: dict, fx: dict,
              broker, cfg: PaperConfig, today: str) -> dict:
    """Run one weekday. Mutates `state`. Returns a summary for the email."""
    state.ensure_inception(today)
    adj = panels["adj"]
    ccy = panels["currency"]
    vol = _annual_vol(adj, cfg.vol_window)
    gross = _gross_scalar(vol, cfg)
    band = set(ranking.head(cfg.hold_n).index)          # names still "good enough" to hold
    marks = {t: float(adj[t].dropna().iloc[-1]) for t in adj.columns if adj[t].notna().any()}
    # PRIOR CLOSE for the price-sanity guard, taken from the SAME adjusted series as `marks`.
    # Using the same series matters: a legitimate split or dividend restates the whole history,
    # so it moves both numbers and cannot false-positive. What survives the comparison is a bad
    # print in the newest bar — which is precisely the one `_refresh_marks` just fetched live.
    # Without this the guard ran with prior=None and only its NaN branch was reachable, so the
    # split/feed-glitch check it exists for never executed.
    priors: dict[str, float] = {}
    for _t in adj.columns:
        _hist = adj[_t].dropna()
        if len(_hist) >= 2:
            priors[_t] = float(_hist.iloc[-2])
    # Tilts are normalised over the names we actually INTEND to hold (the top_n ranking), not
    # the whole universe — otherwise the mean is set by names we will never buy.
    tilts = _normalised_tilts(vol.reindex(ranking.head(cfg.top_n).index).dropna(), cfg)

    sells, holds, buys = [], [], []

    # ---- rotation: process positions whose clock is up ----
    for pos in state.clocks_up(today, cfg.hold_days):
        if pos.ticker in band:
            pos.entry_date = today                      # keep: restart the clock
            holds.append(pos.ticker)
        else:
            f = fx.get(pos.currency, pos.entry_fx)
            res = broker.order(pos.ticker, "SELL", int(pos.shares))
            if res["ok"]:
                px = res["fill_price"] or marks.get(pos.ticker, pos.entry_price)
                state.close_position(pos.ticker, px, f, today, reason="clock: dropped from band")
                sells.append(pos.ticker)

    # ---- margin ceiling (shared account) -------------------------------------------------
    # Checked BEFORE the buy loop and never around the sells: this may only block NEW risk.
    # Several strategies share one IB account because the trend overlay is a margin overlay on
    # common collateral, so the margin constraint is genuinely account-wide and one strategy can
    # legitimately be blocked by another's usage. Being blocked beats being liquidated.
    margin_scale = 1.0
    # Skipped entirely in dry-run: there is no connection by design, so "margin unavailable" is
    # expected rather than anomalous. Warning about it offline would fire on every dry run and
    # train you to ignore the one that matters — when it appears in a LIVE run.
    if (cfg.use_risk_guard and hasattr(broker, "margin_cushion")
            and not getattr(broker, "dry_run", False)):
        mu = broker.margin_cushion()
        lvl, margin_scale, why = liquidity_check(*(mu if mu else (float("nan"), 0.0)),
                                                limits=MarginLimits())
        if why:
            (logging.warning if lvl in ("derisk", "halt", "unknown") else logging.info)(
                "liquidity: %s", why)

    # ---- buys: fill toward top_n at ≤ max_new_buys_per_day, highest-ranked not held ----
    held = state.tickers
    room = cfg.top_n - len(held)
    n_buys = min(cfg.max_new_buys_per_day, max(room, 0))
    if margin_scale <= 0:
        n_buys = 0          # margin ceiling: hold what we have, open nothing new
    rejects: list[str] = []          # collected, then summarised in ONE alert (see below)
    if n_buys > 0:
        for t in ranking.index:
            if n_buys <= 0:
                break
            if t in held or t not in marks:
                continue
            price_local = marks[t]
            f = fx.get(ccy.get(t, "USD"), 1.0)
            # Recomputed per buy so the gap reflects fills already made this session.
            slot = _slot_usd(state, marks, fx, cfg, gross)
            shares = _size_shares(t, price_local, f, tilts, slot, state.cash)
            if shares <= 0:
                continue
            if cfg.use_risk_guard:
                lim = RiskLimits.for_equities(state.nav(marks, fx) or cfg.budget)
                ok_px = price_sane(t, price_local, priors.get(t), lim)
                if not ok_px:
                    logging.info("risk reject %s", ok_px.reason)
                    rejects.append(ok_px.reason)
                    continue
                pos = state.get(t)
                cur = (pos.shares * price_local * f) if pos else 0.0
                chk = check_order(t, "BUY", shares, price_local, 1.0 * f, lim,
                                  current_position_notional=cur,
                                  gross_notional=state.positions_value_usd(marks, fx))
                if not chk:
                    logging.info("risk reject %s", chk.reason)
                    rejects.append(chk.reason)
                    continue
            res = broker.order(t, "BUY", shares)
            if res["ok"]:
                entry = res["fill_price"] or price_local     # actual fill (RTH) or mark (queued/dry)
                state.open_position(Position(
                    ticker=t, shares=shares, entry_price=entry, entry_date=today,
                    entry_fx=f, currency=ccy.get(t, "USD")))
                buys.append((t, shares, round(shares * entry * f, 0)))
                held.add(t)
                n_buys -= 1

    # ONE alert for the whole run, not one per candidate: a systematic reject (e.g. a sizing/cap
    # mismatch) scans the entire ranked list, and 361 identical WARNINGs once buried every other
    # alert in the email (2026-08-13). The per-candidate detail stays at INFO in the log.
    if rejects:
        logging.warning("risk guard rejected %d buy candidate(s) this run (e.g. %s)",
                        len(rejects), rejects[0])

    logging.info("Daily %s: %d bought, %d sold, %d held-through (gross %.2f)",
                 today, len(buys), len(sells), len(holds), gross)
    return {"date": today, "buys": buys, "sells": sells, "holds": holds,
            "gross_scalar": gross, "marks": marks}
