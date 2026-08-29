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
from risk_guard import (MarginLimits, RiskLimits, check_order, liquidity_check,
                        price_sane, stale_columns)


@dataclass
class PaperConfig:
    budget: float = 100_000.0
    top_n: int = 30
    hold_n: int = 45           # no-trade band
    max_new_buys_per_day: int = 1
    vol_target: float = 0.25   # annualized portfolio vol target
    vol_window: int = 63       # days for per-name realised vol
    # Covariance vol-targeting, ported from the trend overlay 2026-08-29. Correlations are more
    # stable than vols, so they get a longer window; corr_weight is the weight on the SAMPLE
    # correlation, the remainder going to a constant-correlation target. The trend repo measured
    # near-raw sample correlation (0.90) to beat heavy shrinkage, monotone in corr_weight and
    # holding in both sub-periods. Sigma is only ever used as a quadratic form (w' Sigma w) and
    # never inverted, so it does not need to be well-conditioned.
    corr_window: int = 252
    corr_weight: float = 0.90
    # Fallback only: book vol ~ median constituent vol x this, used when there is too little
    # history to form a correlation matrix. Measured mean over 2012-2026 is 0.628, but it
    # reaches 0.864 in March 2020 -- which is precisely why the covariance estimate exists.
    diversification: float = 0.60
    # EQUAL WEIGHT as of 2026-08-28: (1.0, 1.0) makes every tilt exactly 1.0, so the code
    # path stays intact and this is reversible by editing one line. Measured on the deployed
    # factor set, at the LIVE $50k book size with IB's $1.00/order floor, inverse-vol tilting
    # loses to equal weight on ALL THREE test universes and in every implementation tried:
    #   scheme                          S&P500 PIT     S&P1500     small-cap
    #   live_tilt   clip (0.5,2.0)      -2.08%/yr      -3.91%      -3.97%   (t -2.6 to -5.0)
    #   tilt_tight  clip (0.7,1.5)      -1.72%         -3.10%      -3.21%
    #   tilt_entry  fixed at entry      -1.85%         -4.73%      -3.74%
    #   tilt_band   20% no-trade band   -2.04%         -4.09%      -3.97%
    # 15 cells, no wins. Crucially `tilt_entry` cut turnover 4.11x -> 3.46x (killing 73% of
    # the pure weight-drift churn) and performance did NOT recover -- so the drag is the
    # WEIGHTING, not the turnover, and no implementation fixes it. See
    # algo_trading/scripts/magic_weighting_universe_lab.py.
    inv_vol_clip: tuple[float, float] = (1.0, 1.0)
    # Cap on a single entry, as a multiple of equal weight. 1.0 is what actually CONVERGES a
    # book that is currently vol-weighted: gap-based sizing recycles existing position sizes
    # forever (simulated 180 rotations at 2.0x -> size spread unchanged at 4.0x), because
    # selling a $2,700 name frees $2,700 and the replacement takes the whole gap. At 1.0x the
    # book converges to exact equality through normal rotation, no forced rebalancing and no
    # extra turnover (~4 months at 3.1x turnover). Also strictly tighter than the 2.0x that
    # was protecting against the slots_free->1 collapse, so no regression there.
    max_entry_mult: float = 1.0
    hold_days: int = HOLD_DAYS
    # Independent pre-trade guard. It re-derives every order from the budget rather than trusting
    # the sizing above, because a guard that reuses the strategy's arithmetic cannot catch the
    # strategy's own bug — and this file had exactly such a bug (gross reaching ~2x budget) two
    # days before it was wired in. Rejections are LOGGED and skipped, never silent.
    use_risk_guard: bool = True
    # FX cash sweep. IB does not auto-convert, so a European buy leaves a NEGATIVE balance in
    # that currency, financed at the first tier (IBKR UK, 2026-08-28: EUR 3.697%, GBP 5.227%,
    # CHF 1.500%, SEK 3.154%, DKK 4.796%, NOK 5.636%). Net of the USD credit forgone that is
    # ~1.64% of NAV a year on a half-European $50k book; sweeping costs ~0.29%.
    #
    # $500 is a deliberately FLAT threshold. The true break-even is rate-dependent — at a
    # one-month hold it is $426 for NOK, $459 GBP, $500 DKK, $649 EUR, $761 SEK and $1,600 for
    # CHF — so $500 is below break-even for EUR, SEK and especially CHF on a ONE-MONTH view.
    # It is still right, for two reasons. Sweeping nets the AGGREGATE balance, not a position,
    # so what gets converted persists for as long as any exposure to that currency does, not
    # 21 days; at a three-month horizon every break-even here falls under $535. And a
    # per-currency threshold table would need maintaining against floating benchmark rates to
    # save a few dollars a year. Raise it if the FX commission is not IB's $2 — LYNX marks
    # this up, and it is the input the whole calculation is most sensitive to.
    fx_sweep_min_usd: float = 500.0
    fx_sweep: bool = True


def plan_fx_sweep(balances: dict[str, float], fx: dict[str, float],
                  min_usd: float = 500.0) -> dict[str, float]:
    """{ccy: units to trade} to net every non-USD cash balance to zero.

    Positive = acquire that currency (covering a short financed at the debit rate); negative =
    sell an idle foreign balance back to USD. Balances worth less than `min_usd` are left
    alone so the $2 FX commission never dominates the interest it saves.

    Pure function of (balances, rates) so the POLICY is testable without a broker: the sweep
    places real orders, and a bug here spends money rather than merely mis-reporting.
    """
    plan: dict[str, float] = {}
    for ccy, bal in balances.items():
        if ccy == "USD" or not bal:
            continue
        rate = fx.get(ccy)
        # An unknown rate means the USD size of this balance is unknown, so the threshold
        # cannot be applied. Skipping is the safe direction: not sweeping costs interest,
        # sweeping the wrong size trades real money.
        if not rate or rate <= 0 or not np.isfinite(rate):
            logging.warning("fx sweep: no rate for %s, balance %.2f left unswept", ccy, bal)
            continue
        if abs(bal * rate) < min_usd:
            continue
        plan[ccy] = -bal          # trade the negative of the balance to reach zero
    return plan


def run_fx_sweep(broker, balances: dict[str, float], fx: dict[str, float],
                 cfg: PaperConfig) -> list[tuple[str, float, str]]:
    """Execute `plan_fx_sweep`. Returns [(ccy, units, status)] for the report."""
    if not cfg.fx_sweep:
        return []
    if not balances:
        # Distinguishes "cannot read balances" from "nothing to sweep": cash_balances()
        # returns {} on failure AND in dry-run, and silently doing nothing on a live failure
        # would let the interest accrue unnoticed.
        logging.info("fx sweep: no balances available (dry-run or account read failed)")
        return []
    out = []
    for ccy, units in plan_fx_sweep(balances, fx, cfg.fx_sweep_min_usd).items():
        res = broker.convert_fx(ccy, units, fx.get(ccy, 0.0))
        out.append((ccy, units, res.get("status", "?")))
    return out


def price_units_agree(ib_price: float | None, mark: float | None,
                      tol: float = 5.0) -> bool:
    """Do the BROKER's price and the DATA FEED's mark use the same units?

    Sizing divides a USD slot by `mark` (yfinance) but the position is then recorded at the
    IB fill, and the two are only comparable if they are denominated identically. That is not
    guaranteed. The London Stock Exchange quotes in PENCE while the IB contract currency says
    GBP; IB's own `ContractDetails.priceMagnifier` exists precisely to reconcile execution
    prices with market data ("allows execution and strike prices to be reported consistently
    with market data, historical data and the order price"), which means the discrepancy is
    real enough that IB ships a field for it. `priceMagnifier` only makes IB self-consistent,
    though — it says nothing about agreeing with yfinance, which is the comparison that
    actually matters here.

    A factor-of-100 disagreement does not fail loudly. It makes `cost_usd` 100x too small, so
    cash barely moves, the sizer believes it has budget left and keeps buying. This returns
    False on any disagreement beyond `tol`, which catches the whole class rather than pence
    specifically — the same failure would arrive silently with any venue quoted in a minor
    unit (ZAc, ILA) if one were ever added.

    A generous 5x tolerance: intraday drift and a stale bar are normal, a unit error is 100x.
    """
    if not ib_price or not mark or ib_price <= 0 or mark <= 0:
        return True                     # nothing to compare — other guards own that case
    r = ib_price / mark
    return (1.0 / tol) <= r <= tol


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
    return min(slot, cfg.max_entry_mult * equal_weight)


def _size_shares(ticker: str, price_local: float, fx: float, tilts: pd.Series,
                 slot_usd: float, cash_usd: float) -> int:
    """Whole shares for one entry: its share of the remaining gap, inverse-vol tilted.

    The cash cap MUST be applied here, after the tilt. Capping `slot_usd` beforehand is not
    enough: the actual spend is slot x tilt, so with any tilt > 1 a single order could still
    overdraw. Long-only and funded, so cash is a hard constraint, not a preference.
    (Tilt is currently pinned to 1.0 by `inv_vol_clip=(1.0,1.0)`, but this stays correct if
    it is ever re-enabled.)
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


def _book_vol(rets: pd.DataFrame, names: list[str], vol: pd.Series,
              cfg: PaperConfig) -> float | None:
    """Annualised book vol from the covariance: sqrt(w' Sigma w), Sigma = D R D.

    Ported from `trend_overlay/execution.py` 2026-08-29. The previous estimate,
    `median constituent vol x 0.60`, uses a CONSTANT for diversification, and a constant cannot
    see correlations rising. Measured over 2012-2026 the true ratio of book vol to median
    constituent vol averages 0.628 -- so 0.60 is fine on average -- but it reaches **0.864 in
    March 2020**, where actual book vol ran ~44% above the estimate and the target therefore
    admitted far more risk than intended at exactly the peak. That is the same defect the trend
    overlay carried until 2026-08-08, where assuming uncorrelated markets left realised vol at
    12-14% against a 10% target.

    Equal weights, because that is what the sleeve now deploys (`inv_vol_clip` pinned to
    (1.0, 1.0) since 2026-08-28). Returns None when there is too little history or the matrix is
    unusable, so the caller can fall back rather than size off a bad number.
    """
    names = [t for t in names if t in rets.columns and t in vol.index]
    n = len(names)
    if n < 2 or not cfg.corr_weight:
        return None
    win = rets[names].tail(cfg.corr_window).dropna(how="all")
    if len(win) < cfg.corr_window:
        return None
    v = np.nan_to_num(vol.reindex(names).values.astype(float))
    if not np.isfinite(v).all() or (v <= 0).all():
        return None
    R = win.corr().values
    if R.shape != (n, n) or not np.isfinite(R).all():
        return None
    off = ~np.eye(n, dtype=bool)
    target = np.full((n, n), float(R[off].mean()))     # constant-correlation target
    np.fill_diagonal(target, 1.0)
    R = cfg.corr_weight * R + (1.0 - cfg.corr_weight) * target
    w = np.repeat(1.0 / n, n)
    D = np.diag(v)
    var = float(w @ (D @ R @ D) @ w)
    return float(np.sqrt(var)) if var > 0 else None


def _gross_scalar(vol: pd.Series, cfg: PaperConfig, rets: pd.DataFrame | None = None,
                  names: list[str] | None = None) -> float:
    """Portfolio vol-target factor: scale gross so estimated book vol ≈ target.

    Prefers the COVARIANCE estimate (see `_book_vol`); falls back to
    `median constituent vol x cfg.diversification` when history is too short to form a
    correlation matrix. Clipped to ≤1, so it can only de-risk and never levers up.
    """
    med = float(np.nanmedian(vol.values)) if len(vol) else np.nan
    if not med or med != med:
        return 1.0
    est_book_vol = None
    if rets is not None and names:
        est_book_vol = _book_vol(rets, names, vol, cfg)
    if est_book_vol is None or est_book_vol <= 0:
        est_book_vol = med * cfg.diversification
    return float(np.clip(cfg.vol_target / est_book_vol, 0.0, 1.0))


def run_daily(state: PortfolioState, ranking: pd.Series, panels: dict, fx: dict,
              broker, cfg: PaperConfig, today: str) -> dict:
    """Run one weekday. Mutates `state`. Returns a summary for the email."""
    state.ensure_inception(today, cfg.budget)
    adj = panels["adj"]
    ccy = panels["currency"]
    vol = _annual_vol(adj, cfg.vol_window)
    # Estimate on the TARGET book (top_n by rank), not the whole candidate universe: the vol
    # being targeted is the portfolio's, and the universe median is not the portfolio's median.
    _target_names = [t for t in ranking.head(cfg.top_n).index if t in adj.columns]
    gross = _gross_scalar(vol, cfg, rets=adj.pct_change(fill_method=None), names=_target_names)
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
    # PER-TICKER STALENESS. `data_fresh` upstream inspects only the panel INDEX, so a single
    # ticker whose feed dies or freezes leaves the panel looking current: `marks` still holds a
    # valid-looking price, `price_sane` passes because the price is a perfectly good number, and
    # sizing divides by it. The result is a real order at a price that no longer exists.
    #
    # BUYS ONLY. A stale name is excluded from new purchases; sells are deliberately untouched,
    # in line with the framework-wide rule that no guard may block a close. Selling at a stale
    # mark is imperfect, but refusing to sell traps the position — and the exit here is
    # clock-driven with no deadline, so a delayed sell costs nothing while a blocked one holds
    # risk we have decided to shed.
    stale_px, _ = stale_columns(adj, pd.Timestamp(today),
                                RiskLimits.for_equities(cfg.budget))
    # SCOPE THE LOG, not the check. Only held names and the top of the ranking get a daily price
    # refresh (`_refresh_marks`); the rest of the ~500-name panel is monthly-cached and is stale
    # BY DESIGN. Flagging all of it warned on ~400 names we never trade — which is how an alert
    # channel is trained to be ignored. The skip below still applies to every candidate, since a
    # name outside the refresh set genuinely does have a month-old price.
    _relevant = (set(ranking.head(max(cfg.hold_n, cfg.top_n)).index) | state.tickers)
    _noisy = sorted(set(stale_px) & _relevant)
    if _noisy:
        logging.warning("stale prices on %d tradeable name(s): %s — excluded from BUYS "
                        "(sells unaffected)", len(_noisy),
                        ", ".join(f"{t} {stale_px[t]}d" for t in _noisy[:10])
                        + (" ..." if len(_noisy) > 10 else ""))
    _held_stale = sorted(set(stale_px) & state.tickers)
    if _held_stale:
        logging.warning("HELD positions marked on stale prices: %s — NAV is approximate",
                        ", ".join(_held_stale))
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
            if t in stale_px:
                logging.info("skip %s — stale price (%dd since it last moved)", t, stale_px[t])
                continue
            price_local = marks[t]
            f = fx.get(ccy.get(t, "USD"), 1.0)
            # Recomputed per buy so the gap reflects fills already made this session.
            slot = _slot_usd(state, marks, fx, cfg, gross)
            shares = _size_shares(t, price_local, f, tilts, slot, state.cash)
            if shares <= 0:
                continue
            # PRE-trade unit check, and it must stay pre-trade: once a fill comes back the
            # trade is done and there is no safe entry_price to record. Only meaningful for
            # non-USD names, and only when the broker can actually quote (never in dry-run).
            if ccy.get(t, "USD") != "USD" and hasattr(broker, "price"):
                ib_px = broker.price(t)
                if not price_units_agree(ib_px, price_local):
                    msg = (f"{t}: broker price {ib_px} vs mark {price_local} "
                           f"({ib_px / price_local:.3g}x) — UNIT MISMATCH, not traded")
                    logging.error(msg)
                    rejects.append(msg)
                    continue
            # Independent pre-trade guard — runs for EVERY name, not just non-USD. It was briefly
            # nested inside the unit-check branch above (2026-08-28 refactor), which silently let
            # US-equity buys — the bulk of the book — reach the broker unguarded. The guard is the
            # second line of defence against the strategy's own sizing bug and must not depend on
            # currency; keep it under use_risk_guard at the loop level.
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

    # ---- FX cash sweep -------------------------------------------------------------------
    # LAST, after every equity trade, so it nets the day's true end-state rather than
    # converting for a buy and back for a sell. It deliberately does NOT run before the buy
    # loop: a sweep that fired first would convert USD the buys then need back again.
    swept = run_fx_sweep(broker, broker.cash_balances() if hasattr(broker, "cash_balances")
                         else {}, fx, cfg)
    if swept:
        logging.info("fx sweep: %s", ", ".join(f"{c} {u:+,.0f} ({s})" for c, u, s in swept))

    logging.info("Daily %s: %d bought, %d sold, %d held-through (gross %.2f)",
                 today, len(buys), len(sells), len(holds), gross)
    return {"date": today, "buys": buys, "sells": sells, "holds": holds,
            "gross_scalar": gross, "marks": marks, "fx_sweep": swept}
