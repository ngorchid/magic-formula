"""The daily weekday loop: staggered entry, clock-based rotation, inverse-vol sizing,
25% vol-target.

Cadence (matches the agreed spec):
  * Build-up: each weekday buy the single highest-ranked eligible name not yet held,
    until the book reaches `top_n` (~30).
  * Rotation: each position runs a 21-trading-day clock. When it's up, KEEP the name if
    it's still within the top-`hold_n` band (~45); otherwise SELL it. Freed slots refill
    at the same ~1 new buy/day pace.
Sizing: base $ = budget/top_n, tilted by inverse 63d volatility (clipped), then scaled by
a portfolio vol-target factor (gentle 25% → ~fully invested except in vol spikes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from paper.state import HOLD_DAYS, PortfolioState, Position


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


def _annual_vol(adj: pd.DataFrame, window: int) -> pd.Series:
    r = adj.pct_change(fill_method=None)
    return r.tail(window).std() * np.sqrt(252)


def _size_shares(ticker: str, price_local: float, fx: float, vol: pd.Series,
                 cfg: PaperConfig, gross_scalar: float) -> int:
    base = cfg.budget / cfg.top_n
    ref = float(np.nanmedian(vol.values)) if len(vol) else np.nan
    v = vol.get(ticker, np.nan)
    tilt = np.clip(ref / v, *cfg.inv_vol_clip) if (v and v == v and ref == ref) else 1.0
    target_usd = base * tilt * gross_scalar
    denom = price_local * fx
    return int(target_usd // denom) if denom and denom > 0 else 0


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

    sells, holds, buys = [], [], []

    # ---- rotation: process positions whose clock is up ----
    for pos in state.clocks_up(today, cfg.hold_days):
        if pos.ticker in band:
            pos.entry_date = today                      # keep: restart the clock
            holds.append(pos.ticker)
        else:
            px = marks.get(pos.ticker, pos.entry_price)
            f = fx.get(pos.currency, pos.entry_fx)
            if broker.order(pos.ticker, "SELL", int(pos.shares)):
                state.close_position(pos.ticker, px, f, today, reason="clock: dropped from band")
                sells.append(pos.ticker)

    # ---- buys: fill toward top_n at ≤ max_new_buys_per_day, highest-ranked not held ----
    held = state.tickers
    room = cfg.top_n - len(held)
    n_buys = min(cfg.max_new_buys_per_day, max(room, 0))
    if n_buys > 0:
        for t in ranking.index:
            if n_buys <= 0:
                break
            if t in held or t not in marks:
                continue
            price_local = marks[t]
            f = fx.get(ccy.get(t, "USD"), 1.0)
            shares = _size_shares(t, price_local, f, vol, cfg, gross)
            if shares <= 0:
                continue
            if broker.order(t, "BUY", shares):
                state.open_position(Position(
                    ticker=t, shares=shares, entry_price=price_local, entry_date=today,
                    entry_fx=f, currency=ccy.get(t, "USD")))
                buys.append((t, shares, round(shares * price_local * f, 0)))
                held.add(t)
                n_buys -= 1

    logging.info("Daily %s: %d bought, %d sold, %d held-through (gross %.2f)",
                 today, len(buys), len(sells), len(holds), gross)
    return {"date": today, "buys": buys, "sells": sells, "holds": holds,
            "gross_scalar": gross, "marks": marks}
