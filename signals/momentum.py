"""Momentum signals.

Canonical Jegadeesh & Titman (1993) / Asness, Moskowitz & Pedersen (2013):
the past 12-month return, *skipping the most recent month* to dodge short-term reversal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """12-1 month price momentum on a wide `[date × ticker]` price panel.

    Implementation: ``P_{t-21} / P_{t-252} - 1`` (≈ 12 months back to 1 month back),
    using trading-day offsets so it stays calendar-agnostic.
    """
    if prices.empty:
        return prices
    p = prices.sort_index()
    return p.shift(21) / p.shift(252) - 1.0


def residual_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Idiosyncratic (residual) momentum — Blitz, Huij & Martens (2011).

    Strip the common market move from each day's cross-section (subtract the equal-
    weight mean return), then take the information ratio of the residual returns over
    the past 12 months skipping the last month: ``mean(resid)/std(resid)``. Higher =
    stronger idiosyncratic uptrend. Empirically ~2× the Sharpe of raw momentum with
    lower turnover and far smaller crash risk; the residual-vol scaling removes the
    volatility bias baked into plain momentum.
    """
    if prices.empty:
        return prices
    p = prices.sort_index()
    rets = p.pct_change(fill_method=None)
    resid = rets.sub(rets.mean(axis=1), axis=0)  # remove equal-weight market each day
    window = lookback - skip
    mp = max(window // 2, 20)
    mu = resid.shift(skip).rolling(window, min_periods=mp).mean()
    sd = resid.shift(skip).rolling(window, min_periods=mp).std().replace(0.0, np.nan)
    return mu / sd
