"""Volatility-based signals (the low-volatility anomaly).

Low-risk stocks have historically delivered higher risk-adjusted returns than high-risk
stocks (Haugen-Baker; Frazzini-Pedersen betting-against-beta). We express it as *total*
realised volatility rather than beta, because the equity_mn pipeline already regresses
out market beta when neutralising — a pure −beta signal would be largely stripped, while
the low-(total-)vol tilt survives.
"""
from __future__ import annotations

import pandas as pd


def low_volatility(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Sign-flipped trailing realised volatility on a `[date × ticker]` price panel.

    Larger value = lower past volatility = more attractive long, matching the
    "higher is better" convention of the other signals.
    """
    if prices.empty:
        return prices
    p = prices.sort_index()
    rets = p.pct_change(fill_method=None)
    vol = rets.rolling(lookback, min_periods=max(lookback // 2, 20)).std()
    return -vol
