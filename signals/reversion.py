"""Short-term reversion signal family."""
from __future__ import annotations

import pandas as pd


def short_term_reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Sign-flipped recent return. Losers over the past `lookback` days tend to bounce.

    Returns a `[date × ticker]` panel where larger values mean *more attractive long*
    (so the same downstream pipeline as momentum can consume it).
    """
    if prices.empty:
        return prices
    p = prices.sort_index()
    return -(p / p.shift(lookback) - 1.0)
