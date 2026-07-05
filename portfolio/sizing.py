"""Simple position sizing — used by the baseline backtest before the cvxpy optimizer
is in place.
"""
from __future__ import annotations

import pandas as pd


def equal_weight_long_short(scores_row: pd.Series, top_quantile: float = 0.2) -> pd.Series:
    """Equal-weighted dollar-neutral long-short from a cross-section of scores."""
    s = scores_row.dropna()
    if len(s) < 10:
        return pd.Series(0.0, index=scores_row.index)
    n = max(int(len(s) * top_quantile), 1)
    ranked = s.sort_values()
    w = pd.Series(0.0, index=scores_row.index)
    w.loc[ranked.index[-n:]] = 0.5 / n
    w.loc[ranked.index[:n]] = -0.5 / n
    return w
