"""Drawdown controls.

`in_drawdown_control` returns a boolean series — True on dates the strategy should
de-risk because the current drawdown breaches `threshold`. Wire this into sizing
once a live trading harness exists.
"""
from __future__ import annotations

import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    return curve / curve.cummax() - 1.0


def in_drawdown_control(returns: pd.Series, threshold: float = -0.10) -> pd.Series:
    return drawdown_series(returns) <= threshold
