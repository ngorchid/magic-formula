"""Backtest performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / sd)


def max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def annualized_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() ** (TRADING_DAYS / max(len(returns), 1)) - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def summary_stats(returns: pd.Series) -> dict[str, float]:
    return {
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": float((returns > 0).mean()),
    }
