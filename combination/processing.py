"""Cross-sectional signal processing primitives.

Operate row-wise (i.e. date-wise) on `[date × ticker]` panels so the resulting scores
are comparable across the cross-section on each rebalance date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(panel: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row to its [lower, upper] cross-sectional quantiles."""
    lo = panel.quantile(lower, axis=1)
    hi = panel.quantile(upper, axis=1)
    return panel.clip(lower=lo, upper=hi, axis=0)


def cs_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (row-wise) z-score. NaNs preserved."""
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1).replace(0.0, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)


def decay(panel: pd.DataFrame, halflife: int) -> pd.DataFrame:
    """Exponentially-weighted smoothing along the time axis (per ticker)."""
    return panel.ewm(halflife=halflife, adjust=False).mean()
