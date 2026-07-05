"""Portfolio exposure monitors."""
from __future__ import annotations

import pandas as pd


def gross_leverage(weights: pd.DataFrame) -> pd.Series:
    return weights.abs().sum(axis=1)


def net_exposure(weights: pd.DataFrame) -> pd.Series:
    return weights.sum(axis=1)
