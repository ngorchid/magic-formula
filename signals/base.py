"""Signal base class.

A signal is a function panel-of-features -> panel-of-scores `[date × ticker]`.
Scores should be cross-sectionally meaningful (rankable across tickers on the same
date). Cross-sectional standardisation lives in `combination.processing`, not here —
each signal returns its raw form so we can study its distribution before z-scoring.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Signal(ABC):
    name: str

    @abstractmethod
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Return a wide DataFrame `[date × ticker]` of raw signal values."""
        ...
