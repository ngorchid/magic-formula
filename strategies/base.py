"""Common contract for return streams.

A Stream is an end-to-end pipeline (signals → portfolio construction → P&L). The
meta-allocator only sees the resulting `StreamResult`, so adding or removing whole
streams stays a one-line config change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass
class StreamResult:
    name: str
    weights: pd.DataFrame                # [date × instrument], target weights as fraction of notional
    gross_returns: pd.Series             # daily, before costs
    net_returns: pd.Series               # daily, after costs
    turnover: pd.Series                  # daily, sum of |Δw|
    costs: pd.Series                     # daily, fraction of notional
    diagnostics: dict = field(default_factory=dict)


class Stream(Protocol):
    name: str

    def run(self) -> StreamResult: ...
