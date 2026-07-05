"""Growth signal family — is the business actually getting bigger?

Value tells you a stock is cheap; growth tells you whether the fundamentals behind it
are expanding or shrinking (the classic "cheap vs. value trap" distinction). Each signal
is a year-over-year change of a *TTM* line item from the PIT fundamentals dict, oriented
so **larger = faster growth = more attractive**, matching the winsorize→z-score→combine
pipeline used by the other families.

Revenue is always positive, so plain YoY %% is well posed. EBIT and free cash flow can be
negative or cross zero, where plain %% growth explodes/flips sign — for those we use a
symmetric change ``(x_t − x_{t−1y}) / (|x_t| + |x_{t−1y}|)`` ∈ (−1, 1), which is monotone
in true growth and robust to sign, and (being rank-combined downstream) needs no rescaling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .quality import free_cash_flow

Fundamentals = dict[str, pd.DataFrame]

_YEAR = 252  # trading days ≈ 1 calendar year


def _yoy(panel: pd.DataFrame, periods: int = _YEAR) -> pd.DataFrame:
    """Plain year-over-year growth rate (for strictly-positive quantities)."""
    prev = panel.shift(periods)
    return panel.div(prev.where(prev > 0)) - 1.0


def _yoy_symmetric(panel: pd.DataFrame, periods: int = _YEAR) -> pd.DataFrame:
    """Sign-robust YoY change in (−1, 1); safe when the level can be ≤ 0."""
    prev = panel.shift(periods)
    denom = panel.abs() + prev.abs()
    return (panel - prev).div(denom.where(denom > 0))


def multi_year_growth(panel: pd.DataFrame, n_years: int = 2, reduce: str = "mean",
                      symmetric: bool = False, periods: int = _YEAR) -> pd.DataFrame:
    """Combine the last `n_years` of YoY growth into one persistence-aware signal.

    Computes YoY growth for each of the trailing years — t-1→t, t-2→t-1, … — then:
      * ``reduce="mean"`` averages them (smooths single-year noise), or
      * ``reduce="min"`` takes the *worst* year, so a name must have grown in **every**
        one of the last `n_years` to score high — a direct encoding of consistency.
    Requires all `n_years` present (NaN otherwise), so it needs ~`n_years`+1 yrs history.
    """
    gs = []
    for lag in range(n_years):
        cur, prev = panel.shift(lag * periods), panel.shift((lag + 1) * periods)
        if symmetric:
            denom = cur.abs() + prev.abs()
            gs.append((cur - prev).div(denom.where(denom > 0)))
        else:
            gs.append(cur.div(prev.where(prev > 0)) - 1.0)
    if reduce == "min":
        out = gs[0]
        for g in gs[1:]:
            out = pd.DataFrame(np.minimum(out.values, g.values), index=out.index, columns=out.columns)
        return out
    return sum(gs) / len(gs)


def revenue_growth(f: Fundamentals) -> pd.DataFrame:
    """YoY growth of trailing-twelve-month revenue. Higher = faster top-line growth."""
    return _yoy(f["revenue"])


def ebit_growth(f: Fundamentals) -> pd.DataFrame:
    """YoY growth of TTM operating income (EBIT), sign-robust. Higher = better."""
    return _yoy_symmetric(f["operating_income"])


def fcf_growth(f: Fundamentals) -> pd.DataFrame:
    """YoY growth of TTM free cash flow (OCF − capex), sign-robust. Higher = better."""
    return _yoy_symmetric(free_cash_flow(f))


GROWTH_SIGNALS = {
    "revenue_growth": revenue_growth,
    "ebit_growth": ebit_growth,
    "fcf_growth": fcf_growth,
}
