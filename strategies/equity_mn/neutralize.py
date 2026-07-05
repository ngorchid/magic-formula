"""Cross-sectional neutralisation of a signal panel.

Signal-level neutralisation is a *first cut* — the proper place to enforce neutrality
is the portfolio optimizer (constraints on `B' w`). Here we approximate by removing
the linear projection of the signal onto the factors, so the resulting quintile
portfolio has small (not zero) factor exposure. The cvxpy optimizer will tighten
this once it lands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _row_regress_residuals(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """For each row, residualise y against x (single-regressor, no intercept).

    Both inputs are `[date × ticker]`; the slope is computed cross-sectionally each
    date. Equivalent to: ``y_t - (sum(y_t*x_t) / sum(x_t^2)) * x_t``.
    """
    aligned_y, aligned_x = y.align(x, join="inner")
    mask = aligned_y.notna() & aligned_x.notna()
    yx = (aligned_y * aligned_x).where(mask)
    xx = (aligned_x ** 2).where(mask)
    slope = yx.sum(axis=1) / xx.sum(axis=1).replace(0.0, np.nan)
    return aligned_y - aligned_x.mul(slope, axis=0)


def neutralize(
    panel: pd.DataFrame,
    betas: pd.DataFrame | None = None,
    sectors: pd.Series | None = None,
) -> pd.DataFrame:
    """Beta- and sector-neutralise a signal panel cross-sectionally."""
    out = panel.copy()
    # Demean each row first (removes any cross-sectional intercept)
    out = out.sub(out.mean(axis=1), axis=0)

    if betas is not None:
        out = _row_regress_residuals(out, betas.reindex_like(out))

    if sectors is not None:
        sectors = sectors.reindex(out.columns)
        valid = sectors.dropna()
        if not valid.empty:
            # Demean within sector, per date. Transpose to group along the ticker axis,
            # then transpose back.
            sub = out[valid.index]
            sector_means = sub.T.groupby(valid.values).transform("mean").T
            out.loc[:, valid.index] = sub.sub(sector_means)
    return out


def rolling_beta(returns: pd.DataFrame, benchmark_returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Rolling regression beta of each column of `returns` against `benchmark_returns`."""
    rb, _ = returns.align(benchmark_returns, axis=0, join="inner")
    bench = benchmark_returns.reindex(rb.index)
    cov = rb.rolling(window).cov(bench)
    var = bench.rolling(window).var()
    return cov.div(var, axis=0)
