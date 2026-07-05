"""Overfitting-aware performance statistics (Bailey & López de Prado).

A raw Sharpe ratio overstates how good a strategy is for three reasons these address:
 - short samples (the estimate is noisy),
 - non-normal returns (fat tails / skew make a high Sharpe less trustworthy),
 - **multiple testing** — if you try N configurations, the best one's Sharpe is inflated
   by luck alone.

`probabilistic_sharpe_ratio` gives P(true Sharpe > benchmark) accounting for sample
length, skew, and kurtosis. `deflated_sharpe_ratio` sets that benchmark to the Sharpe
you'd *expect to beat by chance* given N trials — so it directly haircuts for the
fishing we did. DSR > 0.95 ≈ "real after accounting for the search."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

_EULER = 0.5772156649015329


def _daily_sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > `benchmark_sr`), adjusting for sample length, skew,
    and kurtosis. `benchmark_sr` is in per-period (e.g. daily) units."""
    r = pd.Series(returns).dropna().values
    n = len(r)
    if n < 10:
        return float("nan")
    sr = _daily_sharpe(r)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # non-excess (normal = 3)
    denom = np.sqrt(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2)
    return float(norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe under the null, from `n_trials` independent
    trials whose Sharpe estimates have variance `sr_variance` (per-period units)."""
    if n_trials < 2:
        return 0.0
    g = _EULER
    return float(np.sqrt(sr_variance) * (
        (1 - g) * norm.ppf(1 - 1.0 / n_trials)
        + g * norm.ppf(1 - 1.0 / (n_trials * np.e))
    ))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, sr_variance: float
) -> tuple[float, float]:
    """Deflated Sharpe Ratio: PSR evaluated against the expected-max-from-N-trials
    benchmark. Returns (DSR probability, benchmark per-period Sharpe)."""
    sr_star = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr_star), sr_star
