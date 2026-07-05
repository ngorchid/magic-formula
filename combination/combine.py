"""IC-weighted signal combination.

Each signal's weight is proportional to its rolling rank-IC (Spearman correlation
between the signal at date t and the forward return realized over t..t+h). This is the
straightforward Grinold/Kahn idea — weight by predictive power, not by feel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_rank_ic(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    window: int = 252,
    lag: int = 0,
    standardize: bool = False,
) -> pd.Series:
    """Rolling cross-sectional Spearman IC over `window` days.

    Default returns the rolling **mean** IC. With ``standardize=True`` returns the
    **ICIR** (mean IC / std IC over the window) — this rewards *stable* predictors over
    erratic ones with the same average IC.

    `fwd_returns` realized over t..t+h means the IC at date t depends on data through
    t+h. Pass ``lag=h`` so the returned weight at t uses only information available at
    t — without it, IC-weighting silently looks ahead by the forward-return horizon.
    """
    common = signal.index.intersection(fwd_returns.index)
    s = signal.loc[common].rank(axis=1)
    r = fwd_returns.loc[common].rank(axis=1)
    daily_ic = s.corrwith(r, axis=1)
    roll = daily_ic.rolling(window)
    weight = roll.mean() / roll.std() if standardize else roll.mean()
    return weight.shift(lag)


def ic_weighted_combine(
    signals: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    window: int = 252,
    lag: int = 0,
    standardize: bool = False,
    return_weights: bool = False,
):
    """Combine standardised signals weighted by their rolling IC.

    Inputs should already be cross-sectionally z-scored. Negative-IC signals get
    negative weights — useful when a signal turns out to mean-revert relative to its
    expected sign. `lag` is forwarded to `rolling_rank_ic` to avoid look-ahead.

    Note: the normaliser is a per-date sum of |weight| across *all* signals, so names
    missing some signals (e.g. fundamentals coverage gaps) are shrunk toward zero
    rather than dropped — acceptable, but it tilts selection toward fully-covered names.

    With `return_weights=True` returns ``(combined, weights_df)`` where weights_df is
    ``[date × signal]`` of the lagged rolling IC actually applied.
    """
    if not signals:
        raise ValueError("no signals provided")
    weights = {name: rolling_rank_ic(s, fwd_returns, window, lag, standardize)
               for name, s in signals.items()}
    combined = None
    weight_sum = None
    for name, s in signals.items():
        w = weights[name].reindex(s.index).fillna(0.0)
        contrib = s.mul(w, axis=0)
        combined = contrib if combined is None else combined.add(contrib, fill_value=0.0)
        abs_w = w.abs()
        weight_sum = abs_w if weight_sum is None else weight_sum.add(abs_w, fill_value=0.0)
    weight_sum = weight_sum.replace(0.0, np.nan)
    combined = combined.div(weight_sum, axis=0)
    if return_weights:
        return combined, pd.DataFrame({n: w for n, w in weights.items()})
    return combined
