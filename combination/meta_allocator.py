"""Portfolio-level allocation across return streams.

Equal-risk-contribution is the standard baseline: weight each stream inversely to
its realised vol so each contributes the same risk to the blended portfolio. Smarter
allocators (risk parity with correlation, Black-Litterman across streams, regime-
conditional weighting) drop in at the same interface.

Two implementation details matter a lot in practice and were getting Sharpe badly
wrong before:
  1. **Trim to the all-live window.** A stream that is still in warmup (flat, ≈0
     returns) has ≈0 rolling vol → near-infinite inverse-vol weight, so the blend
     piles into the *flat* stream and misses the others. Blend only once every stream
     is actually trading.
  2. **Stable weights + a gentle vol target.** Short-window inverse vol and an
     aggressive short-window vol-target whipsaw the weights and destroy Sharpe. Use a
     longer window, a per-stream vol floor, and a tightly-clipped long-window target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AllocationResult:
    blended_returns: pd.Series
    weights: pd.DataFrame                # [date × stream]
    stream_returns: pd.DataFrame         # [date × stream] aligned, raw
    diagnostics: dict


def equal_risk_allocate(
    stream_returns: dict[str, pd.Series],
    vol_window: int = 126,
    target_portfolio_vol: float | None = 0.08,
    lag: int = 1,
    vol_floor_q: float = 0.10,
) -> AllocationResult:
    """Inverse-vol weighting across streams, with optional portfolio-vol target.

    Weights are lagged so today's realised vol never weights today's return. The blend
    is restricted to the window where every stream is live, and each stream's rolling
    vol is floored at a low quantile of its own history so a transiently quiet stream
    cannot dominate.
    """
    if not stream_returns:
        raise ValueError("no stream returns provided")
    df = pd.DataFrame(stream_returns)

    # Trim to the first date on which every stream is actually trading (non-trivial).
    live = (df.abs() > 1e-9) & df.notna()
    all_live = live.all(axis=1)
    if all_live.any():
        df = df.loc[all_live.idxmax():]
    df = df.fillna(0.0)
    n = df.shape[1]

    rolling_vol = df.rolling(vol_window, min_periods=max(vol_window // 4, 20)).std()
    if vol_floor_q:
        rolling_vol = rolling_vol.clip(lower=rolling_vol.quantile(vol_floor_q), axis=1)
    inv_vol = 1.0 / rolling_vol.replace(0.0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.shift(lag).fillna(1.0 / n)

    blended = (weights * df).sum(axis=1)

    if target_portfolio_vol is not None:
        # Gentle target on a long, stable window; tight clip avoids leverage whipsaw.
        realised = blended.rolling(252, min_periods=60).std() * np.sqrt(252)
        scale = (target_portfolio_vol / realised.replace(0.0, np.nan)).clip(lower=0.25, upper=3.0)
        scale = scale.shift(lag).fillna(1.0)
        weights = weights.mul(scale, axis=0)
        blended = (weights * df).sum(axis=1)

    diagnostics = {
        "streams": list(df.columns),
        "window_start": str(df.index.min().date()),
        "avg_weights": weights.mean().to_dict(),
        "stream_vols_annualised": (df.std() * np.sqrt(252)).to_dict(),
        "stream_correlations": df.corr().to_dict(),
    }
    return AllocationResult(
        blended_returns=blended,
        weights=weights,
        stream_returns=df,
        diagnostics=diagnostics,
    )
