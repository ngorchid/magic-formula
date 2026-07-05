"""Out-of-sample / overfitting validation for the multi-stream blend.

Our pipeline is already causal (rolling, lagged), so the blend equity curve is a valid
no-look-ahead path. What it does NOT account for is the **search**: we tried many
configurations on the same 2015-26 window, so the headline Sharpe is selection-biased.
This script quantifies that:

  1. Sweep the configs we actually considered (signal subsets × IC/ICIR) → the empirical
     distribution of Sharpes. Its spread = how much room the search had to flatter us.
  2. Probabilistic Sharpe Ratio — P(true Sharpe > 0), adjusting for sample/skew/kurtosis.
  3. Deflated Sharpe Ratio — PSR vs the Sharpe expected to win by chance across N trials.
     DSR > 0.95 ≈ the edge survives the multiple-testing haircut.
  4. Walk-forward: per-year and split-half Sharpe, to see if the edge is stable or
     concentrated in recent data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import deflated_sharpe_ratio, probabilistic_sharpe_ratio, summary_stats
from combination import equal_risk_allocate
from strategies.equity_mn import EquityMarketNeutral
from strategies.equity_mn.stream import EquityMNConfig
from strategies.trend import CrossAssetTrend, TrendConfig

END = "2026-05-26"
ANN = np.sqrt(252)


def main() -> None:
    print("Running trend stream once…")
    trend_ret = CrossAssetTrend(TrendConfig(start="2015-01-01", end=END)).run().net_returns

    def blend_for(signals, weighting):
        eq = EquityMarketNeutral(EquityMNConfig(
            start="2015-01-01", end=END, signals=signals, combine="ic", ic_weighting=weighting,
        )).run().net_returns
        return equal_risk_allocate({"equity_mn": eq, "trend": trend_ret},
                                   target_portfolio_vol=0.08).blended_returns

    # ---- config sweep (the trials we actually fished through) ----
    subsets = [
        ["momentum_12_1", "short_term_reversal"],
        ["momentum_12_1", "short_term_reversal", "residual_momentum"],
        ["momentum_12_1", "short_term_reversal", "residual_momentum", "low_volatility"],
        ["residual_momentum", "short_term_reversal"],
        ["momentum_12_1", "residual_momentum"],
        ["short_term_reversal", "residual_momentum"],
        ["momentum_12_1", "short_term_reversal", "low_volatility"],
    ]
    print("Sweeping configs to estimate the trial distribution…")
    trial_daily_sr = []
    for sub in subsets:
        for w in ("ic", "icir"):
            b = blend_for(sub, w)
            trial_daily_sr.append(b.mean() / b.std())
    trial_daily_sr = np.array(trial_daily_sr)
    n_trials = len(trial_daily_sr)
    sr_var = float(np.var(trial_daily_sr, ddof=1))
    print(f"  {n_trials} trials | blend Sharpe (ann) range "
          f"{trial_daily_sr.min()*ANN:.2f} … {trial_daily_sr.max()*ANN:.2f}, "
          f"mean {trial_daily_sr.mean()*ANN:.2f}")

    # ---- headline config (our best honest choice: 3 signals, ICIR) ----
    head = blend_for(["momentum_12_1", "short_term_reversal", "residual_momentum"], "icir")
    ann_sr = summary_stats(head)["sharpe"]
    psr = probabilistic_sharpe_ratio(head, 0.0)
    dsr_n, srstar_n = deflated_sharpe_ratio(head, n_trials, sr_var)
    dsr_50, srstar_50 = deflated_sharpe_ratio(head, 50, sr_var)

    print("\n================ Overfitting-adjusted verdict (headline blend) ================")
    print(f"  observed Sharpe (annualised) : {ann_sr:.3f}")
    print(f"  PSR  P(true Sharpe > 0)       : {psr:.3f}")
    print(f"  DSR  (N={n_trials} trials)            : {dsr_n:.3f}   "
          f"[must beat ann Sharpe {srstar_n*ANN:.2f} by chance]")
    print(f"  DSR  (N=50, conservative)     : {dsr_50:.3f}   "
          f"[must beat ann Sharpe {srstar_50*ANN:.2f} by chance]")

    # ---- walk-forward stability ----
    print("\n--- per-year blend Sharpe (causal, no look-ahead) ---")
    for y, g in head.groupby(head.index.year):
        if len(g) < 60:
            continue
        s = g.mean() / g.std() * ANN if g.std() else float("nan")
        print(f"    {y}: {s:+.2f}")
    mid = head.index[len(head) // 2]
    h1, h2 = head[head.index < mid], head[head.index >= mid]
    print(f"  1st half {h1.index.min().date()}..{h1.index.max().date()}: "
          f"Sharpe {h1.mean()/h1.std()*ANN:+.2f}")
    print(f"  2nd half {h2.index.min().date()}..{h2.index.max().date()}: "
          f"Sharpe {h2.mean()/h2.std()*ANN:+.2f}")

    print("\nReading: PSR≫0.95 means the Sharpe is reliably positive; DSR≫0.95 means it")
    print("survives the multiple-testing haircut. If DSR is low, the edge is plausibly")
    print("a product of the search, not a real signal.")


if __name__ == "__main__":
    main()
