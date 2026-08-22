"""Does the magic-formula vol target do anything, and should it be tightened?

WHY. The book's drawdown is -32.8%, essentially SPY's (-33.7%). The put-hedge exercise priced
crash protection at ~3.1%/yr of premium to get from -33% to ~-14%, with Sharpe falling to ~0.93.
Vol targeting is the free alternative -- it de-levers rather than buying insurance -- so the
question is what the existing target actually delivers.

⚠ A CORRECTION IS BAKED INTO THIS FILE. It was suggested that "removing the <=1 clip would let
the book shrink in high-vol regimes". That is backwards. `_gross_scalar` returns
np.clip(vol_target/est_book_vol, 0.0, 1.0) -- the clip is an UPPER bound, so de-levering below 1
is ALREADY permitted. Removing it allows LEVERAGE above 1, which is the opposite of drawdown
control, and the table below shows exactly that: gross rises to 1.41x and maxDD gets WORSE
(-39.1%). The clip is doing useful work and should stay.

THE REAL FINDING is that the live target barely engages. At vol_target=25% the raw scalar has a
median of 1.55, so it is clipped to 1.0 and binds on only 7.6% of days -- dormant outside genuine
turmoil. Tightening it traces the risk axis at CONSTANT Sharpe, which is the cheapest drawdown
control available:

    no vol target      +19.89%   Sharpe +0.98   maxDD -32.8%   gross 1.00
    25% (LIVE)         +18.46%          +0.95         -30.3%         0.91
    15%                +15.41%          +0.98         -24.3%         0.80
    12%                +12.82%          +0.97         -20.7%         0.68

⚠ THIS IS A SCALE DECISION WEARING A PARAMETER'S CLOTHES. Returns fall roughly in proportion --
you are running a SMALLER BOOK, not getting protection for nothing. Holding 80% of the strategy
and 20% cash achieves much the same thing; the vol target just concentrates the de-levering into
high-vol periods, which is why Sharpe holds up. There is no optimum here, only a choice of how
much drawdown you want.

⚠ LIVE/BACKTEST DISCREPANCY, worth fixing separately. The BACKTEST applies NO vol target at all
(weights_banded has no scalar), while the LIVE orchestrator applies the 25% clipped version. So
the deployed strategy runs a slightly different configuration (-0.03 Sharpe, avg gross 0.91) than
the +0.98 that has been quoted throughout. Small, but the two should agree.

Run: python3 scripts/magic_voltarget_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from backtest import summary_stats                          # noqa: E402
from strategies.magic_formula import (EnhancedMagicConfig,   # noqa: E402
                                      enhanced_weights)
from strategies.magic_formula.construct import pnl           # noqa: E402
from run_best_magic import _load                              # noqa: E402


def main() -> None:
    cfg = EnhancedMagicConfig(use_graham=False)
    print("[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load(
        "sp500_pit", cfg, "2012-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    w, _ = enhanced_weights(f, mcap, adj, base, cfg)

    # est_book_vol as the live orchestrator computes it: median 63d vol of the HELD names,
    # times a ~0.6 diversification haircut.
    vol63 = adj.pct_change(fill_method=None).rolling(cfg.vol_window).std() * np.sqrt(252)
    est = vol63.where(w > 0).median(axis=1) * 0.6
    print(f"\n  est_book_vol: median {est.median():.1%}, p95 {est.quantile(.95):.1%}, "
          f"max {est.max():.1%}")
    print("  How often would each target actually BIND (raw scalar < 1)?")
    for tgt in (0.25, 0.20, 0.15, 0.12):
        s = tgt / est
        print(f"    vol_target {tgt:.0%}: median raw scalar {s.median():.2f} -> "
              f"binds on {100 * (s < 1).mean():.1f}% of days")

    net0, _ = pnl(w, adj, volume, close)
    idx = net0.replace(0.0, np.nan).dropna().index
    b = summary_stats(net0.reindex(idx).fillna(0.0))
    print("\n" + "=" * 80)
    print(f"VOL TARGET — {label}, top 30 / band 45")
    print("=" * 80)
    print(f"  {'variant':36}{'ann ret':>10}{'Sharpe':>9}{'maxDD':>9}{'avg gross':>11}")
    print("  " + "-" * 76)
    print(f"  {'no vol target (= the backtest)':36}{b['ann_return']:>+10.2%}"
          f"{b['sharpe']:>+9.2f}{b['max_drawdown']:>+9.2%}{1.0:>11.2f}")
    rows = [{"variant": "none", "ann": b["ann_return"], "sharpe": b["sharpe"],
             "dd": b["max_drawdown"], "gross": 1.0}]
    for tgt, clip in ((0.25, True), (0.25, False), (0.20, True), (0.15, True), (0.12, True)):
        sc = (tgt / est).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
        sc = sc.clip(0.0, 1.0) if clip else sc.clip(0.0, 2.0)
        ws = w.mul(sc.shift(1).fillna(1.0), axis=0)      # shift: yesterday's vol sizes today
        n, _ = pnl(ws, adj, volume, close)
        s = summary_stats(n.reindex(idx).fillna(0.0))
        lab = (f"vol_target {tgt:.0%}"
               + (" (clip<=1, LIVE)" if clip else " UNCLIPPED (levers up)"))
        print(f"  {lab:36}{s['ann_return']:>+10.2%}{s['sharpe']:>+9.2f}"
              f"{s['max_drawdown']:>+9.2%}{ws.sum(axis=1).mean():>11.2f}")
        rows.append({"variant": lab, "ann": s["ann_return"], "sharpe": s["sharpe"],
                     "dd": s["max_drawdown"], "gross": float(ws.sum(axis=1).mean())})
    print("\n  Compare with the alternative: a 1-year 10% OTM put costs ~3.1%/yr of notional,")
    print("  takes maxDD to ~-14% and Sharpe to ~0.93. Vol targeting buys most of that")
    print("  protection for NO premium, because it de-levers instead of insuring.")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "magic_voltarget.csv", index=False)


if __name__ == "__main__":
    main()
