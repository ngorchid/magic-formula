"""Magic formula: how wide should the no-trade band be? Sweeps hold_n at fixed top_n=30.

WHY. The live book holds 30 names and keeps one until its rank falls out of the top 45. That
band was never swept -- 45 was chosen as "1.5x top_n", not measured.

THE HYPOTHESIS, and it has a real mechanism behind it. Magic formula ranks on earnings yield
(EBIT/EV) and ROIC, so a name whose PRICE RISES mechanically drops down the ranking: the numerator
is unchanged and EV has grown. A valuation strategy therefore sells its winners BY CONSTRUCTION,
and a tight band makes it sell them sooner. Widening the band lets a winner keep running.

THE COUNTER-ARGUMENT, which is why this has to be measured rather than argued: a wider band also
retains names that have genuinely deteriorated (falling ROIC, not just a risen price), and it
dilutes the average rank quality of the book. Turnover falls either way, so cost falls -- the
question is whether gross return falls faster.

⚠ SELECTION DISCIPLINE. This is a one-dimensional sweep of a parameter that has never been
touched, on a strategy already chosen on this data. Five values is a small search, but it is a
search: an IS/OOS split is reported alongside the full-sample number, and a band that wins only
in one half should be read as noise. The live 45 is the incumbent and needs a real margin to be
displaced, not a hair.

Run: python3 scripts/magic_band_lab.py
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

from backtest import summary_stats                       # noqa: E402
from strategies.magic_formula import (EnhancedMagicConfig,  # noqa: E402
                                      enhanced_weights)
from strategies.magic_formula.construct import pnl        # noqa: E402
from run_best_magic import _load                          # noqa: E402

BANDS = [30, 45, 60, 75, 100, 150]
SPLIT = "2019-07-01"          # ~halfway through 2012-01 .. 2026-08


def main() -> None:
    # LIVE config: use_graham=False is what scripts/run_paper.py deploys.
    cfg0 = EnhancedMagicConfig(use_graham=False)
    print("[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load(
        "sp500_pit", cfg0, "2012-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))

    rows = []
    for b in BANDS:
        cfg = EnhancedMagicConfig(use_graham=False, hold_n=b)
        w, rank = enhanced_weights(f, mcap, adj, base, cfg)
        net, turnover = pnl(w, adj, volume, close)
        idx = net.replace(0.0, np.nan).dropna().index
        net = net.reindex(idx).fillna(0.0)
        held = (w > 0).sum(axis=1)
        row = {"band": b, "turnover": turnover,
               "avg_held": float(held[held > 0].mean())}
        for lab, sl in (("FULL", slice(None)), ("IS", slice(None, SPLIT)),
                        ("OOS", slice(SPLIT, None))):
            s = summary_stats(net.loc[sl])
            row[f"ret_{lab}"] = s["ann_return"]
            row[f"sh_{lab}"] = s["sharpe"]
            row[f"dd_{lab}"] = s["max_drawdown"]
        rows.append(row)
        print(f"  band {b:>3} done")

    spy_s = summary_stats(spy.reindex(idx).fillna(0.0))
    df = pd.DataFrame(rows)
    print("\n" + "=" * 98)
    print(f"NO-TRADE BAND SWEEP — {label}, top_n=30, monthly, use_graham=False (LIVE config)")
    print("=" * 98)
    print(f"  {'band':>5}{'held':>7}{'turn':>7}{'ann ret':>10}{'Sharpe':>9}{'maxDD':>9}"
          f"   {'Sh IS':>7}{'Sh OOS':>8}")
    print("  " + "-" * 94)
    for _, x in df.iterrows():
        star = "  <- LIVE" if x["band"] == 45 else ""
        print(f"  {int(x['band']):>5}{x['avg_held']:>7.1f}{x['turnover']:>6.1f}x"
              f"{x['ret_FULL']:>+10.2%}{x['sh_FULL']:>+9.2f}{x['dd_FULL']:>+9.2%}"
              f"   {x['sh_IS']:>+7.2f}{x['sh_OOS']:>+8.2f}{star}")
    print(f"  {'SPY':>5}{'':>7}{'':>7}{spy_s['ann_return']:>+10.2%}{spy_s['sharpe']:>+9.2f}"
          f"{spy_s['max_drawdown']:>+9.2%}")

    live = df[df["band"] == 45].iloc[0]
    best = df.loc[df["sh_FULL"].idxmax()]
    print(f"\n  live band 45 : Sharpe {live['sh_FULL']:+.2f} (IS {live['sh_IS']:+.2f}, "
          f"OOS {live['sh_OOS']:+.2f}), turnover {live['turnover']:.1f}x")
    print(f"  best band {int(best['band']):<3}: Sharpe {best['sh_FULL']:+.2f} "
          f"(IS {best['sh_IS']:+.2f}, OOS {best['sh_OOS']:+.2f}), turnover {best['turnover']:.1f}x")
    print(f"  delta        : {best['sh_FULL'] - live['sh_FULL']:+.2f} Sharpe, "
          f"{best['ret_FULL'] - live['ret_FULL']:+.2%} ann return")
    n_yr = len(idx) / 252
    print(f"\n  ⚠ Sharpe standard error on {n_yr:.1f} years is ~{1 / np.sqrt(n_yr):.2f}. "
          f"A gap smaller than that is noise,")
    print("    and the same band must win in BOTH halves to be worth acting on.")
    df.to_csv(ROOT / "results" / "magic_band_sweep.csv", index=False)


if __name__ == "__main__":
    main()
