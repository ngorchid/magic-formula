"""Lab: does adding JPY (6J) to the executable trend basket earn its place?

The overlay's RESEARCH basket has 16 markets; only 7 are executable as futures
(strategies/trend_futures/contracts.py: ES, ZN, GC, HG, CL, 6E, 6A). A correlation scan
of the 9 missing markets against the 7 held showed almost all are redundant — UUP 0.95
vs FXE, TLT 0.92 vs IEF, EFA 0.85 vs SPY, SLV 0.79 vs GLD, HYG 0.75 vs SPY. The markets
that HAVE liquid retail futures are exactly the redundant ones, and the ones that would
diversify (credit, EM, DM-ex-US) have no retail future. That validated the decision to
stay at 7.

Two exceptions came out of that scan:
    FXY (yen)         max |corr| 0.51 vs IEF   -- one contract, tradeable
    DBA (agriculture) max |corr| 0.33 vs FXA   -- needs a ZC/ZS/ZW sleeve, 3 contracts

Low correlation is necessary but not sufficient: a market has to survive the overlay's
own sizing, which allocates target_vol/sqrt(N) per market, so adding one SHRINKS every
existing position. This tests whether the diversification beats that dilution.

SLV is included as a control. It is redundant (0.79 vs GLD), so if adding SLV also
"improves" things, the test is measuring noise rather than diversification.

Sub-periods are reported because of the lesson from the vol-spike hedge: a result that
only holds in one regime is not a result.

Run: python scripts/trend_add_jpy_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from strategies.trend_futures.overlay import TREND_BASKET, TrendOverlayConfig, run_trend_overlay

# The 7 markets that actually have a contract in contracts.py.
EXEC7 = {"SPY": "equity_us", "IEF": "rates_us_10y", "GLD": "gold",
         "CPER": "copper", "USO": "oil", "FXE": "fx_eur", "FXA": "fx_aud"}

VARIANTS = {
    "7 executable (current)": EXEC7,
    "+ FXY (yen)": {**EXEC7, "FXY": "fx_jpy"},
    "+ DBA (agriculture)": {**EXEC7, "DBA": "agriculture"},
    "+ FXY + DBA": {**EXEC7, "FXY": "fx_jpy", "DBA": "agriculture"},
    "+ SLV [CONTROL, redundant]": {**EXEC7, "SLV": "silver"},
    "all 16 (research basket)": dict(TREND_BASKET),
}

PERIODS = [("full", "2011-01-01", None),
           ("2011-2018", "2011-01-01", "2018-12-31"),
           ("2019-2026", "2019-01-01", "2026-12-31")]


def run(basket: dict[str, str]) -> pd.Series:
    cfg = TrendOverlayConfig(basket=basket)
    return run_trend_overlay(cfg).net_returns


def main() -> None:
    print("[run] trend overlay per basket …")
    nets = {name: run(b) for name, b in VARIANTS.items()}

    for label, a, b in PERIODS:
        print("\n" + "=" * 104)
        print(f"TREND OVERLAY ALONE — {label}")
        print("=" * 104)
        print(f"  {'basket':30s} {'N':>3s} {'CAGR':>8s} {'vol':>7s} {'Sharpe':>7s} "
              f"{'maxDD':>8s} {'ΔSharpe':>8s}")
        base = None
        for name, net in nets.items():
            r = net.loc[a:b] if b else net.loc[a:]
            s = summary_stats(r.fillna(0.0))
            if base is None:
                base = s["sharpe"]
            n = len(VARIANTS[name])
            print(f"  {name:30s} {n:>3d} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
                  f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.1%} "
                  f"{s['sharpe']-base:>+8.2f}")

    # Book level: does it survive being bolted onto the equity core?
    print("\n" + "=" * 104)
    print("BOOK LEVEL — magic-formula core + trend overlay scaled to 7% vol")
    print("=" * 104)
    mf = pd.read_csv(ROOT / "results" / "beta_hedge" / "beta_hedge.csv",
                     index_col=0, parse_dates=True)["long_book"]
    print(f"  {'basket':30s} {'CAGR':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s} {'ΔSharpe':>8s}")
    base = None
    for name, net in nets.items():
        idx = mf.index.intersection(net.index)
        t = net.reindex(idx).fillna(0.0)
        t = t * (0.07 / (t.std() * np.sqrt(252)))
        book = mf.reindex(idx).fillna(0.0) + t
        s = summary_stats(book)
        if base is None:
            base = s["sharpe"]
        print(f"  {name:30s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.1%} {s['sharpe']-base:>+8.2f}")

    print("\n" + "=" * 104)
    print("PAIRED DIFFERENCE vs the current 7 (trend overlay alone, daily)")
    print("=" * 104)
    b7 = nets["7 executable (current)"]
    for name, net in nets.items():
        if name == "7 executable (current)":
            continue
        idx = b7.index.intersection(net.index)
        d = (net.reindex(idx).fillna(0.0) - b7.reindex(idx).fillna(0.0)).dropna()
        t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        print(f"  {name:30s} ann {d.mean()*252:>+7.2%}  vol {d.std()*np.sqrt(252):>6.2%}  "
              f"t={t:>+6.2f}   {'significant' if abs(t) > 2 else 'not distinguishable from zero'}")

    out = ROOT / "results" / "trend_overlay"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(nets).to_csv(out / "add_jpy_variants.csv")
    print(f"\n  wrote {out}/add_jpy_variants.csv")


if __name__ == "__main__":
    main()
