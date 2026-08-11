"""Core + overlay book — the CORRECTED assessment with the ENHANCED magic formula.

The enhanced MF (results/best_magic/best_sp500_pit_all.csv) is a long-only β≈0.9 equity book,
Sharpe ~1.0, 19.7% vol, −36% maxDD. It's the CORE, not a neutral diversifier — so the useful
question isn't a risk-parity blend (which would tiny-weight a high-vol core) but: bolt the neutral
overlays (trend, VRP) on top via the margin account (capital-efficient — futures/options use margin,
not cash), and measure whether they lift Sharpe and CUT the equity drawdown.

Key expected asymmetry: TREND has crisis-alpha (positive in crashes) → should cut the core's
drawdown. VRP is short-vol (negative in crashes) → adds carry in calm times but does NOT help the
crash drawdown. The crisis windows show this. VRP level is an idealized proxy — caveat loudly.

Run: python scripts/book_assess2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats


def vrp_series() -> pd.Series:
    import yfinance as yf
    tk = yf.download(["^VIX", "^VIX3M", "^GSPC"], start="2011-01-01", auto_adjust=True,
                     progress=False)["Close"].dropna()
    vix, vix3m, spx = tk["^VIX"] / 100, tk["^VIX3M"] / 100, tk["^GSPC"]
    ret = spx.pct_change()
    rv20 = ret.rolling(20).std() * np.sqrt(252)
    vrp, ratio = vix - rv20, vix / vix3m
    raw = ((vix.shift(1) ** 2) - 252 * (ret ** 2)).dropna()
    sv = raw * (0.10 / (raw.std() * np.sqrt(252)))
    gate = (ratio.shift(1) < 1.00).reindex(sv.index).fillna(False) & (vrp.shift(1) > 0).reindex(sv.index).fillna(False)
    return sv * gate


def scale_to_vol(s: pd.Series, target: float) -> pd.Series:
    v = s.std() * np.sqrt(252)
    return s * (target / v) if v > 0 else s


def row(name, r, idx):
    s = summary_stats(r.reindex(idx).fillna(0.0))
    return (f"  {name:22s} {s['ann_return']:>+8.1%} {s['ann_vol']:>7.1%} {s['sharpe']:>+7.2f} "
            f"{s['max_drawdown']:>+8.1%}")


def main():
    mf = pd.read_csv(ROOT / "results" / "best_magic" / "best_sp500_pit_all.csv",
                     index_col=0, parse_dates=True)["net_return"]
    trend = pd.read_csv(ROOT / "results" / "trend_overlay" / "trend_overlay_net.csv",
                        index_col=0, parse_dates=True)["trend"]
    vrp = vrp_series()
    for s in (mf, trend, vrp):
        s.index = pd.to_datetime(s.index)

    S = pd.DataFrame({"magic_f (core)": mf, "trend": trend, "vrp": vrp}).dropna()
    idx = S.index

    print("\n" + "=" * 72)
    print(f"CORRECTED BOOK — enhanced magic-formula core + overlays  ({idx[0].date()}->{idx[-1].date()})")
    print("=" * 72)
    print(f"  {'stream':22s} {'annRet':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s}")
    for c in S:
        print(row(c, S[c], idx))
    print("\n  correlation matrix:")
    corr = S.corr()
    print("    " + "".ljust(16) + "".join(f"{c.split()[0]:>10s}" for c in corr.columns))
    for r in corr.index:
        print("    " + f"{r:16s}" + "".join(f"{corr.loc[r, c]:>+10.2f}" for c in corr.columns))

    # Core + overlay: overlays added ON TOP (margin-efficient), sized to modest vol contributions.
    trend_o = scale_to_vol(S["trend"], 0.07)
    vrp_o = scale_to_vol(S["vrp"], 0.05)
    core = S["magic_f (core)"]
    books = {"MF core alone": core,
             "MF + trend": core + trend_o,
             "MF + trend + vrp": core + trend_o + vrp_o}
    print("\n" + "-" * 72)
    print("  CORE + OVERLAY (overlays bolted on via margin; trend→7% vol, vrp→5% vol):")
    print(f"  {'book':22s} {'annRet':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s}")
    for name, b in books.items():
        print(row(name, b, idx))

    print("\n  crisis windows (cumulative return — who saves the core?):")
    wins = [("2015-16 selloff", "2015-08-01", "2016-02-29"),
            ("2018 Q4", "2018-10-01", "2018-12-31"),
            ("2020 COVID", "2020-02-15", "2020-03-31"),
            ("2022 bear", "2022-01-01", "2022-12-31")]
    cols = {"core": core, "trend(7%)": trend_o, "vrp(5%)": vrp_o, "MF+trend": core + trend_o,
            "MF+tr+vrp": core + trend_o + vrp_o}
    print("    " + "window".ljust(18) + "".join(f"{k:>11s}" for k in cols))
    for lbl, a, b in wins:
        print("    " + lbl.ljust(18) + "".join(f"{((1+cols[k].loc[a:b]).prod()-1):>+11.1%}" for k in cols))

    print("\n  NB trend = live-cfg (real). vrp = idealized proxy — level optimistic; note it's the one")
    print("  overlay that does NOT help the crash drawdown (short-vol), only calm-period carry.")


if __name__ == "__main__":
    main()
