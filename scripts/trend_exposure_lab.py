"""Fixing the vanishing-volatility exposure blow-up in the trend overlay.

THE BUG. Sizing is inverse-vol: expo_i = budget * target_vol / sqrt(N) / vol_i. As vol_i -> 0
the notional explodes. The portfolio vol-target does NOT catch it, because it is circular:

    per_mkt_vol_i = (expo_i / budget) * vol_i = target_vol / sqrt(N)     <- vol_i CANCELS
    est_vol       = sqrt(sum per_mkt_vol^2)   = target_vol               <- always on target
    scale         = target_vol / est_vol      = 1.0                      <- a NO-OP

The model sizes with the same collapsed estimate it uses to measure risk, so notional runs away
while estimated risk looks perfect. And the LIVE code (`trend_overlay/execution.py`) has NO
leverage cap at all — the backtest's `max_leverage=5.0` was never carried into
`TrendPaperConfig`.

Measured on the live 10-market basket: gross runs ~2.8x budget at the median and has hit 4.8x;
IEF at its 2.8% vol floor implies a $225k position on a $200k book, in ONE market.

FIXES TESTED (each addresses a different part of it):
  vol floor       per-market, at a percentile of that market's OWN vol history. Adaptive across
                  markets whose normal vol differs 5x (IEF 5.6% vs USO 31.2%), unlike an
                  absolute floor. Also leans against the estimate itself: a 60d vol at its 1st
                  percentile is far more likely to mean-revert UP than to persist.
  blended vol     max(vol_60d, k * vol_252d) — a slower estimate cannot collapse as fast.
  per-market cap  hard ceiling on any single market's notional, as a fraction of budget.
                  Targets CONCENTRATION, which the gross cap alone does not fix.
  gross cap       total notional <= L * budget. The blunt backstop.

Judged on: does it tame exposure WITHOUT damaging the vol-target's left-tail protection
(removing vol-targeting entirely took skew to -0.86, so that protection is load-bearing).

Run: python scripts/trend_exposure_lab.py
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from backtest import summary_stats  # noqa: E402

# the LIVE 10-market basket, via ETF proxies (ES/ZN/GC/HG/CL/6E/6A + ZB/SI/6J)
LIVE10 = ["SPY", "IEF", "GLD", "CPER", "USO", "FXE", "FXA", "TLT", "SLV", "FXY"]
OUT = ROOT / "results" / "trend_overlay"


@dataclass
class Fix:
    name: str
    vol_floor_pct: float | None = None    # percentile of own trailing vol, e.g. 0.20
    vol_blend_k: float | None = None      # vol_used = max(vol60, k * vol252)
    per_market_cap: float | None = None   # max |notional| per market, as a fraction of budget
    gross_cap: float | None = None        # max gross notional, as a multiple of budget


def build(prices: pd.DataFrame, fix: Fix, target_vol=0.10, vol_window=60,
          lookbacks=(126, 252), rebalance="D"):
    rets = prices.pct_change(fill_method=None)
    N = prices.shape[1]
    sig = sum(np.sign(prices / prices.shift(lb) - 1.0) for lb in lookbacks) / len(lookbacks)

    vol = (rets.rolling(vol_window).std() * np.sqrt(252)).replace(0.0, np.nan)
    vol_used = vol.copy()
    if fix.vol_blend_k:
        slow = rets.rolling(252).std() * np.sqrt(252)
        vol_used = np.maximum(vol_used, fix.vol_blend_k * slow)
    if fix.vol_floor_pct:
        # trailing percentile of each market's OWN vol — expanding, so no lookahead
        floor = vol.expanding(252).quantile(fix.vol_floor_pct)
        vol_used = np.maximum(vol_used, floor)

    # weights as a FRACTION OF BUDGET (so 1.0 = one budget of notional)
    w = sig * (target_vol / np.sqrt(N)) / vol_used
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if fix.per_market_cap:
        w = w.clip(lower=-fix.per_market_cap, upper=fix.per_market_cap)

    rebal = prices.resample(rebalance).last().index.intersection(prices.index)
    tgt = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    tgt.loc[rebal] = w.loc[rebal]
    w = tgt.ffill().fillna(0.0).shift(1).fillna(0.0)

    if fix.gross_cap:
        gross = w.abs().sum(axis=1)
        w = w.mul((fix.gross_cap / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0), axis=0)

    gross = w.abs().sum(axis=1)
    ret = (w * rets.fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().fillna(w.abs()).sum(axis=1)
    net = ret - turn * (2.0 / 1e4)          # same 2bp cost model as the live backtest
    return net, gross, w


def main() -> None:
    from data import download_ohlcv
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"loading live 10-market basket {LIVE10} ...")
    px = download_ohlcv(LIVE10, "2011-01-01", end)["adj_close"].dropna(how="all", axis=1).sort_index()

    fixes = [
        Fix("A baseline (live, no cap)"),
        Fix("B vol floor 20th pct", vol_floor_pct=0.20),
        Fix("C vol floor 10th pct", vol_floor_pct=0.10),
        Fix("D blended max(v60,.75*v252)", vol_blend_k=0.75),
        Fix("E per-market cap 40%", per_market_cap=0.40),
        Fix("F gross cap 3x", gross_cap=3.0),
        Fix("G floor20 + mkt40 + gross3", vol_floor_pct=0.20, per_market_cap=0.40, gross_cap=3.0),
        Fix("H floor20 + gross2.5", vol_floor_pct=0.20, gross_cap=2.5),
    ]
    print("\n" + "=" * 112)
    print("EXPOSURE FIXES — live 10-market basket, variant D config (126/252, daily), net of 2bp")
    print("=" * 112)
    print(f"  {'variant':30s} {'ret':>7s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s} {'skew':>6s} | "
          f"{'gross med':>10s} {'gross p95':>10s} {'gross max':>10s} {'max 1-mkt':>10s}")
    print("  " + "-" * 108)
    rows = []
    for f in fixes:
        net, gross, w = build(px, f)
        s = summary_stats(net.dropna())
        mx = w.abs().max().max()
        print(f"  {f.name:30s} {s['ann_return']:>+7.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
              f"{s['max_drawdown']:>+8.2%} {net.skew():>+6.2f} | {gross.median():>9.2f}x "
              f"{gross.quantile(.95):>9.2f}x {gross.max():>9.2f}x {mx:>9.2f}x")
        rows.append({"variant": f.name, "sharpe": s["sharpe"], "maxdd": s["max_drawdown"],
                     "skew": net.skew(), "gross_med": gross.median(), "gross_max": gross.max(),
                     "max_one_market": mx})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "exposure_fixes.csv", index=False)
    print(f"\n  wrote {OUT}/exposure_fixes.csv")
    print("\n  'gross' is total notional as a MULTIPLE OF BUDGET; 'max 1-mkt' is the largest")
    print("  single-market notional ever taken, also as a multiple of budget.")


if __name__ == "__main__":
    main()
