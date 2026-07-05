"""European momentum — a clean geographic out-of-sample test of the price signal.

Same momentum logic that worked in the US, applied to eurozone (EUR) large caps — deep,
free yfinance prices, so unlike the fundamental strategy this is a *proper* OOS: different
continent, same rules. Two questions:

  1. Does momentum EXIST as a factor in Europe? -> dollar-neutral top/bottom-quintile L/S
     of 12-1 and residual momentum (the pure factor).
  2. Does a practical LONG-ONLY momentum book beat the European market? -> top-30 residual
     momentum (monthly, 30/45 band) vs an equal-weight-of-the-universe benchmark.

Caveat: current index constituents => survivorship-biased (milder for momentum than value,
but present). EUR-only so no FX pollution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import LinearCostModel, VectorizedBacktester, summary_stats
from combination import cs_zscore
from data import download_ohlcv, european_eur_tickers
from signals import momentum_12_1, residual_momentum
from strategies.magic_formula.construct import pnl, weights_banded


def main(start: str = "2007-01-01", end: str | None = None, top_n: int = 30) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = european_eur_tickers()
    print(f"[1/3] {len(tickers)} eurozone tickers; downloading prices {start}→{end} …")
    panel = download_ohlcv(tickers, start, end)
    adj = panel["adj_close"].dropna(how="all", axis=1)
    close = panel["close"].reindex_like(adj)
    volume = panel["volume"].reindex_like(adj)
    print(f"      {adj.shape[1]}/{len(tickers)} names have price history")

    rets = adj.pct_change(fill_method=None).where(lambda x: x.abs() < 1.0)
    bench = rets.mean(axis=1)  # equal-weight-of-universe = "European market" proxy (EUR)

    print("[2/3] pure factor: dollar-neutral L/S quintiles …")
    ls = {}
    cost = LinearCostModel(half_spread_bps=3.0, impact_coef_bps=12.0)  # Europe a touch wider
    for name, sig in [("mom_12_1", momentum_12_1(adj)), ("residual_mom", residual_momentum(adj))]:
        bt = VectorizedBacktester(top_quantile=0.2, rebalance="ME", cost_model=cost)
        res = bt.run(cs_zscore(sig), adj, volume=volume)
        ls[name] = res.net_returns

    print("[3/3] practical long-only: top-30 residual momentum (30/45 band) …")
    rank = residual_momentum(adj)
    w = weights_banded(rank, adj, "ME", top_n, 45)
    lo_net, turn = pnl(w, adj, volume, close)

    # Common window for fair comparison.
    series = {"L/S mom_12_1": ls["mom_12_1"], "L/S residual_mom": ls["residual_mom"],
              "long-only resid top30": lo_net, "EW European market": bench}
    common = None
    for s in series.values():
        idx = s.replace(0.0, np.nan).dropna().index
        common = idx if common is None else common.intersection(idx)

    print("\n" + "=" * 78)
    print(f"EUROPEAN MOMENTUM  ({start}→{end}, EUR large caps, net of costs)")
    print("=" * 78)
    print(f"  {'strategy':26s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'corr_mkt':>8s}")
    for name, s in series.items():
        ss = summary_stats(s.reindex(common).fillna(0.0))
        c = s.reindex(common).fillna(0.0).corr(bench.reindex(common).fillna(0.0))
        extra = f"  turnover {turn:.1f}x/yr" if name.startswith("long-only") else ""
        print(f"  {name:26s} {ss['ann_return']:>+8.2%} {ss['ann_vol']:>7.2%} "
              f"{ss['sharpe']:>+7.2f} {ss['max_drawdown']:>+8.2%} {c:>+8.2f}{extra}")

    out = ROOT / "results" / "europe_momentum"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(series).to_csv(out / "europe_momentum_net.csv")
    print(f"\n  wrote {out}/europe_momentum_net.csv")


if __name__ == "__main__":
    main()
