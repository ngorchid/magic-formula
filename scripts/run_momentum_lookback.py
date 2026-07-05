"""6-month vs 12-month residual momentum in the value+growth+momentum Magic Formula.

Only the momentum lookback changes (skip stays 1 month); value and growth families are
identical. Compared across the setups we care about: the biased S&P 1500 (all + small-cap
bucket) and the clean point-in-time S&P 500.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from data import (
    download_ohlcv,
    load_fundamentals,
    sp500_pit_eligible,
    sp500_pit_universe,
    sp500_sectors,
    sp1500_sectors,
    sp1500_tickers,
)
from signals import fcf_ev_yield, fcf_growth, fcf_return_on_capital, residual_momentum, revenue_growth
from strategies.magic_formula.construct import combine_ranks, long_only_backtest, size_bucket

_ITEMS = [
    "revenue", "operating_income", "operating_cash_flow", "capex", "shares_diluted",
    "short_term_debt", "long_term_debt", "cash",
    "total_current_assets", "total_current_liabilities", "ppe_net",
]
EXCLUDE_SECTORS = ("Financial Services", "Utilities")
LOOKBACKS = {"6mo": 126, "12mo": 252}


def _load(universe: str, start: str, end: str):
    if universe == "sp500_pit":
        tickers, sector_src = sp500_pit_universe(start, end), sp500_sectors
    else:
        tickers, sector_src = sp1500_tickers(), sp1500_sectors
    full = sorted(set(tickers + ["SPY"]))
    panel = download_ohlcv(full, start, end)
    adj_all = panel["adj_close"].dropna(how="all", axis=1)
    close_all = panel["close"].reindex_like(adj_all)
    vol_all = panel["volume"].reindex_like(adj_all)
    spy = adj_all["SPY"].pct_change(fill_method=None)
    adj = adj_all.drop(columns=["SPY"], errors="ignore")
    close = close_all.drop(columns=["SPY"], errors="ignore")
    volume = vol_all.drop(columns=["SPY"], errors="ignore")

    excluded = sector_src().reindex(adj.columns).isin(EXCLUDE_SECTORS)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(
        excluded, index=adj.columns
    )
    if universe == "sp500_pit":
        base = base & sp500_pit_eligible(adj.index, list(adj.columns))

    f = load_fundamentals(list(adj.columns), start, end, items=_ITEMS,
                          sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()
    value = [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)]
    growth = [revenue_growth(f), fcf_growth(f)]
    return adj, close, volume, spy, base, mcap, value, growth


def main(start: str = "2012-01-01", end: str | None = None, top_n: int = 30,
         rebalance: str = "ME") -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    rows, spy_ref = [], {}

    for universe, buckets in [("sp1500", ["all", "small"]), ("sp500_pit", ["all"])]:
        print(f"[load] {universe} …")
        adj, close, volume, spy, base, mcap, value, growth = _load(universe, start, end)
        spy_ref[universe] = spy
        bucket_elig = {"all": base, "small": size_bucket(mcap, base, 0.0, 1 / 3)}
        # Precompute momentum once per lookback (independent of bucket).
        mom = {k: [residual_momentum(adj, lookback=lb, skip=21)] for k, lb in LOOKBACKS.items()}
        for b in buckets:
            elig = bucket_elig[b]
            for k in LOOKBACKS:
                rank = combine_ranks([value, growth, mom[k]], elig)
                net = long_only_backtest(rank, adj, volume, close, rebalance, top_n)
                rows.append((f"{universe}:{b}", k, net))

    print("\n" + "=" * 84)
    print(f"RESIDUAL MOMENTUM LOOKBACK  6mo vs 12mo  ({start}→{end}, top-{top_n}, {rebalance}, net)")
    print("=" * 84)
    print(f"  {'universe:bucket':18s} {'mom':6s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s}")
    for label, k, net in rows:
        idx = net.replace(0.0, np.nan).dropna().index
        s = summary_stats(net.reindex(idx).fillna(0.0))
        print(f"  {label:18s} {k:6s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")
    for uni, spy in spy_ref.items():
        s = summary_stats(spy.dropna())
        print(f"  {'SPY ('+uni+')':18s} {'—':6s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")

    out = ROOT / "results" / "momentum_lookback"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{l}:{k}": n for l, k, n in rows}).to_csv(out / "lookback_net_returns.csv")
    print(f"\nwrote {out}/lookback_net_returns.csv")


if __name__ == "__main__":
    main()
