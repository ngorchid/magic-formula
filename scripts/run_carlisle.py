"""Carlisle's test: does the quality (return-on-capital) leg help, or does pure
cheapness win?

Tobias Carlisle (Quantitative Value / The Acquirer's Multiple) found that Greenblatt's
quality rank (ROC) *reduced* returns versus buying the cheapest stocks alone. We test the
FCF analogue: cheapness = FCF/EV; quality = FCF/capital.

Two contexts:
  * PURE value — cheap-only vs cheap+quality as an equal two-rank sum (faithful to the
    Magic-Formula vs Acquirer's-Multiple comparison).
  * FULL strategy — the same value leg embedded in our value+growth+momentum book, to see
    whether the finding survives once growth and momentum are present.

Run on the clean point-in-time S&P 500 (trustworthy) and the biased S&P 1500 small-cap
bucket (where value looks strongest). Long-only, equal-weight top-30, net of costs.
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
    return adj, close, volume, spy, base, mcap, f


def main(start: str = "2012-01-01", end: str | None = None, top_n: int = 30,
         rebalance: str = "ME") -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    rows, spy_ref = [], {}

    for universe, bucket in [("sp500_pit", "all"), ("sp1500", "small")]:
        print(f"[load] {universe} …")
        adj, close, volume, spy, base, mcap, f = _load(universe, start, end)
        spy_ref[universe] = spy
        elig = base if bucket == "all" else size_bucket(mcap, base, 0.0, 1 / 3)

        cheap = fcf_ev_yield(f, mcap)              # FCF / EV  (Acquirer's-Multiple analog)
        quality = fcf_return_on_capital(f)         # FCF / capital  (Greenblatt quality leg)
        growth = [revenue_growth(f), fcf_growth(f)]
        mom = [residual_momentum(adj)]

        # PURE value: cheap-only vs cheap+quality (equal two-rank sum, Greenblatt-style).
        variants = {
            "pure: cheap-only":        [[cheap]],
            "pure: cheap+quality":     [[cheap], [quality]],
            # FULL strategy: value leg (1 family) + growth + momentum.
            "full: cheap-only":        [[cheap], growth, mom],
            "full: cheap+quality":     [[cheap, quality], growth, mom],
        }
        for name, families in variants.items():
            rank = combine_ranks(families, elig)
            net = long_only_backtest(rank, adj, volume, close, rebalance, top_n)
            rows.append((f"{universe}:{bucket}", name, net))

    print("\n" + "=" * 90)
    print(f"CARLISLE  cheapness-only vs +quality  ({start}→{end}, top-{top_n}, {rebalance}, net)")
    print("=" * 90)
    cur = None
    for label, name, net in rows:
        if label != cur:
            print(f"\n  [{label}]")
            cur = label
        idx = net.replace(0.0, np.nan).dropna().index
        s = summary_stats(net.reindex(idx).fillna(0.0))
        print(f"    {name:22s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"sharpe {s['sharpe']:>+5.2f}  maxDD {s['max_drawdown']:>+7.2%}")
    print()
    for uni, spy in spy_ref.items():
        s = summary_stats(spy.dropna())
        print(f"  SPY ({uni}): ret {s['ann_return']:+.2%}  sharpe {s['sharpe']:+.2f}  "
              f"maxDD {s['max_drawdown']:+.2%}")

    out = ROOT / "results" / "carlisle"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{l}|{n}": s for l, n, s in rows}).to_csv(out / "carlisle_net_returns.csv")
    print(f"\nwrote {out}/carlisle_net_returns.csv")


if __name__ == "__main__":
    main()
