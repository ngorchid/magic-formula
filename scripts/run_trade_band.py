"""No-trade band (hysteresis) vs. plain monthly top-30 — does it cut churn without hurting?

Motivation: a hard top-30 cutoff re-ranked monthly churns names sitting near the boundary
(sell at rank 31, rebuy at 29) — turnover with no informational content, and at odds with
value's slow horizon. The band buys to fill 30 slots but *holds* a name until it drops out
of a wider `hold_n` band, so still-cheap names aren't sold over rank noise.

Compares plain top-30 against 30/45 and 30/60 bands on the clean PIT S&P 500 and the biased
small-cap bucket, reporting Sharpe, drawdown, and annualised turnover. Strategy = the
FCF+growth+residual-momentum book.
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
from strategies.magic_formula.construct import (
    combine_ranks,
    pnl,
    size_bucket,
    weights_banded,
    weights_top_n,
)

_ITEMS = [
    "revenue", "operating_cash_flow", "capex", "shares_diluted",
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
    rank = combine_ranks(
        [[fcf_ev_yield(f, mcap), fcf_return_on_capital(f)],
         [revenue_growth(f), fcf_growth(f)],
         [residual_momentum(adj)]],
        base,
    )
    return adj, close, volume, spy, base, mcap, rank


def main(start: str = "2012-01-01", end: str | None = None, top_n: int = 30,
         rebalance: str = "ME") -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    rows, spy_ref = [], {}
    for universe, bucket in [("sp500_pit", "all"), ("sp1500", "small")]:
        print(f"[load] {universe} …")
        adj, close, volume, spy, base, mcap, rank = _load(universe, start, end)
        spy_ref[universe] = spy
        elig = base if bucket == "all" else size_bucket(mcap, base, 0.0, 1 / 3)
        r = rank.where(elig)
        schemes = {
            "top-30 (no band)":  weights_top_n(r, adj, rebalance, top_n),
            "band 30/45":        weights_banded(r, adj, rebalance, top_n, 45),
            "band 30/60":        weights_banded(r, adj, rebalance, top_n, 60),
        }
        for name, w in schemes.items():
            net, turn = pnl(w, adj, volume, close)
            rows.append((f"{universe}:{bucket}", name, net, turn))

    print("\n" + "=" * 90)
    print(f"NO-TRADE BAND vs plain top-30  ({start}→{end}, {rebalance}, net)")
    print("=" * 90)
    cur = None
    for label, name, net, turn in rows:
        if label != cur:
            print(f"\n  [{label}]")
            cur = label
        idx = net.replace(0.0, np.nan).dropna().index
        s = summary_stats(net.reindex(idx).fillna(0.0))
        print(f"    {name:18s} ret {s['ann_return']:>+7.2%}  sharpe {s['sharpe']:>+5.2f}  "
              f"maxDD {s['max_drawdown']:>+7.2%}  turnover {turn:>5.1f}x/yr")
    print()
    for uni, spy in spy_ref.items():
        s = summary_stats(spy.dropna())
        print(f"  SPY ({uni}): ret {s['ann_return']:+.2%}  sharpe {s['sharpe']:+.2f}")

    out = ROOT / "results" / "trade_band"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{l}|{n}": net for l, n, net, _ in rows}).to_csv(out / "band_net_returns.csv")
    print(f"\nwrote {out}/band_net_returns.csv")


if __name__ == "__main__":
    main()
