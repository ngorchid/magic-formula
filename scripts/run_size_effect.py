"""Does market cap matter? — the size effect for the FCF+growth+momentum Magic Formula.

Greenblatt's claim is that the Magic Formula works best in *smaller* stocks. The S&P 500
can't test this (its smallest members are ~$5-10B), so here we use the broader **current**
S&P 1500 (large + mid + small; survivorship bias explicitly accepted) and run the *same*
value+growth+residual-momentum strategy inside three size tiers — bottom/middle/top third
by market cap each day — plus the whole universe. If the small-cap tier outperforms, the
size tilt is real; if not, the earlier S&P 500 "small-cap tilt hurt" result generalises.

Caveats (stated so the result is read correctly): current constituents only (survivorship-
biased); S&P 600 small caps are ~$1-3B, not micro-caps; the index has a profitability screen
for inclusion, so it's a cleaner small-cap set than the raw Russell 2000.
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


def main(start: str = "2012-01-01", end: str | None = None, top_n: int = 30,
         rebalance: str = "ME", universe: str = "sp1500") -> None:
    """`universe`: "sp1500" (current, survivorship-biased, wide size range) or
    "sp500_pit" (point-in-time S&P 500 — survivorship-corrected but compressed to large
    caps, so it reads the size gradient off *clean* data)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    if universe == "sp500_pit":
        tickers = sp500_pit_universe(start, end)
        sector_src, label = sp500_sectors, "S&P 500 (point-in-time)"
    else:
        tickers = sp1500_tickers()
        sector_src, label = sp1500_sectors, "S&P 1500 (current)"
    full = sorted(set(tickers + ["SPY"]))
    print(f"[1/4] prices {start}→{end} for {len(full)} {label} names …")
    panel = download_ohlcv(full, start, end)
    adj_all = panel["adj_close"].dropna(how="all", axis=1)
    close_all = panel["close"].reindex_like(adj_all)
    vol_all = panel["volume"].reindex_like(adj_all)
    spy = adj_all["SPY"]
    adj = adj_all.drop(columns=["SPY"], errors="ignore")
    close = close_all.drop(columns=["SPY"], errors="ignore")
    volume = vol_all.drop(columns=["SPY"], errors="ignore")
    print(f"      {adj.shape[1]}/{len(tickers)} names have price history")

    # Base eligibility: not an excluded sector, and (for PIT) an index member that day.
    excluded = sector_src().reindex(adj.columns).isin(EXCLUDE_SECTORS)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(
        excluded, index=adj.columns
    )
    if universe == "sp500_pit":
        base = base & sp500_pit_eligible(adj.index, list(adj.columns))

    print(f"[2/4] EDGAR fundamentals ({adj.shape[1]} names) …")
    f = load_fundamentals(list(adj.columns), start, end, items=_ITEMS,
                          sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    # Names with a market cap on that day are the tradeable base (also drops non-filers).
    base = base & mcap.notna()

    print("[3/4] factor families + size buckets …")
    families = [
        [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)],   # value
        [revenue_growth(f), fcf_growth(f)],                  # growth
        [residual_momentum(adj)],                            # momentum
    ]
    buckets = {
        "all sizes":       base,
        "small (0-33%)":   size_bucket(mcap, base, 0.0, 1 / 3),
        "mid (33-67%)":    size_bucket(mcap, base, 1 / 3, 2 / 3),
        "large (67-100%)": size_bucket(mcap, base, 2 / 3, 1.0),
    }

    print("[4/4] backtests …\n")
    net = {}
    for name, elig in buckets.items():
        rank = combine_ranks(families, elig)
        net[name] = long_only_backtest(rank, adj, volume, close, rebalance, top_n)

    spy_ret = spy.pct_change(fill_method=None)
    common = None
    for r in net.values():
        idx = r.replace(0.0, np.nan).dropna().index
        common = idx if common is None else common.intersection(idx)

    # Median market cap actually held in each bucket, for context.
    med_mcap = {n: float(mcap.where(e).median(axis=1).median() / 1e9) for n, e in buckets.items()}

    print("=" * 86)
    print(f"SIZE EFFECT  ({start}→{end}, {label}, top-{top_n}, {rebalance}, net)")
    print("=" * 86)
    print(f"  {'size bucket':17s} {'med_mcap':>9s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s}")
    for name, r in net.items():
        s = summary_stats(r.reindex(common).fillna(0.0))
        print(f"  {name:17s} {med_mcap[name]:>7.1f}B {s['ann_return']:>+8.2%} "
              f"{s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")
    s = summary_stats(spy_ret.reindex(common).fillna(0.0))
    print(f"  {'SPY buy & hold':17s} {'—':>8s} {s['ann_return']:>+8.2%} "
          f"{s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")

    out = ROOT / "results" / "size_effect"
    out.mkdir(parents=True, exist_ok=True)
    tag = "sp500pit" if universe == "sp500_pit" else "sp1500"
    pd.DataFrame(net).to_csv(out / f"size_bucket_net_returns_{tag}.csv")
    print(f"\nwrote {out}/size_bucket_net_returns_{tag}.csv")


if __name__ == "__main__":
    uni = sys.argv[1] if len(sys.argv) > 1 else "sp1500"
    main(universe=uni)
