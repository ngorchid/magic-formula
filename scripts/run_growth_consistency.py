"""Growth persistence — single-year YoY vs 2-year mean vs 2-year min (consistency).

Does requiring a longer-term growth track record help? The growth family is swapped between:
  * 1yr        — current: this-year YoY of revenue & FCF (noisy single snapshot)
  * 2yr mean   — average of the last two YoY growths (smooths one-off years)
  * 2yr min    — the WORSE of the last two YoY growths, so a name must have grown in
                 BOTH years to rank high (a direct consistency requirement)
Everything else (FCF value + residual momentum, top-30 monthly) is held fixed. Clean PIT
S&P 500 and biased small-cap.
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
from signals import (
    fcf_ev_yield,
    fcf_growth,
    fcf_return_on_capital,
    free_cash_flow,
    multi_year_growth,
    residual_momentum,
    revenue_growth,
)
from strategies.magic_formula.construct import combine_ranks, long_only_backtest, size_bucket

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

        value = [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)]
        mom = [residual_momentum(adj)]
        rev, fcf = f["revenue"], free_cash_flow(f)
        growth_variants = {
            "1yr (current)": [revenue_growth(f), fcf_growth(f)],
            "2yr mean":      [multi_year_growth(rev, 2, "mean"), multi_year_growth(fcf, 2, "mean", symmetric=True)],
            "2yr min (consistency)": [multi_year_growth(rev, 2, "min"), multi_year_growth(fcf, 2, "min", symmetric=True)],
        }
        for gname, growth in growth_variants.items():
            rank = combine_ranks([value, growth, mom], elig)
            net = long_only_backtest(rank, adj, volume, close, rebalance, top_n)
            rows.append((f"{universe}:{bucket}", gname, net))

    print("\n" + "=" * 82)
    print(f"GROWTH PERSISTENCE  ({start}→{end}, FCF+growth+mom, top-{top_n}, {rebalance}, net)")
    print("=" * 82)
    cur = None
    for label, gname, net in rows:
        if label != cur:
            print(f"\n  [{label}]")
            cur = label
        idx = net.replace(0.0, np.nan).dropna().index
        s = summary_stats(net.reindex(idx).fillna(0.0))
        print(f"    {gname:24s} ret {s['ann_return']:>+7.2%}  sharpe {s['sharpe']:>+5.2f}  "
              f"maxDD {s['max_drawdown']:>+7.2%}")
    print()
    for uni, spy in spy_ref.items():
        s = summary_stats(spy.dropna())
        print(f"  SPY ({uni}): ret {s['ann_return']:+.2%}  sharpe {s['sharpe']:+.2f}")

    out = ROOT / "results" / "growth_consistency"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{l}|{g}": net for l, g, net in rows}).to_csv(out / "growth_consistency_net.csv")
    print(f"\nwrote {out}/growth_consistency_net.csv")


if __name__ == "__main__":
    main()
