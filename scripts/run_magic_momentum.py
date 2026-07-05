"""Improved Magic Formula on the point-in-time universe — value + growth + momentum,
with an optional small-cap tilt.

Builds on the FCF Magic Formula (FCF replaces Greenblatt's EBIT) and layers on the
lessons from the earlier run:

  * **residual momentum** (Blitz-Huij-Martens) instead of raw 12-1 price momentum, and
    **monthly** rebalancing so the momentum signal isn't stale by the time we trade it.
  * a **growth** family (revenue + FCF growth) — is the cheap business actually
    expanding, or a value trap?
  * combine by **family** (value / growth / momentum each an equal-weighted block of
    cross-sectional percentile ranks), so adding a factor doesn't silently dilute value.
  * an optional **market-cap cap** (keep the smaller names) — Greenblatt's own point that
    the effect is strongest in smaller stocks. NB: our PIT universe is the S&P 500, so
    this is a "smaller-large-cap" tilt, not true small caps (which need a broader universe).

Everything runs on the point-in-time S&P 500 membership so results aren't survivorship-
inflated. Long-only, equal-weight top-N, benchmarked directly against SPY (β≈1 books).
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
)
from signals import (
    fcf_ev_yield,
    fcf_growth,
    fcf_return_on_capital,
    residual_momentum,
    revenue_growth,
)
from strategies.magic_formula.construct import combine_ranks, long_only_backtest, mcap_cap

_ITEMS = [
    "revenue", "operating_income", "operating_cash_flow", "capex", "shares_diluted",
    "short_term_debt", "long_term_debt", "cash",
    "total_current_assets", "total_current_liabilities", "ppe_net",
]
EXCLUDE_SECTORS = ("Financial Services", "Utilities")  # Greenblatt exclusions


def main(start: str = "2012-01-01", end: str | None = None, top_n: int = 30,
         small_cap_pctile: float = 0.6) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sp500_pit_universe(start, end)
    full = sorted(set(tickers + ["SPY"]))
    print(f"[1/4] prices {start}→{end} for {len(full)} PIT names …")
    panel = download_ohlcv(full, start, end)
    adj_all = panel["adj_close"].dropna(how="all", axis=1)
    close_all = panel["close"].reindex_like(adj_all)
    vol_all = panel["volume"].reindex_like(adj_all)
    spy = adj_all["SPY"]
    adj = adj_all.drop(columns=["SPY"], errors="ignore")
    close = close_all.drop(columns=["SPY"], errors="ignore")
    volume = vol_all.drop(columns=["SPY"], errors="ignore")
    print(f"      {adj.shape[1]}/{len(tickers)} names have price history")

    # PIT membership AND not an excluded sector (sectors known for current names only).
    pit = sp500_pit_eligible(adj.index, list(adj.columns))
    excluded = sp500_sectors().reindex(adj.columns).isin(EXCLUDE_SECTORS)
    eligible = pit & ~pd.Series(excluded, index=adj.columns)

    print(f"[2/4] EDGAR fundamentals ({adj.shape[1]} names) …")
    f = load_fundamentals(list(adj.columns), start, end, items=_ITEMS,
                          sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)

    print("[3/4] factor families …")
    value = [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)]
    growth = [revenue_growth(f), fcf_growth(f)]
    momentum = [residual_momentum(adj)]

    small = mcap_cap(mcap, eligible, small_cap_pctile)
    variants = {
        "FCF value, annual":                 (combine_ranks([value], eligible), "YE"),
        "FCF value, monthly":                (combine_ranks([value], eligible), "ME"),
        "FCF + growth":                      (combine_ranks([value, growth], eligible), "ME"),
        "FCF + resid-mom":                   (combine_ranks([value, momentum], eligible), "ME"),
        "FCF + growth + resid-mom":          (combine_ranks([value, growth, momentum], eligible), "ME"),
        "FCF + growth + mom + small-cap":    (combine_ranks([value, growth, momentum], small), "ME"),
    }

    print("[4/4] backtests …\n")
    net = {n: long_only_backtest(r, adj, volume, close, rb, top_n)
           for n, (r, rb) in variants.items()}
    spy_ret = spy.pct_change(fill_method=None)

    common = None
    for r in net.values():
        idx = r.replace(0.0, np.nan).dropna().index
        common = idx if common is None else common.intersection(idx)

    print("=" * 84)
    print(f"MAGIC FORMULA v2  ({start}→{end}, PIT universe, top-{top_n}, long-only, net)")
    print("=" * 84)
    print(f"  {'variant':34s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'corrSPY':>8s}")
    for name, r in net.items():
        rr = r.reindex(common).fillna(0.0)
        s = summary_stats(rr)
        c = rr.corr(spy_ret.reindex(common).fillna(0.0))
        print(f"  {name:34s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%} {c:>+8.3f}")
    s = summary_stats(spy_ret.reindex(common).fillna(0.0))
    print(f"  {'SPY buy & hold':34s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
          f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%} {1.0:>+8.3f}")

    out = ROOT / "results" / "magic_momentum"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(net).to_csv(out / "v2_net_returns.csv")
    print(f"\nwrote {out}/v2_net_returns.csv")


if __name__ == "__main__":
    main()
