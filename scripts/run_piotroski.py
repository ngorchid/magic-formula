"""Piotroski F-score health screen — the proper version of the growth-gate idea.

Instead of the brittle "revenue>0 & fcf_growth>0" gate, screen on the 9-point Piotroski
F-score (financial strength: profitability, leverage/liquidity, operating efficiency),
then rank the survivors on value + momentum. Compared against the ungated baseline and the
old growth-gate, on the clean point-in-time S&P 500 and the biased S&P 1500 small-cap
bucket. Prints per-rebalance surviving-name counts so we can see if a screen gets too thin.
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
    piotroski_f_score,
    residual_momentum,
    revenue_growth,
)
from strategies.magic_formula.construct import combine_ranks, growth_gate, long_only_backtest, size_bucket

_ITEMS = [
    "revenue", "net_income", "total_assets", "gross_profit", "cogs",
    "operating_cash_flow", "capex", "shares_diluted",
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
        growth = [revenue_growth(f), fcf_growth(f)]
        mom = [residual_momentum(adj)]
        fscore = piotroski_f_score(f)

        # mode -> (eligibility, families ranked). Screens rank value+momentum on survivors;
        # baseline keeps growth as a soft factor.
        modes = {
            "baseline (soft growth)": (elig, [value, growth, mom]),
            "growth-gate (rev&fcf>0)": (growth_gate(elig, revenue_growth(f), fcf_growth(f)),
                                        [value, mom]),
            "Piotroski F>=6": (elig & (fscore >= 6), [value, mom]),
            "Piotroski F>=7": (elig & (fscore >= 7), [value, mom]),
        }
        for name, (e, fams) in modes.items():
            rank = combine_ranks(fams, e)
            net = long_only_backtest(rank, adj, volume, close, rebalance, top_n)
            rows.append((f"{universe}:{bucket}", name, net, e.sum(axis=1)))

    print("\n" + "=" * 92)
    print(f"PIOTROSKI HEALTH SCREEN  ({start}→{end}, top-{top_n}, {rebalance}, net)")
    print("=" * 92)
    cur = None
    for label, name, net, surv in rows:
        if label != cur:
            print(f"\n  [{label}]")
            cur = label
        idx = net.replace(0.0, np.nan).dropna().index
        s = summary_stats(net.reindex(idx).fillna(0.0))
        rb = pd.DatetimeIndex(net.index.to_series().resample(rebalance).last().dropna().values)
        sc = surv.reindex(rb).dropna()
        names = f"{int(sc.median())}/{int(sc.min())}" if "baseline" not in name else "—"
        print(f"    {name:26s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"sharpe {s['sharpe']:>+5.2f}  maxDD {s['max_drawdown']:>+7.2%}  names {names:>8s}")
    print()
    for uni, spy in spy_ref.items():
        s = summary_stats(spy.dropna())
        print(f"  SPY ({uni}): ret {s['ann_return']:+.2%}  sharpe {s['sharpe']:+.2f}  "
              f"maxDD {s['max_drawdown']:+.2%}")

    out = ROOT / "results" / "piotroski"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{l}|{n}": net for l, n, net, _ in rows}).to_csv(out / "piotroski_net_returns.csv")
    print(f"\nwrote {out}/piotroski_net_returns.csv")


if __name__ == "__main__":
    main()
