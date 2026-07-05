"""Growth as a hard gate vs. a soft factor — and does gating reduce survivorship reliance?

Two questions (see the brainstorm in memory current-focus):

  1. Does turning growth from a *soft rank* into a *hard filter* (require revenue YoY > 0
     AND FCF growth > 0, then rank survivors on value + momentum) improve the strategy,
     especially on clean (survivorship-corrected) data? The soft blend lets a cheap-but-
     shrinking value trap through when its value rank overcompensates; the gate refuses it.

  2. Does gating make the strategy *less dependent on survivorship bias*? A collapsing
     company shows declining revenue / negative FCF before it delists, so a positive-growth
     gate should screen out exactly the names the biased data wrongly omits. If so, gating
     should shrink the biased→clean Sharpe gap (and barely dent the biased number if the
     soft factors were already avoiding those names).

Runs gated vs ungated on the biased S&P 1500 (small-cap bucket + all sizes) and the clean
point-in-time S&P 500, and prints the gate's surviving-name count so we can spot thinness.
Ungated = value+growth+momentum (growth soft). Gated = value+momentum, growth as filter.
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
    growth_gate,
    long_only_backtest,
    size_bucket,
)

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

    rev_g, fcf_g = revenue_growth(f), fcf_growth(f)
    families = {
        "value": [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)],
        "growth": [rev_g, fcf_g],
        "momentum": [residual_momentum(adj)],
    }
    return adj, close, volume, spy, base, mcap, families, rev_g, fcf_g


def _run(elig, families, adj, close, volume, gated, rev_g, fcf_g, rebalance, top_n):
    """Return (net_returns, gate_survivor_count_series). gated: growth as filter, not rank."""
    if gated:
        elig = growth_gate(elig, rev_g, fcf_g)
        rank = combine_ranks([families["value"], families["momentum"]], elig)
    else:
        rank = combine_ranks([families["value"], families["growth"], families["momentum"]], elig)
    net = long_only_backtest(rank, adj, volume, close, rebalance, top_n)
    return net, elig.sum(axis=1)


def main(start: str = "2012-01-01", end: str | None = None, top_n: int = 30,
         rebalance: str = "ME") -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    rows = []          # (universe_bucket, gated?, stats, survivor_count)
    spy_common = {}

    for universe, buckets in [
        ("sp1500", ["all", "small"]),
        ("sp500_pit", ["all"]),
    ]:
        print(f"[load] {universe} …")
        adj, close, volume, spy, base, mcap, families, rev_g, fcf_g = _load(universe, start, end)
        bucket_elig = {
            "all": base,
            "small": size_bucket(mcap, base, 0.0, 1 / 3),
        }
        for b in buckets:
            elig = bucket_elig[b]
            for gated in (False, True):
                net, surv = _run(elig, families, adj, close, volume, gated, rev_g, fcf_g,
                                 rebalance, top_n)
                rows.append((f"{universe}:{b}", gated, net, surv))
        spy_common[universe] = spy

    # Align each row's stats on its own valid window; report next to SPY of its universe.
    print("\n" + "=" * 92)
    print(f"GATED (growth as hard filter) vs UNGATED (growth soft)  ({start}→{end}, top-{top_n}, {rebalance}, net)")
    print("=" * 92)
    print(f"  {'universe:bucket':18s} {'mode':8s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} "
          f"{'maxDD':>8s} {'gate_names(med/min)':>19s}")
    for label, gated, net, surv in rows:
        idx = net.replace(0.0, np.nan).dropna().index
        s = summary_stats(net.reindex(idx).fillna(0.0))
        # survivor count only meaningful on gated rows, at rebalance points
        rb = pd.DatetimeIndex(net.index.to_series().resample(rebalance).last().dropna().values)
        sc = surv.reindex(rb).dropna()
        gate_str = f"{int(sc.median())}/{int(sc.min())}" if gated else "—"
        print(f"  {label:18s} {'GATED' if gated else 'ungated':8s} {s['ann_return']:>+8.2%} "
              f"{s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%} {gate_str:>19s}")

    # SPY reference per window.
    for uni, spy in spy_common.items():
        idx = spy.dropna().index
        s = summary_stats(spy.reindex(idx).fillna(0.0))
        print(f"  {'SPY ('+uni+')':18s} {'—':8s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")

    out = ROOT / "results" / "gated_growth"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{l}:{'gated' if g else 'ungated'}": n for l, g, n, _ in rows}).to_csv(
        out / "gated_vs_ungated_net_returns.csv"
    )
    print(f"\nwrote {out}/gated_vs_ungated_net_returns.csv")


if __name__ == "__main__":
    main()
