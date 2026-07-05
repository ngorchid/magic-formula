"""Winners/losers autopsy — how concentrated are outcomes, and were the losers flaggable?

For the FCF+growth+momentum book on the (broad, survivorship-biased) S&P 1500, compute each
name's total contribution to the strategy's return (Σ weight·return over the days held),
then:
  * concentration — do a few names drive the P&L (positive skew), and how big is the loser drag?
  * the biggest losers — and their factor readings AT ENTRY and averaged while held
    (Piotroski F-score, residual momentum, revenue/FCF growth, leverage) vs the winners,
    to ask: could a signal we already have flagged them going in?

CAVEAT printed at the end: the *worst* losers (outright bankruptcies) are absent from the
survivorship-biased data, so this understates the true left tail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import download_ohlcv, load_fundamentals, sp1500_sectors, sp1500_tickers
from signals import (
    fcf_ev_yield,
    fcf_growth,
    fcf_return_on_capital,
    piotroski_f_score,
    residual_momentum,
    revenue_growth,
)
from strategies.magic_formula.construct import combine_ranks, weights_top_n

_ITEMS = [
    "revenue", "net_income", "total_assets", "gross_profit", "cogs", "total_equity",
    "operating_cash_flow", "capex", "shares_diluted",
    "short_term_debt", "long_term_debt", "cash",
    "total_current_assets", "total_current_liabilities", "ppe_net",
]
EXCLUDE_SECTORS = ("Financial Services", "Utilities")


def _entry_value(panel: pd.DataFrame, held: pd.DataFrame, name: str):
    """Factor value on the first day `name` was held (NaN if never/again unavailable)."""
    col = held[name]
    if not col.any():
        return np.nan
    d0 = col[col].index[0]
    return panel.at[d0, name] if name in panel.columns and d0 in panel.index else np.nan


def main(start: str = "2012-01-01", end: str | None = None, top_n: int = 30) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sp1500_tickers()
    full = sorted(set(tickers + ["SPY"]))
    print(f"[1/3] prices + fundamentals for {len(full)} S&P 1500 names …")
    panel = download_ohlcv(full, start, end)
    adj = panel["adj_close"].dropna(how="all", axis=1).drop(columns=["SPY"], errors="ignore")
    close = panel["close"].reindex_like(adj)
    volume = panel["volume"].reindex_like(adj)
    excluded = sp1500_sectors().reindex(adj.columns).isin(EXCLUDE_SECTORS)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    f = load_fundamentals(list(adj.columns), start, end, items=_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()

    print("[2/3] build book (FCF+growth+momentum, top-30 monthly) …")
    value = [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)]
    growth = [revenue_growth(f), fcf_growth(f)]
    mom = residual_momentum(adj)
    fscore = piotroski_f_score(f)
    lev = f["long_term_debt"].fillna(0.0) / f["total_assets"]
    val = fcf_ev_yield(f, mcap)
    rank = combine_ranks([value, growth, [mom]], base)
    w = weights_top_n(rank, adj, "ME", top_n)

    rets = adj.where(adj > 0).pct_change(fill_method=None).where(lambda x: x.abs() < 1.0).fillna(0.0)
    contrib = (w * rets).sum(axis=0)                 # total P&L contribution per name (return pts)
    held = w > 0
    days_held = held.sum(axis=0)
    contrib = contrib[days_held > 0].sort_values()

    print("\n[3/3] results")
    n_pos = int((contrib > 0).sum()); n_neg = int((contrib < 0).sum())
    top = contrib.nlargest(15); bot = contrib.nsmallest(15)
    print("=" * 78)
    print("CONCENTRATION (contribution = Σ weight·return over days held; in return points)")
    print("=" * 78)
    print(f"  names ever held: {len(contrib)}   positive: {n_pos}   negative: {n_neg}")
    print(f"  total contribution        : {contrib.sum():+.2f}")
    print(f"  top-15 winners contribute : {top.sum():+.2f}")
    print(f"  bottom-15 losers contribute: {bot.sum():+.2f}")
    print(f"  top-15 as % of gross gains: {top.sum()/contrib[contrib>0].sum():.0%}")

    def facts(name):
        return dict(
            contrib=contrib[name], days=int(days_held[name]),
            mom_entry=_entry_value(mom, held, name), F_entry=_entry_value(fscore, held, name),
            revg_entry=_entry_value(revenue_growth(f), held, name),
            mom_avg=float(mom.where(held)[name].mean()), F_avg=float(fscore.where(held)[name].mean()),
            lev_avg=float(lev.where(held)[name].mean()),
        )

    print("\n  BIGGEST LOSERS — entry & held-average factor readings")
    print(f"  {'ticker':8s}{'contrib':>8s}{'days':>6s}{'mom@in':>8s}{'F@in':>6s}{'revg@in':>9s}{'mom~':>7s}{'F~':>5s}{'lev~':>6s}")
    for name in bot.index[:12]:
        d = facts(name)
        print(f"  {name:8s}{d['contrib']:>+8.2f}{d['days']:>6d}{d['mom_entry']:>8.2f}"
              f"{d['F_entry']:>6.1f}{d['revg_entry']:>+9.0%}{d['mom_avg']:>7.2f}{d['F_avg']:>5.1f}{d['lev_avg']:>6.2f}")

    # Group comparison: losers vs winners, average entry/held characteristics.
    def group_mean(names, fn, key):
        vals = [facts(n)[key] for n in names]
        vals = [v for v in vals if pd.notna(v)]
        return np.mean(vals) if vals else np.nan

    print("\n  GROUP AVERAGES (could a signal have separated them?)")
    print(f"  {'group':16s}{'mom@entry':>10s}{'F@entry':>9s}{'revg@entry':>11s}{'F~held':>8s}{'lev~held':>9s}")
    for gname, idx in [("bottom-15 losers", bot.index), ("top-15 winners", top.index)]:
        print(f"  {gname:16s}{group_mean(idx, facts, 'mom_entry'):>10.2f}"
              f"{group_mean(idx, facts, 'F_entry'):>9.1f}{group_mean(idx, facts, 'revg_entry'):>+11.0%}"
              f"{group_mean(idx, facts, 'F_avg'):>8.1f}{group_mean(idx, facts, 'lev_avg'):>9.2f}")

    print("\n  CAVEAT: survivorship-biased universe — the worst losers (bankruptcies/delistings)")
    print("  are absent from the data, so the true left tail is worse than shown here.")


if __name__ == "__main__":
    main()
