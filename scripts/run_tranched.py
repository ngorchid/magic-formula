"""Overlapping tranches — stagger entries daily instead of one lumpy monthly rebalance.

Rather than buying all 30 on one day and holding a month (exposed to that single day's
prices), build the book in daily tranches: each day hold the AVERAGE of the last 21 days'
top-30 target portfolios, so ~1/21 of the book rolls over daily and every position lives
~1 month. This averages away timing luck and should smooth the equity curve.

Compares, on the same enhanced signal (inverse-vol weighted, no Graham):
  * monthly single-day rebalance (top-30)
  * monthly + no-trade band (the current best version)
  * daily overlapping tranches (21-day)
Clean PIT S&P 500. Reports Sharpe/vol/drawdown/turnover.
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
    download_ohlcv, load_fundamentals, sp500_pit_eligible, sp500_pit_universe, sp500_sectors,
)
from strategies.magic_formula import ENHANCED_ITEMS, EnhancedMagicConfig, enhanced_rank
from strategies.magic_formula.construct import pnl, weights_banded, weights_top_n

TOP_N, HOLD_DAYS = 30, 21


def _tranched(rank, base, adj, vol, top_n=TOP_N, hold_days=HOLD_DAYS):
    """Daily overlapping tranches: hold the rolling mean of daily top-N target weights."""
    r = rank.where(base)
    is_top = r.rank(axis=1, ascending=False, method="first") <= top_n
    raw = is_top.astype(float) * (1.0 / vol).where(vol > 0)
    daily = raw.div(raw.sum(axis=1), axis=0)             # daily inverse-vol top-N target
    held = daily.rolling(hold_days, min_periods=1).mean()  # overlapping tranches
    return held.shift(1).fillna(0.0)


def main(start: str = "2012-01-01", end: str | None = None) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg = EnhancedMagicConfig(use_graham=False, weighting="inverse_vol")
    tickers = sp500_pit_universe(start, end)
    full = sorted(set(tickers + ["SPY"]))
    print(f"[load] PIT S&P 500 {start}→{end} …")
    panel = download_ohlcv(full, start, end)
    adj = panel["adj_close"].dropna(how="all", axis=1)
    spy = adj["SPY"].pct_change(fill_method=None)
    close = panel["close"].reindex_like(adj).drop(columns=["SPY"], errors="ignore")
    volume = panel["volume"].reindex_like(adj).drop(columns=["SPY"], errors="ignore")
    adj = adj.drop(columns=["SPY"], errors="ignore")
    excluded = sp500_sectors().reindex(adj.columns).isin(cfg.exclude_sectors)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    base = base & sp500_pit_eligible(adj.index, list(adj.columns))
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()

    rank = enhanced_rank(f, mcap, adj, base, cfg)
    vol = adj.pct_change(fill_method=None).rolling(cfg.vol_window).std()
    r = rank.where(base)

    variants = {
        "monthly single-rebalance": weights_top_n(r, adj, "ME", TOP_N),  # equal-wt (no vol arg here)
        "monthly + no-trade band": weights_banded(r, adj, "ME", TOP_N, 45, vol=vol),
        "daily overlapping tranches": _tranched(rank, base, adj, vol),
    }

    common, results = None, {}
    for name, w in variants.items():
        net, turn = pnl(w, adj, volume, close)
        results[name] = (net, turn)
        idx = net.replace(0.0, np.nan).dropna().index
        common = idx if common is None else common.intersection(idx)

    print("\n" + "=" * 82)
    print(f"OVERLAPPING TRANCHES vs monthly  ({start}→{end}, PIT S&P 500, inverse-vol, net)")
    print("=" * 82)
    print(f"  {'method':28s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'turn':>6s}")
    for name, (net, turn) in results.items():
        s = summary_stats(net.reindex(common).fillna(0.0))
        print(f"  {name:28s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
              f"{s['max_drawdown']:>+8.2%} {turn:>5.1f}x")
    s = summary_stats(spy.reindex(common).fillna(0.0))
    print(f"  {'SPY':28s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
          f"{s['max_drawdown']:>+8.2%}")

    out = ROOT / "results" / "tranched"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({n: r for n, (r, _) in results.items()}).to_csv(out / "tranched_net.csv")
    print(f"\n  wrote {out}/tranched_net.csv")


if __name__ == "__main__":
    main()
