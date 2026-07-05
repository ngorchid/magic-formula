"""Run THE canonical enhanced Magic Formula (the accumulated best version).

Backtests the single strategy defined in strategies/magic_formula/enhanced.py and prints
both the performance and the *current target holdings* — the actual names to trade (e.g. to
seed IBKR paper trading).

Usage:
    python scripts/run_best_magic.py                 # clean PIT S&P 500 (default)
    python scripts/run_best_magic.py sp1500 small    # broad universe, small-cap tercile
    python scripts/run_best_magic.py sp500_pit all --no-graham
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
from strategies.magic_formula import (
    ENHANCED_ITEMS,
    EnhancedMagicConfig,
    current_targets,
    enhanced_weights,
)
from strategies.magic_formula.construct import pnl, size_bucket


def _load(universe: str, cfg: EnhancedMagicConfig, start: str, end: str):
    if universe == "sp500_pit":
        tickers, sector_src, label = sp500_pit_universe(start, end), sp500_sectors, "S&P 500 (point-in-time)"
    else:
        tickers, sector_src, label = sp1500_tickers(), sp1500_sectors, "S&P 1500 (current)"
    full = sorted(set(tickers + ["SPY"]))
    panel = download_ohlcv(full, start, end)
    adj = panel["adj_close"].dropna(how="all", axis=1)
    close = panel["close"].reindex_like(adj)
    volume = panel["volume"].reindex_like(adj)
    spy = adj["SPY"].pct_change(fill_method=None)
    adj = adj.drop(columns=["SPY"], errors="ignore")
    close = close.drop(columns=["SPY"], errors="ignore")
    volume = volume.drop(columns=["SPY"], errors="ignore")
    excluded = sector_src().reindex(adj.columns).isin(cfg.exclude_sectors)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    if universe == "sp500_pit":
        base = base & sp500_pit_eligible(adj.index, list(adj.columns))
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS,
                          sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()
    return adj, close, volume, spy, base, mcap, f, label


def main(universe: str = "sp500_pit", bucket: str = "all", use_graham: bool = True,
         start: str = "2012-01-01", end: str | None = None) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg = EnhancedMagicConfig(use_graham=use_graham)
    print(f"[load] {universe} …")
    adj, close, volume, spy, base, mcap, f, label = _load(universe, cfg, start, end)
    elig = base if bucket == "all" else size_bucket(mcap, base, 0.0, 1 / 3)

    weights, rank = enhanced_weights(f, mcap, adj, elig, cfg)
    net, turnover = pnl(weights, adj, volume, close)

    idx = net.replace(0.0, np.nan).dropna().index
    s = summary_stats(net.reindex(idx).fillna(0.0))
    sp = summary_stats(spy.reindex(idx).fillna(0.0))

    print("\n" + "=" * 74)
    print(f"ENHANCED MAGIC FORMULA — {label}, {bucket}  ({start}→{end}, net)")
    print("=" * 74)
    print(f"  config: top {cfg.top_n} / band {cfg.hold_n} / {cfg.rebalance} / "
          f"mom {cfg.momentum_lookback}-{cfg.momentum_skip} / graham={cfg.use_graham}")
    print(f"  strategy   ann_ret {s['ann_return']:>+7.2%}  vol {s['ann_vol']:>6.2%}  "
          f"sharpe {s['sharpe']:>+5.2f}  maxDD {s['max_drawdown']:>+7.2%}  turnover {turnover:.1f}x/yr")
    print(f"  SPY        ann_ret {sp['ann_return']:>+7.2%}  vol {sp['ann_vol']:>6.2%}  "
          f"sharpe {sp['sharpe']:>+5.2f}  maxDD {sp['max_drawdown']:>+7.2%}")

    picks = current_targets(weights, rank)
    asof = weights.index[-1].date()
    print(f"\n  CURRENT TARGET HOLDINGS as of {asof}  ({len(picks)} names, equal-weight "
          f"{100/max(len(picks),1):.1f}% each), ordered by rank:")
    line = "   "
    for i, t in enumerate(picks.index, 1):
        line += f"{t:<7}"
        if i % 8 == 0:
            print("   " + line.strip()); line = "   "
    if line.strip():
        print("   " + line.strip())

    out = ROOT / "results" / "best_magic"
    out.mkdir(parents=True, exist_ok=True)
    net.to_frame("net_return").to_csv(out / f"best_{universe}_{bucket}.csv")
    picks.to_frame("rank").to_csv(out / f"targets_{universe}_{bucket}.csv")
    print(f"\n  wrote {out}/best_{universe}_{bucket}.csv and targets_{universe}_{bucket}.csv")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    uni = args[0] if len(args) > 0 else "sp500_pit"
    buck = args[1] if len(args) > 1 else "all"
    main(universe=uni, bucket=buck, use_graham="--no-graham" not in sys.argv)
