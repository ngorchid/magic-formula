"""Market-cap ceiling scan of the enhanced (best-version) Magic Formula on the S&P 1500.

For each maximum market-cap cutoff, restrict the eligible universe to names *below* it
(a small-cap tilt), re-rank within that restricted set, and run the canonical strategy.
Shows how the small-cap cutoff impacts the result. Survivorship-biased universe (current
S&P 1500) — accepted, as agreed. Reports eligible-name counts + median held market cap so
the thin low cutoffs (the S&P 1500's smallest names are ~$1B) are transparent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from data import download_ohlcv, load_fundamentals, sp1500_sectors, sp1500_tickers
from strategies.magic_formula import ENHANCED_ITEMS, EnhancedMagicConfig, enhanced_rank
from strategies.magic_formula.construct import pnl, weights_banded

# Ceiling: only hold names BELOW the cutoff (small-cap tilt).
CEILINGS = {
    "<$100M": 1e8, "<$500M": 5e8, "<$1B": 1e9, "<$2B": 2e9, "<$5B": 5e9,
    "<$10B": 1e10, "<$20B": 2e10, "none": np.inf,
}
# Floor: only hold names ABOVE the cutoff (raise the minimum size, i.e. exclude small names).
FLOORS = {
    "all": 0.0, ">$100M": 1e8, ">$500M": 5e8, ">$1B": 1e9, ">$2B": 2e9,
    ">$5B": 5e9, ">$10B": 1e10, ">$20B": 2e10, ">$50B": 5e10,
}


def main(start: str = "2012-01-01", end: str | None = None, use_graham: bool = True,
         mode: str = "ceiling") -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg = EnhancedMagicConfig(use_graham=use_graham)
    cuts = FLOORS if mode == "floor" else CEILINGS
    keep = (lambda m, c: m >= c) if mode == "floor" else (lambda m, c: m <= c)
    col = "min cap" if mode == "floor" else "max cap"
    tickers = sp1500_tickers()
    full = sorted(set(tickers + ["SPY"]))
    print(f"[load] S&P 1500 prices {start}→{end} …")
    panel = download_ohlcv(full, start, end)
    adj = panel["adj_close"].dropna(how="all", axis=1)
    close = panel["close"].reindex_like(adj)
    volume = panel["volume"].reindex_like(adj)
    spy = adj["SPY"].pct_change(fill_method=None)
    adj = adj.drop(columns=["SPY"], errors="ignore")
    close = close.drop(columns=["SPY"], errors="ignore")
    volume = volume.drop(columns=["SPY"], errors="ignore")
    excluded = sp1500_sectors().reindex(adj.columns).isin(cfg.exclude_sectors)
    base0 = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    print("[load] EDGAR fundamentals …")
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS,
                          sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base0 = base0 & mcap.notna()

    print(f"[run] cap-{mode} scan …\n")
    rows = []
    for label, cut in cuts.items():
        elig = base0 & keep(mcap, cut)
        rank = enhanced_rank(f, mcap, adj, elig, cfg)
        w = weights_banded(rank.where(elig), adj, cfg.rebalance, cfg.top_n, cfg.hold_n)
        net, turn = pnl(w, adj, volume, close)
        n_elig = int(np.nan_to_num(elig.sum(axis=1).replace(0, np.nan).median()))
        med_held = float(mcap.where(w > 0).median(axis=1).median() / 1e9)
        traded = bool((w.abs().sum(axis=1) > 0).any())
        rows.append((label, cut, net, turn, n_elig, med_held, traded))

    # Common window across buckets that actually traded (thin buckets excluded from it).
    common = None
    for row in rows:
        net, traded = row[2], row[6]
        if not traded:
            continue
        idx = net.replace(0.0, np.nan).dropna().index
        common = idx if common is None else common.intersection(idx)

    print("=" * 88)
    print(f"CAP-{mode.upper()} SCAN — enhanced Magic Formula, S&P 1500 [survivorship-biased] "
          f"(graham={use_graham})")
    print("=" * 88)
    print(f"  {col:8s} {'#elig':>6s} {'med_held':>9s} {'ann_ret':>8s} {'vol':>7s} "
          f"{'sharpe':>7s} {'maxDD':>8s} {'turn':>6s}")
    for label, cut, net, turn, n_elig, med_held, traded in rows:
        if not traded or n_elig < cfg.top_n:
            print(f"  {label:8s} {n_elig:>6d}  — too few names (<{cfg.top_n}) to hold a book")
            continue
        s = summary_stats(net.reindex(common).fillna(0.0))
        mh = f"{med_held:>7.1f}B" if med_held == med_held else "     —"
        print(f"  {label:8s} {n_elig:>6d} {mh:>9s} {s['ann_return']:>+8.2%} "
              f"{s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%} {turn:>5.1f}x")
    s = summary_stats(spy.reindex(common).fillna(0.0))
    print(f"  {'SPY':8s} {'—':>6s} {'—':>9s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
          f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")

    out = ROOT / "results" / "capscan"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({l: net for l, _, net, *_ in rows}).to_csv(out / f"capscan_{mode}_net.csv")
    print(f"\n  wrote {out}/capscan_{mode}_net.csv")


if __name__ == "__main__":
    m = "floor" if "floor" in sys.argv else "ceiling"
    main(use_graham="--no-graham" not in sys.argv, mode=m)
