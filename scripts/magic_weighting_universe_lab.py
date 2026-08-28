"""Lab: does the LIVE inverse-vol tilt earn its place, and does the universe decide it?

Two loose ends from documenting the book (2026-08-24).

1. The weighting lab found equal weight beats `inverse_vol` on PIT S&P 500 (OOS Sharpe
   0.942 vs 0.878), yet the LIVE book runs an inverse-vol tilt. Why would it have been
   implemented if it makes things worse? Hypothesis: it was beneficial on the BROADER
   universe, where the vol cross-section is much wider.

2. The lab's `inverse_vol` is `raw = 1/sd` -- UNBOUNDED. The live tilt is
   `clip(median_vol/vol, 0.5, 2.0)` then normalised to mean 1, i.e. bounded to a
   factor-of-four spread. The lab therefore tested a MORE AGGRESSIVE scheme than the one
   deployed, so its verdict may not transfer.

This runs equal / unbounded inverse-vol / the actual live tilt, on S&P 500 PIT and on
S&P 1500, with selection held identical so any difference is sizing alone.

NOTE the sp1500 universe is CURRENT constituents (survivorship-biased). Its levels are
not comparable to sp500_pit; only the RANKING of schemes within a universe is meaningful.

Run: python scripts/magic_weighting_universe_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import summary_stats
from strategies.magic_formula import EnhancedMagicConfig, enhanced_rank
from strategies.magic_formula.construct import _rebal_dates, pnl, size_bucket
from run_best_magic import _load

CAP = 0.10
SPLIT = "2019-07-01"
LIVE_CLIP = (0.5, 2.0)


def build(rank, adj, vol, top_n, hold_n, scheme):
    """weights_banded with pluggable weighting. Selection identical across schemes."""
    cal = adj.index
    target = pd.DataFrame(np.nan, index=cal, columns=adj.columns)
    held: list[str] = []
    for dt in _rebal_dates(cal, "ME"):
        row = rank.loc[dt].dropna()
        if len(row) < top_n:
            continue
        pos = pd.Series(range(len(row)), index=row.sort_values(ascending=False).index)
        keep = [t for t in held if t in pos.index and pos[t] < hold_n]
        need = top_n - len(keep)
        if need > 0:
            held = keep + [t for t in pos.sort_values().index if t not in keep][:need]
        else:
            held = sorted(keep, key=lambda t: pos[t])[:top_n]

        sd = vol.loc[dt, held]
        sd = sd.where(sd > 0).fillna(sd.median() if sd.notna().any() else 1.0)
        if scheme == "equal":
            raw = pd.Series(1.0, index=held)
        elif scheme == "inverse_vol_unbounded":       # what the original lab tested
            raw = 1.0 / sd
        elif scheme == "live_tilt":                   # what paper/orchestrator.py does
            ref = float(np.nanmedian(sd.values))
            t = pd.Series(np.clip(ref / sd, *LIVE_CLIP), index=held).fillna(1.0)
            raw = t / float(np.nanmean(t.values))     # normalise to mean exactly 1
        else:
            raise ValueError(scheme)
        w = (raw / raw.sum()).clip(upper=CAP)
        w = w / w.sum()
        out = pd.Series(0.0, index=adj.columns)
        out.loc[held] = w.values
        target.loc[dt] = out.values
    return target.ffill().fillna(0.0).shift(1).fillna(0.0)


# Live sleeve reality: $50k budget, IB $1.00 minimum per US equity order. The default
# backtest assumes a $1,000,000 book, where the fixed floor is ~0.3bp and invisible.
BOOK_SIZES = [(1_000_000, 0.0, "$1m book, proportional costs only (the BACKTEST default)"),
              (50_000, 1.00, "$50k book, + $1.00/order fixed fee (the LIVE sleeve)")]


def run_universe(universe: str, bucket: str, label: str) -> None:
    # Graham OFF: matches scripts/run_paper.py, i.e. the factor set actually deployed.
    # With it on there are 4 families at 1/4; off there are 3 at 1/3, so the whole rank
    # differs and a weighting comparison run with it on is not the live book's comparison.
    cfg = EnhancedMagicConfig(use_graham=False)
    print(f"\n[load] {label} …")
    adj, close, volume, spy, base, mcap, f, _ = _load(
        universe, cfg, "2012-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    elig = base if bucket == "all" else size_bucket(mcap, base, 0.0, 1 / 3)
    rank = enhanced_rank(f, mcap, adj, elig, cfg)
    vol = adj.pct_change(fill_method=None).rolling(cfg.vol_window).std()

    for notional, fee, size_label in BOOK_SIZES:
        _run_costs(rank, adj, volume, close, vol, cfg, label, notional, fee, size_label)


def _run_costs(rank, adj, volume, close, vol, cfg, label, notional, fee, size_label) -> None:
    print("=" * 104)
    print(f"{label}")
    print(f"  {size_label}")
    print("=" * 104)
    print(f"  {'scheme':26s} {'turn':>5s} {'maxwt':>6s} | {'Sh FULL':>8s} {'ret':>7s} "
          f"{'dd':>7s} | {'Sh IS':>7s} | {'Sh OOS':>7s} {'ret OOS':>8s}")
    nets = {}
    for scheme in ("equal", "inverse_vol_unbounded", "live_tilt"):
        w = build(rank, adj, vol, cfg.top_n, cfg.hold_n, scheme)
        net, turn = pnl(w, adj, volume, close, notional=notional, fixed_fee=fee)
        idx = net.replace(0.0, np.nan).dropna().index
        net = net.reindex(idx).fillna(0.0)
        nets[scheme] = net
        full = summary_stats(net)
        is_ = summary_stats(net.loc[:SPLIT])
        oos = summary_stats(net.loc[SPLIT:])
        mw = float(w.max().max())
        print(f"  {scheme:26s} {turn:5.2f} {mw:6.3f} | {full['sharpe']:+8.3f} "
              f"{full['ann_return']:+7.2%} {full['max_drawdown']:+7.1%} | "
              f"{is_['sharpe']:+7.3f} | {oos['sharpe']:+7.3f} {oos['ann_return']:+8.2%}")

    # Paired tests vs equal -- the arms share most of their positions.
    print("\n  paired difference vs equal (daily):")
    base_net = nets["equal"]
    for scheme, net in nets.items():
        if scheme == "equal":
            continue
        d = (net - base_net).dropna()
        t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        print(f"    {scheme:26s} ann {d.mean()*252:+7.2%}  vol {d.std()*np.sqrt(252):6.2%}  "
              f"t={t:+6.2f}   {'significant' if abs(t) > 2 else 'not distinguishable'}")


def main() -> None:
    run_universe("sp500_pit", "all", "S&P 500 point-in-time (the BACKTEST universe)")
    run_universe("sp1500", "all", "S&P 1500 current constituents (survivorship-biased)")
    run_universe("sp1500", "small", "S&P 1500 SMALL-CAP tercile (widest vol cross-section)")
    print("\n  NB the live book is US + eurozone large caps, which is not testable here: "
          "\n  european_eur_tickers() is CURRENT constituents with no PIT membership data.")


if __name__ == "__main__":
    main()
