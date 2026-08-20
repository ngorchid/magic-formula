"""Reversal: BREADTH x REBALANCE FREQUENCY, jointly. Neither lever works alone.

WHY THIS GRID. The two levers were only ever tested separately, and each looked dead:

  CONCENTRATION alone (reversal_fees_lab, DAILY rebalance): IB cost was ~flat across book
  widths (7.7% -> 6.6%) while gross Sharpe fell 0.98 -> 0.47. Concluded "strictly harmful".

  FREQUENCY alone (reversal_turnover_lab, 200 names): cost fell 12x and monthly rebalancing
  produced the first positive net Sharpe, but gross collapsed 0.77 -> 0.34.

The reason concentration looked flat is the $1 ORDER MINIMUM. At 200 names on $1M each
position is ~$5,000, a full turn is ~100 shares, and $0.005/sh = $0.50 -- so you pay the $1
MINIMUM, i.e. ABOVE the per-share rate. Concentrating to 20 names makes positions 10x bigger,
which lifts you off the minimum and onto the rate, where cost is proportional to NOTIONAL and
no longer falls with name count. That is why the two effects cancelled.

Periodic rebalancing changes the regime: trades get SMALLER (a fraction of the position, not a
full turn), pushing you back UNDER the minimum -- where cost is proportional to ORDER COUNT, and
halving the names really does halve the cost. So the levers should compose, and the grid is the
only way to see it.

Reports NET-of-IB Sharpe. Weights are built ONCE (they depend on neither lever) and the grid is
swept over them, so this is cheap despite the cell count.

Run: python3 scripts/reversal_grid_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from backtest import LinearCostModel, summary_stats      # noqa: E402
from data import download_ohlcv                           # noqa: E402
from data.universe import (sp1500_constituents, sp1500_sectors,  # noqa: E402
                           sp1500_tickers)
from strategies.equity_mn.neutralize import rolling_beta  # noqa: E402
from reversal_lab import Variant, build_weights           # noqa: E402
from reversal_fees_lab import concentrate                 # noqa: E402

NAMES = [200, 100, 50, 30, 20]          # total (half per side)
KS = [1, 3, 5, 10, 21]                  # rebalance every k trading days


def evaluate_periodic(w, prices, volume, budget, k, spread_bps=1.5):
    """Shares are SET on rebalance days and CARRIED. See reversal_turnover_lab for why the
    shared evaluate() cannot be used: it re-derives shares from drifting prices every day, which
    fabricates an order daily even when the target is frozen."""
    px = prices.reindex_like(w).ffill()
    rebal = pd.Series(False, index=w.index)
    rebal.iloc[::k] = True
    tgt = (w * budget).div(px).round().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shares = tgt.where(rebal, np.nan).ffill().fillna(0.0)
    traded = (shares - shares.shift(1)).abs().fillna(shares.abs())
    pnl = (shares.shift(1) * px.diff()).sum(axis=1) / budget
    notional = traded * px
    adv = (prices * volume).rolling(21).mean().reindex_like(w).ffill()
    adv = adv.fillna(adv.median().median())
    pct = LinearCostModel(spread_bps, 10.0).charge(notional, adv) / budget
    per = np.minimum(0.005 * traded, 0.01 * notional).where(traded > 0, 0.0).clip(lower=0.0)
    per = per.mask((traded > 0) & (per < 1.0), 1.0)          # $1 order minimum
    ib = per.sum(axis=1) / budget
    # How often the MINIMUM binds, rather than the per-share rate -- the whole mechanism here.
    at_min = float(((traded > 0) & (0.005 * traded < 1.0)).sum().sum()
                   / max((traded > 0).sum().sum(), 1))
    return (pnl, pnl - pct - ib, float((traded > 0).sum(axis=1).mean()),
            float(ib.mean() * 252), at_min)


def main() -> None:
    print("loading mid+small …")
    panel = download_ohlcv(sorted(set(sp1500_tickers() + ["SPY"])), "2011-01-01", None)
    pf = panel["adj_close"].dropna(how="all", axis=1)
    vol = panel["volume"].reindex_like(pf)
    bench = pf["SPY"].pct_change(fill_method=None)
    prices = pf.drop(columns=["SPY"], errors="ignore")
    vol = vol.drop(columns=["SPY"], errors="ignore")
    r = prices.pct_change(fill_method=None)
    betas = rolling_beta(r, bench, 252)
    sectors = sp1500_sectors().reindex(prices.columns)
    idio = r.rolling(20).std()
    import yfinance as yf
    vix = (yf.download("^VIX", start="2011-01-01", auto_adjust=True,
                       progress=False)["Close"].squeeze() / 100.0)
    tier = sp1500_constituents().set_index("ticker")["tier"].reindex(prices.columns)
    ms = [c for c in prices.columns if tier.get(c) in ("mid", "small")]
    prices, vol = prices[ms], vol[ms]

    w_full = build_weights(
        Variant("champion", horizons=(1, 3, 5, 10), news_filter=True, smooth=2,
                inv_vol=True, vix_scale=True),
        prices, vol, betas.reindex(columns=ms), sectors.reindex(ms),
        idio.reindex(columns=ms), vix)
    common = w_full.replace(0.0, np.nan).dropna(how="all").index
    w_full = w_full.reindex(common).fillna(0.0)

    for budget, tag in ((1_000_000.0, "$1M"), (100_000.0, "$100k")):
        cells, gross_by_n = {}, {}
        for n in NAMES:
            held = concentrate(w_full, n // 2, n // 2)
            for k in KS:
                g, net, ordd, ibyr, at_min = evaluate_periodic(held, prices, vol, budget, k)
                cells[(n, k)] = (summary_stats(net.fillna(0.0))["sharpe"], ibyr, ordd, at_min)
                if k == 1:
                    gross_by_n[n] = summary_stats(g.fillna(0.0))["sharpe"]

        print("\n" + "=" * 92)
        print(f"NET-OF-IB SHARPE — {tag} book.   rows = names held, cols = rebalance every k days")
        print("=" * 92)
        print(f"  {'names':>6}{'gross':>8}   " + "".join(f"{'k=' + str(k):>11}" for k in KS))
        print("  " + "-" * 86)
        for n in NAMES:
            row = "".join(f"{cells[(n, k)][0]:>+11.2f}" for k in KS)
            print(f"  {n:>6}{gross_by_n[n]:>+8.2f}   {row}")
        print("\n  IB cost %/yr:")
        print(f"  {'names':>6}          " + "".join(f"{'k=' + str(k):>11}" for k in KS))
        for n in NAMES:
            print(f"  {n:>6}          " + "".join(f"{cells[(n, k)][1]:>10.1%} " for k in KS))
        print("\n  share of orders paying the $1 MINIMUM rather than the per-share rate:")
        print(f"  {'names':>6}          " + "".join(f"{'k=' + str(k):>11}" for k in KS))
        for n in NAMES:
            print(f"  {n:>6}          " + "".join(f"{cells[(n, k)][3]:>10.0%} " for k in KS))

        best = max(cells.items(), key=lambda kv: kv[1][0])
        print(f"\n  BEST {tag}: {best[0][0]} names, rebalance every {best[0][1]}d "
              f"-> net {best[1][0]:+.2f} (IB {best[1][1]:.1%}/yr, {best[1][2]:.0f} orders/day)")


if __name__ == "__main__":
    main()
