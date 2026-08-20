"""At what portfolio size does the reversal book actually work?

There is a sweet spot, because the two cost components run in OPPOSITE directions with size:

  TOO SMALL   the $1 IB order minimum dominates. A $2,000 position trading ~100 shares costs
              $0.005 x 100 = $0.50 in per-share terms, so you pay the $1 MINIMUM -- double the
              true rate. Cost per round trip is ~0.10% of the position against an edge of
              ~0.087%, and the strategy loses.
  TOO LARGE   market impact dominates. 50 names in mid/small caps means a $50M book carries $1M
              positions, which is a meaningful fraction of daily volume, and impact scales with
              size while the edge does not.

Between them the per-share rate applies and cost per round trip falls to ~2.5bp on a $40 stock
(0.005/40, twice) -- roughly a quarter of what the minimum charges.

Config is the one CHOSEN ON THE 2012-2019 HALF in reversal_holdout_lab (inv_vol=True, Q4+ vol
filter, 50 names, weekly), so the OOS column is the honest number. Full-sample is shown beside it
because the strategy was negative pre-2020 and that regime split matters more than it looks.

⚠ CAPACITY IS NOT JUST COST. The impact model here is a sqrt-law on notional/ADV, which is a
reasonable first-order estimate and no substitute for actually trading it. Treat the large end as
indicative of WHERE the ceiling is, not as a promise about what fills you would get.

Run: python3 scripts/reversal_capacity_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from backtest import LinearCostModel, summary_stats       # noqa: E402
from data import download_ohlcv                            # noqa: E402
from data.universe import (sp1500_constituents, sp1500_sectors,  # noqa: E402
                           sp1500_tickers)
from strategies.equity_mn.neutralize import rolling_beta    # noqa: E402
from reversal_lab import Variant, build_weights             # noqa: E402
from reversal_fees_lab import concentrate                   # noqa: E402

BUDGETS = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000,
           10_000_000, 25_000_000, 50_000_000]
NAMES, K, SPLIT = 50, 5, "2020-01-01"


def split_costs(w, prices, volume, budget, k, spread_bps=1.5):
    """Return the P&L and each cost component separately, shares carried between rebalances."""
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
    per = per.mask((traded > 0) & (per < 1.0), 1.0)
    ib = per.sum(axis=1) / budget
    at_min = float(((traded > 0) & (0.005 * traded < 1.0)).sum().sum()
                   / max((traded > 0).sum().sum(), 1))
    # cost of ONE average round trip, as a fraction of the position traded
    rt = float((per.values[traded.values > 0].sum() / notional.values[traded.values > 0].sum()))
    return pnl, pct, ib, at_min, rt


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
    prices, vol, idio = prices[ms], vol[ms], idio[ms]

    w_full = build_weights(
        Variant("chosen", horizons=(1, 3, 5, 10), news_filter=True, smooth=2,
                inv_vol=True, vix_scale=True),
        prices, vol, betas.reindex(columns=ms), sectors.reindex(ms), idio, vix)
    common = w_full.replace(0.0, np.nan).dropna(how="all").index
    w_full = w_full.reindex(common).fillna(0.0)
    vranks = idio.reindex(common).rank(axis=1, pct=True)
    held = concentrate(w_full.where(vranks > 0.6, 0.0), NAMES // 2, NAMES // 2)   # Q4+

    print("\n" + "=" * 100)
    print(f"CAPACITY — IS-chosen config (inv_vol=True, Q4+, {NAMES} names, weekly)")
    print("=" * 100)
    print(f"  {'budget':>12}{'pos size':>11}{'$1min':>8}{'rt cost':>9}"
          f"{'IB%/yr':>8}{'impact%/yr':>11}{'net FULL':>10}{'net OOS':>9}")
    print("  " + "-" * 96)
    rows = []
    for b in BUDGETS:
        pnl, pct, ib, at_min, rt = split_costs(held, prices, vol, float(b), K)
        net = pnl - pct - ib
        full = summary_stats(net.fillna(0.0))["sharpe"]
        oos = summary_stats(net.loc[SPLIT:].fillna(0.0))["sharpe"]
        rows.append({"budget": b, "at_min": at_min, "rt_bp": rt * 1e4,
                     "ib_yr": ib.mean() * 252, "pct_yr": pct.mean() * 252,
                     "net_full": full, "net_oos": oos})
        print(f"  {b:>12,}{b / NAMES:>11,.0f}{at_min:>8.0%}{rt * 1e4:>8.1f}bp"
              f"{ib.mean() * 252:>8.1%}{pct.mean() * 252:>11.1%}{full:>+10.2f}{oos:>+9.2f}")

    df = pd.DataFrame(rows)
    b_full = df.loc[df["net_full"].idxmax()]
    b_oos = df.loc[df["net_oos"].idxmax()]
    print(f"\n  peak FULL-sample: ${b_full['budget']:,.0f}  -> {b_full['net_full']:+.2f}")
    print(f"  peak OOS        : ${b_oos['budget']:,.0f}  -> {b_oos['net_oos']:+.2f}")
    ok = df[df["net_oos"] >= 0.5]
    if len(ok):
        print(f"  clears net 0.5 OOS from ${ok['budget'].min():,.0f} upward")
    else:
        print(f"  NEVER clears net 0.5 OOS at any size tested "
              f"(best {df['net_oos'].max():+.2f})")
    print("\n  '$1min' = share of orders paying the minimum rather than the per-share rate.")
    print("  'rt cost' = commission on one average round trip, in bp of the traded notional.")
    print("  Edge for comparison: ~8.7bp per position per weekly round trip.")
    df.to_csv(ROOT / "results" / "reversal_capacity.csv", index=False)


if __name__ == "__main__":
    main()
