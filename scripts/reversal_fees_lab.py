"""Fixed-commission stress — does the reversal book survive ~$5/order, and does concentrating
to a few names fix it?

The prior cost model charged only % costs (spread + sqrt-impact on notional). But a flat
per-order commission scales with the NUMBER OF ORDERS, not notional — and a ~200-name daily book
places ~200 orders/day. At $5/order on a $1M book that's ~$1000/day ≈ ~19%/yr, which would bury a
~6% gross return. Concentrating to N_long/N_short cuts order count (fewer fees) but also cuts breadth
(lower gross Sharpe, the sqrt(N) law). This measures the tradeoff, fee included, across book widths.

Run: python scripts/reversal_fees_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from backtest import LinearCostModel, summary_stats
from data import download_ohlcv
from data.universe import sp1500_constituents, sp1500_sectors, sp1500_tickers
from strategies.equity_mn.neutralize import rolling_beta
from reversal_lab import Variant, build_weights

FEE = 5.0  # $ per order


def concentrate(held: pd.DataFrame, n_long: int, n_short: int) -> pd.DataFrame:
    """Per day keep the n_long strongest longs + n_short strongest shorts, dollar-neutral (±0.5)."""
    out = pd.DataFrame(0.0, index=held.index, columns=held.columns)
    hv = held.values
    cols = held.columns
    for i in range(hv.shape[0]):
        row = hv[i]
        pos_idx = np.where(row > 0)[0]; neg_idx = np.where(row < 0)[0]
        if len(pos_idx):
            top = pos_idx[np.argsort(row[pos_idx])[::-1][:n_long]]
            s = row[top].sum()
            if s > 0: out.iloc[i, top] = 0.5 * row[top] / s
        if len(neg_idx):
            bot = neg_idx[np.argsort(row[neg_idx])[:n_short]]
            s = np.abs(row[bot]).sum()
            if s > 0: out.iloc[i, bot] = -0.5 * np.abs(row[bot]) / s
    return out


def evaluate(held_c, prices, volume, budget, spread_bps=1.5):
    """Returns (gross, net_pctonly, net_flat5, net_ib, orders/day, flat5%/yr, ib%/yr)."""
    px = prices.reindex_like(held_c)
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    gross = (held_c * rets).sum(axis=1)
    dw = held_c.diff().abs().fillna(held_c.abs())
    adv = (prices * volume).rolling(21).mean().reindex_like(held_c).ffill()
    adv = adv.fillna(adv.median().median())
    pct_cost = LinearCostModel(spread_bps, 10.0).charge(dw * budget, adv) / budget

    shares = (held_c * budget).div(px).round()
    shares = shares.where(held_c.abs() > 0, 0.0).fillna(0.0)
    dshares = (shares - shares.shift(1)).abs()
    traded = dshares.where(dshares > 0, 0.0)
    orders = (traded > 0).sum(axis=1)
    flat = orders * FEE / budget                                   # $5 flat/order (user's number)
    # IB US-stock fixed tier: $0.005/share, min $1/order, max 1% of trade value
    per = np.minimum(0.005 * traded, 0.01 * traded * px.fillna(0.0))
    per = per.where(traded > 0, 0.0).clip(lower=0.0)
    per = per.mask((traded > 0) & (per < 1.0), 1.0)               # $1 order minimum
    ib = per.sum(axis=1) / budget
    return (gross, gross - pct_cost, gross - pct_cost - flat, gross - pct_cost - ib,
            float(orders.mean()), float(flat.mean() * 252), float(ib.mean() * 252))


def main():
    print("loading mid+small …")
    panel = download_ohlcv(sorted(set(sp1500_tickers() + ["SPY"])), "2011-01-01", None)
    pf = panel["adj_close"].dropna(how="all", axis=1); vol = panel["volume"].reindex_like(pf)
    bench = pf["SPY"].pct_change(fill_method=None)
    prices = pf.drop(columns=["SPY"], errors="ignore"); vol = vol.drop(columns=["SPY"], errors="ignore")
    r = prices.pct_change(fill_method=None); betas = rolling_beta(r, bench, 252)
    sectors = sp1500_sectors().reindex(prices.columns); idio = r.rolling(20).std()
    import yfinance as yf
    vix = (yf.download("^VIX", start="2011-01-01", auto_adjust=True, progress=False)["Close"].squeeze() / 100.0)
    tier = sp1500_constituents().set_index("ticker")["tier"].reindex(prices.columns)
    ms = [c for c in prices.columns if tier.get(c) in ("mid", "small")]
    prices, vol = prices[ms], vol[ms]
    held = build_weights(Variant("r", horizons=(1, 3, 5, 10), news_filter=True, smooth=2, inv_vol=True,
                                 vix_scale=True), prices, vol, betas.reindex(columns=ms),
                         sectors.reindex(ms), idio.reindex(columns=ms), vix)
    common = held.replace(0.0, np.nan).dropna(how="all").index

    configs = [("broad ~100/100", 100, 100), ("50L/50S", 50, 50), ("25L/25S", 25, 25),
               ("10L/20S (asked)", 10, 20), ("10L/10S", 10, 10), ("5L/5S", 5, 5)]

    def sh(r):
        return summary_stats(r.reindex(common).fillna(0.0))["sharpe"]

    for budget in (1_000_000.0, 100_000.0):
        print("\n" + "=" * 100)
        print(f"FIXED-FEE STRESS  (${budget:,.0f} book, mid+small, net Sharpe under each cost model)")
        print("=" * 100)
        print(f"  {'config':16s} {'names':>6s} {'grossSh':>7s} {'%only':>6s} {'+$5flat':>8s} {'+IBreal':>8s} "
              f"{'ord/day':>7s} {'$5%/yr':>6s} {'IB%/yr':>6s}")
        print("  " + "-" * 96)
        for name, nl, ns in configs:
            hc = concentrate(held, nl, ns)
            g, npct, nflat, nib, ordday, flatyr, ibyr = evaluate(hc, prices, vol, budget, 1.5)
            print(f"  {name:16s} {nl+ns:>6d} {sh(g):>+7.2f} {sh(npct):>+6.2f} {sh(nflat):>+8.2f} "
                  f"{sh(nib):>+8.2f} {ordday:>7.0f} {flatyr:>5.1%} {ibyr:>5.1%}")
        print("  %only = spread+impact (old model); +$5flat = your stated fee; +IBreal = IB $0.005/sh, $1 min, 1% cap.")


if __name__ == "__main__":
    main()
