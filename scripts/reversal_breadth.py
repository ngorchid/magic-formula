"""Reversal breadth test — does the edge survive (and decay less) down-cap?

Large-cap S&P500 reversal decayed to net-MOC Sharpe ~0.34 in 2021-26 (crowding). Thesis:
mid/small caps are less crowded, so the reversal premium should be stronger and less decayed
there — the catch is worse auction liquidity, which the √impact-on-real-ADV cost term captures
automatically (smaller ADV -> higher participation -> higher impact). We run the SAME champion
config (multi-horizon + inverse-vol + VIX-conditioning) on three cap tiers and read net Sharpe
at three flat auction-spread assumptions (large caps live near 0.5-1bp, small caps 3bp+), with
special attention to the 2021-26 sub-period.

Run: python scripts/reversal_breadth.py   (slow first run: ~1500-name pull)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import LinearCostModel, summary_stats
from data import download_ohlcv
from data.universe import sp1500_constituents, sp1500_sectors, sp1500_tickers
from strategies.equity_mn.neutralize import rolling_beta
from scripts.reversal_lab import Variant, build_weights

CHAMP = Variant("champion", horizons=(1, 3, 5, 10), news_filter=True, smooth=2,
                inv_vol=True, vix_scale=True)


def _sh(r, common):
    return summary_stats(r.reindex(common).fillna(0.0))["sharpe"]


def run_universe(label, cols, prices, volume, betas, sectors, idio_vol, vix):
    p, v = prices[cols], volume[cols]
    if p.shape[1] < 30:
        print(f"  {label:14s}  (only {p.shape[1]} names — skipped)"); return
    w = build_weights(CHAMP, p, v, betas.reindex(columns=cols),
                      sectors.reindex(cols), idio_vol.reindex(columns=cols), vix)
    rets = p.pct_change(fill_method=None).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    dw = w.diff().abs().fillna(w.abs())
    turnover = float(dw.sum(axis=1).mean())
    adv = (p * v).rolling(21).mean().reindex_like(w).ffill()
    adv = adv.fillna(adv.median().median())
    common = gross.replace(0.0, np.nan).dropna().index

    nets = {}
    for spread in (0.5, 1.5, 3.0):
        cm = LinearCostModel(half_spread_bps=spread, impact_coef_bps=10.0)
        nets[spread] = gross - cm.charge(dw * 1_000_000.0, adv) / 1_000_000.0
    gs = _sh(gross, common)
    n05, n15, n30 = (_sh(nets[s], common) for s in (0.5, 1.5, 3.0))
    recent = nets[1.5].loc["2021-01-01":]
    rec_sh = summary_stats(recent.replace(0.0, np.nan).dropna().fillna(0.0))["sharpe"]
    ann = summary_stats(nets[1.5].reindex(common).fillna(0.0))["ann_return"]
    print(f"  {label:14s} {p.shape[1]:>5d} {gs:>+7.2f} {n05:>+6.2f} {n15:>+6.2f} {n30:>+6.2f} "
          f"{ann:>+7.1%} {turnover:>7.1%} {rec_sh:>+8.2f}")


def main(start="2011-01-01", end=None):
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sp1500_tickers()
    print(f"loading {len(tickers)} S&P 1500 names + SPY, {start}->{end}  (first run is slow) …")
    panel = download_ohlcv(sorted(set(tickers + ["SPY"])), start, end)
    prices_full = panel["adj_close"].dropna(how="all", axis=1)
    volume = panel["volume"].reindex_like(prices_full)
    bench = prices_full["SPY"].pct_change(fill_method=None)
    prices = prices_full.drop(columns=["SPY"], errors="ignore")
    volume = volume.drop(columns=["SPY"], errors="ignore")
    rets = prices.pct_change(fill_method=None)
    betas = rolling_beta(rets, bench, window=252)
    sectors = sp1500_sectors().reindex(prices.columns)
    idio_vol = rets.rolling(20).std()
    vix = (yf.download("^VIX", start=start, end=end, auto_adjust=True,
                       progress=False)["Close"].squeeze() / 100.0)
    print(f"  {prices.shape[1]} names with data")

    tier = sp1500_constituents().set_index("ticker")["tier"].reindex(prices.columns)
    large = [c for c in prices.columns if tier.get(c) == "large"]
    midsm = [c for c in prices.columns if tier.get(c) in ("mid", "small")]
    allc = list(prices.columns)

    print("\n" + "=" * 104)
    print(f"REVERSAL BREADTH  ({start}->{end}, champion cfg, daily, net of auction cost)")
    print("=" * 104)
    print(f"  {'universe':14s} {'nNames':>5s} {'grossSh':>7s} {'0.5bp':>6s} {'1.5bp':>6s} "
          f"{'3.0bp':>6s} {'ann1.5':>7s} {'turn/d':>7s} {'2021-26':>8s}")
    print("  " + "-" * 100)
    run_universe("large (SP500)", large, prices, volume, betas, sectors, idio_vol, vix)
    run_universe("mid+small", midsm, prices, volume, betas, sectors, idio_vol, vix)
    run_universe("all SP1500", allc, prices, volume, betas, sectors, idio_vol, vix)
    print("\n  Read the spread column matching the tier: large ~0.5-1bp, mid ~1.5bp, small ~3bp+.")
    print("  '2021-26' = recent-period net Sharpe @1.5bp — the decay check (large-cap was ~0.34).")


if __name__ == "__main__":
    main()
