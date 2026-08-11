"""Tune the daily cvxpy optimizer for the reversal book — does it beat the heuristic on NET?

The heuristic book (signal-proportional, concentrate, 2d-smooth) hits a turnover cost wall.
The optimizer maximizes alpha - risk - TURNOVER-COST with exact neutrality (no-trade bands via
the L1 turnover term). Probe showed it slashes turnover but under-deploys at default risk_aversion
(tuned for monthly hold-21). Here we re-tune risk_aversion x hold_days for the 1-2 day horizon.

Scope discipline (the 1000-name daily QP is prohibitive): MID-CAP only (~400, the tradeable tier),
tight windows (slice data, don't solve 15yr). Judged net-of-cost vs the heuristic champion on the
SAME universe/window. Run: python scripts/reversal_opt_lab.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from backtest import LinearCostModel, summary_stats
from combination import cs_zscore, winsorize
from data import download_ohlcv
from data.universe import sp1500_constituents, sp1500_sectors, sp1500_tickers
from signals import short_term_reversal
from strategies.equity_mn.neutralize import neutralize, rolling_beta
from portfolio import run_optimized_backtest
from reversal_lab import Variant, build_weights


def _sh(r, idx):
    return summary_stats(r.reindex(idx).fillna(0.0))["sharpe"]


def heuristic_net(prices, volume, betas, sectors, idio_vol, vix, spread=1.0):
    """The current champion book's net-MOC returns on this universe."""
    v = Variant("h", horizons=(1, 3, 5, 10), news_filter=True, smooth=2, inv_vol=True, vix_scale=True)
    w = build_weights(v, prices, volume, betas, sectors, idio_vol, vix)
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    dw = w.diff().abs().fillna(w.abs())
    adv = (prices * volume).rolling(21).mean().reindex_like(w).ffill(); adv = adv.fillna(adv.median().median())
    return gross - LinearCostModel(spread, 10.0).charge(dw * 1e6, adv) / 1e6


def main():
    print("loading S&P1500, slicing to mid-cap …")
    panel = download_ohlcv(sorted(set(sp1500_tickers() + ["SPY"])), "2011-01-01", None)
    pf = panel["adj_close"].dropna(how="all", axis=1); vol = panel["volume"].reindex_like(pf)
    bench = pf["SPY"].pct_change(fill_method=None)
    prices = pf.drop(columns=["SPY"], errors="ignore"); vol = vol.drop(columns=["SPY"], errors="ignore")
    tier = sp1500_constituents().set_index("ticker")["tier"].reindex(prices.columns)
    mid = [c for c in prices.columns if tier.get(c) == "mid"]
    prices, vol = prices[mid], vol[mid]
    rets = prices.pct_change(fill_method=None)
    betas = rolling_beta(rets, bench, 252); sectors = sp1500_sectors().reindex(prices.columns)
    idio_vol = rets.rolling(20).std()
    import yfinance as yf
    vix = (yf.download("^VIX", start="2011-01-01", auto_adjust=True, progress=False)["Close"].squeeze() / 100.0)
    print(f"  mid-cap: {prices.shape[1]} names")

    # alpha for the optimizer = neutralized multi-horizon reversal + news filter (it does risk/neutrality itself)
    parts = [cs_zscore(winsorize(short_term_reversal(prices, h).reindex_like(prices), 0.01, 0.99)) for h in (1, 3, 5, 10)]
    alpha = neutralize(sum(parts) / len(parts), betas=betas, sectors=sectors)
    volz = (vol - vol.rolling(63).mean()) / vol.rolling(63).std()
    alpha = alpha.where(~((volz > 3.0).rolling(5).max().fillna(0.0) > 0), 0.0)

    def window(lo, hi, warm):
        """Slice to [warm, hi]; solve all, evaluate on [lo, hi]."""
        return slice(warm, hi), slice(lo, hi)

    for label, lo, hi, warm in [("2017-2019", "2017-01-01", "2019-12-31", "2016-01-01"),
                                ("2022-2024", "2022-01-01", "2024-12-31", "2021-01-01")]:
        dsl, esl = window(lo, hi, warm)
        p, v, a, b = prices.loc[dsl], vol.loc[dsl], alpha.loc[dsl], betas.loc[dsl]
        eidx = p.loc[esl].index
        hnet = heuristic_net(p, v, b, sectors, idio_vol.loc[dsl], vix)
        print(f"\n=== {label}  (mid-cap, {len(eidx)} eval days) ===")
        print(f"  {'construction':22s} {'netSh':>6s} {'turn/d':>7s} {'grossLev':>8s}")
        print(f"  {'HEURISTIC champion':22s} {_sh(hnet, eidx):>+6.2f} {'—':>7s} {'—':>8s}")
        for ra in (1.0, 2.0, 4.0):
            for hd in (1, 2):
                t = time.time()
                res = run_optimized_backtest(a, p, b, sectors, volume=v, rebalance="D",
                        cost_model=LinearCostModel(1.0, 10.0), hold_days=hd,
                        risk_aversion=ra, turnover_cost=0.0010, max_position=0.03)
                net = res.net_returns
                gl = res.weights.abs().sum(axis=1).mean()
                print(f"  opt ra={ra:g} hold={hd}d      {_sh(net, eidx):>+6.2f} "
                      f"{res.turnover.mean():>6.1%} {gl:>7.2f}x   [{time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
