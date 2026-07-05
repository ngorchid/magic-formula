"""Tactical beta hedge — only short SPY when the market is in a downtrend.

Idea: keep market beta while SPY is healthy (above its N-day moving average), but switch on
the beta hedge when SPY drops below it — capture the bull-market beta, hedge away the crashes.
A time-series-momentum overlay on the beta exposure. Sweep the MA speed (20/50/200d) vs the
always- and never-hedged extremes; the fast filter reacts quicker but whipsaws and can hedge
into V-shaped bottoms (e.g. COVID 2020). Ex-ante (trend state uses yesterday's close); incl
SPY borrow + toggle costs. Clean PIT S&P 500.
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
)
from strategies.magic_formula import ENHANCED_ITEMS, EnhancedMagicConfig, enhanced_weights
from strategies.magic_formula.construct import _rebal_dates, pnl

BORROW_RATE = 0.004
TOGGLE_BPS = 3.0
MAS = [20, 50, 200]


def _beta_daily(port, spy, rebal, window=252):
    betas = {}
    for t in rebal:
        p = port.loc[:t].iloc[-window:]; s = spy.loc[:t].iloc[-window:]
        if p.notna().sum() >= 120 and s.var() > 0:
            betas[t] = float(p.cov(s) / s.var())
    return pd.Series(betas).reindex(port.index, method="ffill").shift(1)


def main(start: str = "2012-01-01", end: str | None = None, use_graham: bool = True) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg = EnhancedMagicConfig(use_graham=use_graham)
    tickers = sp500_pit_universe(start, end)
    full = sorted(set(tickers + ["SPY"]))
    print(f"[load] PIT S&P 500 {start}→{end} …")
    panel = download_ohlcv(full, start, end)
    adj_all = panel["adj_close"].dropna(how="all", axis=1)
    spy_px = adj_all["SPY"]
    spy = spy_px.pct_change(fill_method=None)
    adj = adj_all.drop(columns=["SPY"], errors="ignore")
    close = panel["close"].reindex_like(adj_all).drop(columns=["SPY"], errors="ignore")
    volume = panel["volume"].reindex_like(adj_all).drop(columns=["SPY"], errors="ignore")
    excluded = sp500_sectors().reindex(adj.columns).isin(cfg.exclude_sectors)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    base = base & sp500_pit_eligible(adj.index, list(adj.columns))
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()

    print("[run] long book + tactical hedge sweep …")
    weights, _ = enhanced_weights(f, mcap, adj, base, cfg)
    port, _ = pnl(weights, adj, volume, close)
    beta = _beta_daily(port, spy, _rebal_dates(adj.index, cfg.rebalance)).reindex(port.index)

    def hedged(active: pd.Series) -> pd.Series:
        """active: daily 0/1 fraction of beta to hedge (already ex-ante)."""
        hn = (active * beta).fillna(0.0)                       # hedge notional (× SPY)
        borrow = hn.abs() * (BORROW_RATE / 252)
        toggle = hn.diff().abs().fillna(hn.abs()) * (TOGGLE_BPS / 1e4)
        return port - hn * spy - borrow - toggle

    variants = {"never-hedged (long book)": port,
                "always-hedged": hedged(pd.Series(1.0, index=port.index))}
    for n in MAS:
        downtrend = (spy_px < spy_px.rolling(n).mean()).shift(1).fillna(False).astype(float)
        downtrend = downtrend.reindex(port.index).fillna(0.0)
        variants[f"tactical MA{n}"] = hedged(downtrend)

    common = port.replace(0.0, np.nan).dropna().index.intersection(beta.dropna().index)
    print("\n" + "=" * 80)
    print(f"TACTICAL BETA HEDGE  ({start}→{end}, PIT S&P 500, net, graham={use_graham})")
    print("=" * 80)
    print(f"  {'variant':26s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'corrSPY':>8s} {'%hedged':>8s}")
    hedged_frac = {"always-hedged": 1.0}
    for n in MAS:
        dt = (spy_px < spy_px.rolling(n).mean()).shift(1).fillna(False).reindex(common)
        hedged_frac[f"tactical MA{n}"] = float(dt.mean())
    for name, r in variants.items():
        s = summary_stats(r.reindex(common).fillna(0.0))
        c = r.reindex(common).fillna(0.0).corr(spy.reindex(common).fillna(0.0))
        hf = hedged_frac.get(name)
        hfs = f"{hf:>7.0%}" if hf is not None else "     — "
        print(f"  {name:26s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
              f"{s['max_drawdown']:>+8.2%} {c:>+8.2f} {hfs}")
    s = summary_stats(spy.reindex(common).fillna(0.0))
    print(f"  {'SPY':26s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
          f"{s['max_drawdown']:>+8.2%} {1.0:>+8.2f}")

    out = ROOT / "results" / "tactical_hedge"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(variants).to_csv(out / "tactical_hedge_net.csv")
    print(f"\n  wrote {out}/tactical_hedge_net.csv")


if __name__ == "__main__":
    main(use_graham="--no-graham" not in sys.argv)
