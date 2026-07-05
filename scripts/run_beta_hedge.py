"""Beta-hedge the enhanced Magic Formula — is there market-neutral alpha, or just beta?

The long-only book is ~beta 0.9 to SPY, so most of its ~20%/yr is market return in a bull
decade. Short SPY sized to the portfolio's trailing beta (re-estimated and resized MONTHLY)
to neutralise that, and see what's left:
  * if the hedged stream keeps a positive return at low SPY correlation -> genuine market-
    neutral alpha, i.e. the diversifying sleeve the project has wanted;
  * if it collapses to ~zero -> the 'edge' was mostly market beta.

Beta = trailing-252d OLS of the book's own returns on SPY, computed at each month-end and
applied to the following month (shifted, no lookahead). Includes small hedge costs (SPY
rehedge turnover + short borrow). Clean PIT S&P 500.
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

BORROW_RATE = 0.004    # ~0.4%/yr to short SPY
REHEDGE_BPS = 3.0      # cost of resizing the SPY hedge at each rehedge


def _beta_daily(port: pd.Series, spy: pd.Series, rebal: pd.DatetimeIndex, window: int = 252) -> pd.Series:
    """Monthly-updated trailing beta, ffilled to daily and shifted 1d (ex-ante)."""
    betas = {}
    for t in rebal:
        p = port.loc[:t].iloc[-window:]
        s = spy.loc[:t].iloc[-window:]
        if p.notna().sum() >= 120 and s.var() > 0:
            betas[t] = float(p.cov(s) / s.var())
    b = pd.Series(betas).reindex(port.index, method="ffill")
    return b.shift(1)


def main(start: str = "2012-01-01", end: str | None = None, use_graham: bool = True) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg = EnhancedMagicConfig(use_graham=use_graham)
    tickers = sp500_pit_universe(start, end)
    full = sorted(set(tickers + ["SPY"]))
    print(f"[load] PIT S&P 500 {start}→{end} …")
    panel = download_ohlcv(full, start, end)
    adj = panel["adj_close"].dropna(how="all", axis=1)
    close = panel["close"].reindex_like(adj)
    volume = panel["volume"].reindex_like(adj)
    spy = adj["SPY"].pct_change(fill_method=None)
    adj = adj.drop(columns=["SPY"], errors="ignore")
    close = close.drop(columns=["SPY"], errors="ignore")
    volume = volume.drop(columns=["SPY"], errors="ignore")
    excluded = sp500_sectors().reindex(adj.columns).isin(cfg.exclude_sectors)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    base = base & sp500_pit_eligible(adj.index, list(adj.columns))
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()

    print("[run] long book + monthly beta hedge …")
    weights, _ = enhanced_weights(f, mcap, adj, base, cfg)
    port, _ = pnl(weights, adj, volume, close)   # unhedged long-book net returns

    rebal = _rebal_dates(adj.index, cfg.rebalance)
    beta = _beta_daily(port, spy, rebal).reindex(port.index)
    # hedge P&L overlay: short beta*SPY, minus borrow and rehedge costs
    borrow = beta.abs() * (BORROW_RATE / 252)
    rehedge = pd.Series(0.0, index=port.index)
    b_on_rebal = beta.reindex(rebal).dropna()
    rehedge.loc[b_on_rebal.index] = b_on_rebal.diff().abs().fillna(b_on_rebal.abs()) * (REHEDGE_BPS / 1e4)
    hedged = port - beta * spy - borrow - rehedge

    common = port.replace(0.0, np.nan).dropna().index
    common = common.intersection(beta.dropna().index)

    def line(name, r, extra=""):
        s = summary_stats(r.reindex(common).fillna(0.0))
        c = r.reindex(common).fillna(0.0).corr(spy.reindex(common).fillna(0.0))
        print(f"  {name:22s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
              f"{s['max_drawdown']:>+8.2%} {c:>+8.2f}{extra}")

    # realized beta of the hedged book (should be ~0)
    hb = hedged.reindex(common).fillna(0.0); sp = spy.reindex(common).fillna(0.0)
    resid_beta = float(hb.cov(sp) / sp.var())

    print("\n" + "=" * 78)
    print(f"BETA-HEDGED ENHANCED MAGIC FORMULA  ({start}→{end}, PIT S&P 500, net, graham={use_graham})")
    print("=" * 78)
    print(f"  {'variant':22s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'corrSPY':>8s}")
    line("long book (unhedged)", port, f"   avg beta {beta.reindex(common).mean():.2f}")
    line("beta-hedged", hedged, f"   resid beta {resid_beta:+.2f}")
    line("SPY", spy)

    out = ROOT / "results" / "beta_hedge"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"long_book": port, "beta_hedged": hedged, "beta": beta}).to_csv(out / "beta_hedge.csv")
    print(f"\n  wrote {out}/beta_hedge.csv")


if __name__ == "__main__":
    main(use_graham="--no-graham" not in sys.argv)
