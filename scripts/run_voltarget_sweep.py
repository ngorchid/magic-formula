"""Vol-target sweep — map the return-vs-drawdown trade so you can pick the target (or none).

Vol-targeting trades return for a smoother ride; the vol *target* sets how much. Higher
target = de-risk less = keep more return but a deeper tail. This sweeps the target on the
enhanced (inverse-vol) long book so the frontier is explicit. `none` = the raw book.
Also an 'extreme-only' variant that de-risks solely when SPY vol is very high (>25%) — keeps
almost all the return, clips only the worst crashes. Clean PIT S&P 500 and biased S&P 1500.
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
    download_ohlcv, load_fundamentals, sp500_pit_eligible, sp500_pit_universe,
    sp500_sectors, sp1500_sectors, sp1500_tickers,
)
from strategies.magic_formula import ENHANCED_ITEMS, EnhancedMagicConfig, enhanced_weights
from strategies.magic_formula.construct import pnl

TRADE_BPS = 5.0
TARGETS = [0.12, 0.15, 0.18, 0.20, 0.25]


def main(universe: str = "sp1500", start: str = "2012-01-01", end: str | None = None) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg = EnhancedMagicConfig(weighting="inverse_vol")
    if universe == "sp1500":
        tickers, sector_src, label = sp1500_tickers(), sp1500_sectors, "S&P 1500 (biased)"
    else:
        tickers, sector_src, label = sp500_pit_universe(start, end), sp500_sectors, "PIT S&P 500"
    full = sorted(set(tickers + ["SPY"]))
    print(f"[load] {label} …")
    panel = download_ohlcv(full, start, end)
    adj_all = panel["adj_close"].dropna(how="all", axis=1)
    spy_px = adj_all["SPY"]; spy = spy_px.pct_change(fill_method=None)
    adj = adj_all.drop(columns=["SPY"], errors="ignore")
    close = panel["close"].reindex_like(adj_all).drop(columns=["SPY"], errors="ignore")
    volume = panel["volume"].reindex_like(adj_all).drop(columns=["SPY"], errors="ignore")
    excluded = sector_src().reindex(adj.columns).isin(cfg.exclude_sectors)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    if universe != "sp1500":
        base = base & sp500_pit_eligible(adj.index, list(adj.columns))
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()

    weights, _ = enhanced_weights(f, mcap, adj, base, cfg)
    port, _ = pnl(weights, adj, volume, close)
    rvol = (spy.rolling(20).std() * np.sqrt(252))

    def gated(exposure):
        e = exposure.reindex(port.index).fillna(1.0).clip(0, 1)
        cost = e.diff().abs().fillna(0.0) * (TRADE_BPS / 1e4)
        return e * port - cost, float(e.mean())

    rows = [("none (raw book)", port, 1.0)]
    for tgt in TARGETS:
        r, ae = gated((tgt / rvol).clip(upper=1.0).shift(1))
        rows.append((f"vol-target {tgt:.0%}", r, ae))
    # extreme-only: full exposure unless SPY 20d vol > 25%, then vol-target to 25%
    extreme = (0.25 / rvol).clip(upper=1.0).where(rvol > 0.25, 1.0).shift(1)
    r, ae = gated(extreme)
    rows.append(("extreme-only (>25% vol)", r, ae))

    common = port.replace(0.0, np.nan).dropna().index
    print("\n" + "=" * 78)
    print(f"VOL-TARGET SWEEP — enhanced (inverse-vol) long book, {label}")
    print("=" * 78)
    print(f"  {'variant':24s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'avg_expo':>8s}")
    for name, r, ae in rows:
        s = summary_stats(r.reindex(common).fillna(0.0))
        print(f"  {name:24s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
              f"{s['max_drawdown']:>+8.2%} {ae:>7.0%}")
    s = summary_stats(spy.reindex(common).fillna(0.0))
    print(f"  {'SPY':24s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
          f"{s['max_drawdown']:>+8.2%} {'—':>7s}")

    out = ROOT / "results" / "voltarget_sweep"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({n: r for n, r, _ in rows}).to_csv(out / f"voltarget_{universe}.csv")
    print(f"\n  wrote {out}/voltarget_{universe}.csv")


if __name__ == "__main__":
    uni = "sp500_pit" if "sp500_pit" in sys.argv else "sp1500"
    main(universe=uni)
