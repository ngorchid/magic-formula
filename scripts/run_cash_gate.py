"""Risk-off-to-CASH gates (no shorting) — trend gate and volatility gate on the long book.

Instead of hedging with a SPY short, simply reduce equity exposure and hold cash when the
market is risky:
  * trend gate  — go to cash when SPY closes below its 200-day MA (binary),
  * vol gate    — vol-target: scale exposure by min(1, 15% / SPY 20d realised vol), so a vol
                  spike smoothly de-risks (Moreira-Muir volatility-managed portfolios),
  * both        — multiply the two.
Cash assumed to earn 0 (conservative — real T-bill yield, esp. 2023-24, would help the gated
versions). Ex-ante signals (yesterday's close); small cost to move the book to/from cash.
Clean PIT S&P 500.
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
    sp1500_sectors,
    sp1500_tickers,
)
from strategies.magic_formula import ENHANCED_ITEMS, EnhancedMagicConfig, enhanced_weights
from strategies.magic_formula.construct import pnl

TRADE_BPS = 5.0       # cost to shift the book to/from cash
VOL_TARGET = 0.15     # annualised vol target for the vol gate


def main(start: str = "2012-01-01", end: str | None = None, use_graham: bool = True,
         universe: str = "sp500_pit", weighting: str = "equal") -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg = EnhancedMagicConfig(use_graham=use_graham, weighting=weighting)
    if universe == "sp1500":
        tickers, sector_src, label = sp1500_tickers(), sp1500_sectors, "S&P 1500 (current, biased)"
    else:
        tickers, sector_src, label = sp500_pit_universe(start, end), sp500_sectors, "PIT S&P 500"
    full = sorted(set(tickers + ["SPY"]))
    print(f"[load] {label} {start}→{end} …")
    panel = download_ohlcv(full, start, end)
    adj_all = panel["adj_close"].dropna(how="all", axis=1)
    spy_px = adj_all["SPY"]
    spy = spy_px.pct_change(fill_method=None)
    adj = adj_all.drop(columns=["SPY"], errors="ignore")
    close = panel["close"].reindex_like(adj_all).drop(columns=["SPY"], errors="ignore")
    volume = panel["volume"].reindex_like(adj_all).drop(columns=["SPY"], errors="ignore")
    excluded = sector_src().reindex(adj.columns).isin(cfg.exclude_sectors)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    if universe == "sp500_pit":
        base = base & sp500_pit_eligible(adj.index, list(adj.columns))
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()

    print("[run] long book + cash gates …")
    weights, _ = enhanced_weights(f, mcap, adj, base, cfg)
    port, _ = pnl(weights, adj, volume, close)

    trend_on = (spy_px >= spy_px.rolling(200).mean()).shift(1).reindex(port.index).fillna(True).astype(float)
    rvol = (spy.rolling(20).std() * np.sqrt(252))
    vol_scale = (VOL_TARGET / rvol).clip(upper=1.0).shift(1).reindex(port.index).fillna(1.0)

    def gated(exposure: pd.Series) -> pd.Series:
        e = exposure.clip(0.0, 1.0)
        cost = e.diff().abs().fillna(0.0) * (TRADE_BPS / 1e4)
        return e * port - cost  # (1-e) in cash @ 0%

    variants = {
        "long book (no gate)": port,
        "trend gate (MA200->cash)": gated(trend_on),
        "vol gate (vol-target 15%)": gated(vol_scale),
        "trend + vol gate": gated(trend_on * vol_scale),
    }
    avg_exposure = {"trend gate (MA200->cash)": trend_on.mean(),
                    "vol gate (vol-target 15%)": vol_scale.mean(),
                    "trend + vol gate": float((trend_on * vol_scale).mean())}

    common = port.replace(0.0, np.nan).dropna().index
    print("\n" + "=" * 82)
    print(f"RISK-OFF-TO-CASH GATES  ({start}→{end}, {label}, net, graham={use_graham}, weighting={cfg.weighting})")
    print("=" * 82)
    print(f"  {'variant':28s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'corrSPY':>8s} {'avg_expo':>8s}")
    for name, r in variants.items():
        s = summary_stats(r.reindex(common).fillna(0.0))
        c = r.reindex(common).fillna(0.0).corr(spy.reindex(common).fillna(0.0))
        ae = avg_exposure.get(name)
        aes = f"{ae:>7.0%}" if ae is not None else "    100%"
        print(f"  {name:28s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
              f"{s['max_drawdown']:>+8.2%} {c:>+8.2f} {aes}")
    s = summary_stats(spy.reindex(common).fillna(0.0))
    print(f"  {'SPY':28s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
          f"{s['max_drawdown']:>+8.2%} {1.0:>+8.2f}")

    out = ROOT / "results" / "cash_gate"
    out.mkdir(parents=True, exist_ok=True)
    tag = "sp1500" if universe == "sp1500" else "sp500pit"
    pd.DataFrame(variants).to_csv(out / f"cash_gate_{tag}.csv")
    print(f"\n  wrote {out}/cash_gate_{tag}.csv")


if __name__ == "__main__":
    uni = "sp1500" if "sp1500" in sys.argv else "sp500_pit"
    wt = "inverse_vol" if "--inverse-vol" in sys.argv else "equal"
    main(use_graham="--no-graham" not in sys.argv, universe=uni, weighting=wt)
