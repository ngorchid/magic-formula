"""Validate the fundamentals pipeline and measure raw signal quality.

Run after wiring a fundamentals source. Reports, for the current S&P 500:
  1. coverage   — fraction of the universe with a non-NaN value over time
  2. sanity     — latest values for a hand-checkable name (AAPL)
  3. IC         — monthly cross-sectional rank-IC of each signal vs forward returns
                  (the direct "does this predict returns?" read)
  4. provenance — SimFin vs yfinance agreement on overlapping cells (the
                  "is paying for better data worth it?" read), on a small sample

This is a diagnostic, not a backtest — no costs, no portfolio construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import download_ohlcv, load_fundamentals, sp500_tickers
from signals import accruals, earnings_yield, gross_profitability, return_on_equity


def monthly_rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Cross-sectional Spearman IC on each rebalance date."""
    s = signal.reindex(dates).rank(axis=1)
    r = fwd.reindex(dates).rank(axis=1)
    return s.corrwith(r, axis=1)


def main(start: str = "2015-01-01", end: str | None = None) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sp500_tickers()
    print(f"Universe: {len(tickers)} current S&P 500 names | {start} → {end}\n")

    print("[1/4] Loading prices (yfinance, cached)…")
    panel = download_ohlcv(tickers, start, end)
    adj = panel["adj_close"].dropna(how="all", axis=1)
    close = panel["close"].reindex_like(adj)
    cal = adj.index

    print("[2/4] Loading fundamentals (SimFin, PIT)…")
    f = load_fundamentals(tickers, start, end, sources=("simfin",), calendar=cal)

    sigs = {
        "gross_profitability": gross_profitability(f),
        "return_on_equity": return_on_equity(f),
        "accruals": accruals(f),
        "earnings_yield": earnings_yield(f, close),
    }

    # Coverage: mean fraction of the (price-available) universe with a signal value.
    print("\n--- Coverage (mean % of universe with a value, last 5y) ---")
    recent = cal[cal >= (cal.max() - pd.Timedelta(days=365 * 5))]
    have_px = adj.reindex(recent).notna()
    for name, s in sigs.items():
        cov = (s.reindex(recent).notna() & have_px).sum(axis=1) / have_px.sum(axis=1).clip(lower=1)
        print(f"  {name:22} {cov.mean():6.1%}")

    # Sanity check on a known name.
    print("\n--- Sanity: AAPL latest fundamentals & signals ---")
    last = cal.max()
    for item in ("revenue", "gross_profit", "net_income", "total_assets", "total_equity"):
        v = f[item].get("AAPL")
        if v is not None:
            print(f"  {item:14} {v.loc[:last].dropna().iloc[-1]:,.0f}" if v.dropna().size else f"  {item:14} (none)")
    for name, s in sigs.items():
        col = s.get("AAPL")
        val = col.loc[:last].dropna().iloc[-1] if col is not None and col.dropna().size else float("nan")
        print(f"  signal {name:20} {val: .4f}")

    # IC vs forward 21-day returns, on month-end dates (limits overlap).
    print("\n--- Monthly rank-IC vs forward 21d return ---")
    fwd = adj.pct_change(21, fill_method=None).shift(-21)
    me = adj.resample("ME").last().index.intersection(cal)
    print(f"  {'signal':22} {'mean IC':>9} {'IR':>7} {'t-stat':>8}  (n={len(me)} months)")
    for name, s in sigs.items():
        ic = monthly_rank_ic(s, fwd, me).dropna()
        if ic.empty:
            print(f"  {name:22} {'n/a':>9}")
            continue
        ir = ic.mean() / ic.std() if ic.std() else float("nan")
        t = ir * np.sqrt(len(ic))
        print(f"  {name:22} {ic.mean():9.4f} {ir:7.2f} {t:8.2f}")

    # Provenance / reconciliation on a small sample (yfinance is slow + shallow).
    print("\n[3/4] SimFin vs yfinance reconciliation (15-name sample)…")
    sample = [t for t in ("AAPL", "MSFT", "NVDA", "JPM", "XOM", "PG", "KO", "JNJ",
                          "WMT", "HD", "CVX", "MRK", "PEP", "ABBV", "CSCO") if t in tickers]
    both, prov = load_fundamentals(
        sample, start, end, items=["gross_profit", "total_assets", "net_income"],
        sources=("simfin", "yfinance"), calendar=cal, return_provenance=True,
    )
    sf_only = load_fundamentals(sample, start, end, items=["gross_profit"],
                                sources=("simfin",), calendar=cal)
    yf_only = load_fundamentals(sample, start, end, items=["gross_profit"],
                                sources=("yfinance",), calendar=cal)
    overlap = sf_only["gross_profit"].notna() & yf_only["gross_profit"].notna()
    print(f"  cells with BOTH sources (gross_profit): {int(overlap.sum().sum()):,}")
    if overlap.sum().sum() > 0:
        rel_diff = ((sf_only["gross_profit"] - yf_only["gross_profit"]).abs()
                    / sf_only["gross_profit"].abs().where(overlap)).where(overlap)
        print(f"  median |SimFin−yfinance| / SimFin: {rel_diff.stack().median():.2%}")
    yf_cov = yf_only["gross_profit"].reindex(recent).notna().mean().mean()
    sf_cov = sf_only["gross_profit"].reindex(recent).notna().mean().mean()
    print(f"  coverage last 5y — SimFin: {sf_cov:.1%}   yfinance: {yf_cov:.1%}")

    print("\n[4/4] Done.")


if __name__ == "__main__":
    main()
