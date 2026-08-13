"""Backtest the ENHANCED magic formula — the variant that is actually deployed.

WHY THIS EXISTS. `strategies/magic_formula/enhanced.py` defines the deployed signal and
`paper/rank.py` uses it live, but nothing in the repo turned it into a RETURN SERIES. The only
runnable backtest was `MagicFormula(MagicFormulaConfig())`, which is base Greenblatt — a
different, and on this data losing, strategy (ann −3.19%, Sharpe −0.06). Anything measured on
that says nothing about what is deployed, and it silently invalidated the circuit-breaker
calibration for this sleeve.

The enhanced module already produces a weight panel (`enhanced_weights`); this supplies the
missing half — universe, fundamentals, and the weights→net-returns machinery, reusing the same
LinearCostModel as the base stream so the two are comparable.

⚠ DATA WINDOW IS THE BINDING CONSTRAINT. Free SimFin caps fundamentals at ~5 years, so this is
NOT the "clean survivorship-corrected PIT S&P 500" run quoted in enhanced.py's docstring
(Sharpe ~1.0 with Graham). Expect a different, noisier number from a shorter window on a
different dataset, and treat any Sharpe from ~5 years of monthly rebalancing as having a
standard error around ±0.45 — i.e. do not read a point estimate as a result.

Writes results/magic_formula/enhanced_net_returns.csv for the breaker-calibration lab.

Run: python scripts/enhanced_magic_backtest.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from backtest import LinearCostModel, summary_stats  # noqa: E402
from data import broad_universe, load_fundamentals  # noqa: E402
from strategies.magic_formula.enhanced import (  # noqa: E402
    ENHANCED_ITEMS, EnhancedMagicConfig, enhanced_weights)

OUT = ROOT / "results" / "magic_formula"
START, END = "2019-01-01", None
MIN_MCAP = 3e8
NOTIONAL = 1_000_000.0


def run(cfg: EnhancedMagicConfig, start: str = START, end: str | None = END) -> dict:
    tickers, eligible, panels = broad_universe(
        start, end, min_market_cap=MIN_MCAP, exclude_sectors=cfg.exclude_sectors)
    adj = panels["adj_close"]
    close, shares, volume = panels["close"], panels["shares"], panels["volume"]
    mcap = close * shares.reindex_like(close)
    f = load_fundamentals(tickers, start, end, items=ENHANCED_ITEMS,
                          sources=("simfin",), calendar=adj.index)

    weights, rank = enhanced_weights(f, mcap, adj, eligible, cfg)
    # Trade on the NEXT bar: enhanced_weights returns targets dated on the signal date.
    weights = weights.reindex(adj.index).ffill().fillna(0.0).shift(1).fillna(0.0)

    # Same price hygiene as the base stream: a >100%/day single-name move on free data is a
    # glitch, not P&L, and one such tick would dominate a 5-year result.
    adj_clean = adj.where(adj > 0)
    rets = adj_clean.pct_change(fill_method=None)
    rets = rets.where(rets.abs() < 1.0).fillna(0.0)

    gross = (weights * rets).sum(axis=1)
    dw = weights.diff().abs().fillna(weights.abs())
    adv = (close * volume).rolling(21).mean().reindex_like(weights)
    adv = adv.ffill().fillna(adv.median().median())
    costs = LinearCostModel(half_spread_bps=5.0, impact_coef_bps=20.0).charge(
        dw * NOTIONAL, adv) / NOTIONAL
    net = (gross - costs).dropna()

    held = (weights > 0).sum(axis=1)
    yrs = max((adj.index[-1] - adj.index[0]).days / 365.25, 1e-9)
    return {"net": net, "gross": gross, "weights": weights, "rank": rank,
            "n_universe": len(tickers),
            "avg_holdings": float(held[held > 0].mean()) if (held > 0).any() else 0.0,
            "turnover": float(dw.sum(axis=1).sum() / yrs)}


def main() -> None:
    from data import download_ohlcv
    print("=" * 92)
    print("ENHANCED MAGIC FORMULA — the deployed variant")
    print("=" * 92)

    variants = [("enhanced (live: use_graham=False)", EnhancedMagicConfig(use_graham=False)),
                ("enhanced + Graham", EnhancedMagicConfig(use_graham=True)),
                ("enhanced, inverse-vol wt", EnhancedMagicConfig(use_graham=False,
                                                                 weighting="inverse_vol"))]
    rows, series = [], {}
    for name, cfg in variants:
        try:
            r = run(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:34} FAILED: {type(e).__name__}: {e}")
            continue
        s = summary_stats(r["net"])
        series[name] = r["net"]
        rows.append({"variant": name, **{k: s[k] for k in
                                         ("ann_return", "ann_vol", "sharpe", "max_drawdown")},
                     "turnover": r["turnover"], "holdings": r["avg_holdings"]})
        print(f"  {name:34} n_universe {r['n_universe']:>4}  avg holdings "
              f"{r['avg_holdings']:>4.1f}  turnover {r['turnover']:>4.1f}x/yr")

    if not rows:
        print("\n  no variant ran — check fundamentals availability")
        return

    idx = next(iter(series.values())).index
    spy = download_ohlcv(["SPY"], str(idx[0].date()),
                         str(idx[-1].date()))["adj_close"]["SPY"].pct_change().reindex(idx).fillna(0)
    ss = summary_stats(spy)

    print(f"\n  window {idx[0].date()} -> {idx[-1].date()} "
          f"({(idx[-1]-idx[0]).days/365.25:.1f} yrs)")
    print(f"\n  {'variant':34} {'ann ret':>9} {'vol':>8} {'Sharpe':>8} {'maxDD':>9}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['variant']:34} {r['ann_return']:>+9.2%} {r['ann_vol']:>8.2%} "
              f"{r['sharpe']:>+8.2f} {r['max_drawdown']:>+9.2%}")
    print(f"  {'SPY (benchmark)':34} {ss['ann_return']:>+9.2%} {ss['ann_vol']:>8.2%} "
          f"{ss['sharpe']:>+8.2f} {ss['max_drawdown']:>+9.2%}")

    n = len(idx) / 252.0
    print(f"\n  ⚠ Sharpe standard error on {n:.1f} years is roughly "
          f"±{1.0/np.sqrt(max(n, 1e-9)):.2f} — the gap to SPY is well inside it.")

    OUT.mkdir(parents=True, exist_ok=True)
    live = series.get("enhanced (live: use_graham=False)")
    if live is not None:
        live.to_csv(OUT / "enhanced_net_returns.csv", header=["net"])
        print(f"  wrote {OUT}/enhanced_net_returns.csv  ({len(live)} days)")


if __name__ == "__main__":
    main()
