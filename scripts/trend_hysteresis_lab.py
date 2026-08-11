"""Does a hysteresis band on contract counts actually help, or just cut turnover?

THE PROBLEM. Sizing produces a CONTINUOUS dollar target but positions are WHOLE contracts, and
`round()` puts a knife-edge at exactly 0.5 contracts. A market whose target sits near that
boundary flips 0 -> 1 -> 0 on tiny changes in vol or signal, each flip a full-contract round trip.
Measured on the live basket: at $200k, rates_10y flips 12.4x/yr at $112,000 a time = $1.39M of
notional churned by noise. It does not shrink with budget, it MOVES -- every budget level parks
some market on its own boundary.

THE FIX UNDER TEST. Hold position n unless |target - n| >= band, then move to round(target). With
band 0.7: flat, you need target >= 0.7 to open; long 1, you need <= 0.3 to close. Reversing then
requires a 0.4-contract move instead of an arbitrarily small one. NB band 0.5 is EXACTLY today's
behaviour (`round()` changes precisely when |target - n| >= 0.5), which is a useful null check --
the sweep should reproduce the baseline there.

WHY THIS SCRIPT EXISTS. Turnover and tracking error were already measured (band 0.8 cuts dollar
turnover 33% for 0.06 contracts of drift). But LOWER TURNOVER IS NOT AUTOMATICALLY BETTER -- the
same basket showed daily rebalancing beating weekly, because the signal decays. The claim needing
a test is that hysteresis removes NOISE trades while keeping SIGNAL trades. That has to show up in
Sharpe, drawdown and skew, not in a turnover column.

CONTRACT-LEVEL, deliberately. The earlier exposure lab worked in weight space, where rounding does
not exist and the whole effect is invisible.

Run: python scripts/trend_hysteresis_lab.py
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

from backtest import summary_stats  # noqa: E402

# live basket: (proxy ETF, $ notional per contract with use_micro=True)
BASKET = [("SPY", 27_500), ("IEF", 112_000), ("GLD", 30_000), ("CPER", 11_200),
          ("USO", 7_000), ("FXE", 13_500), ("FXA", 6_600), ("TLT", 115_000),
          ("SLV", 35_000), ("FXY", 8_200)]
OUT = ROOT / "results" / "trend_overlay"

COMMISSION = 0.85      # $ per contract per side, IB all-in (see broker-fees memory)
HALF_SPREAD_BPS = 1.0  # bps of traded notional


def contract_path(target: pd.Series, band: float) -> pd.Series:
    """Held contracts over time. band<=0.5 reproduces plain round()."""
    out = np.zeros(len(target))
    n = 0
    vals = target.values
    for i, x in enumerate(vals):
        if np.isfinite(x) and (band <= 0 or abs(x - n) >= band):
            n = int(np.round(x))
        out[i] = n
    return pd.Series(out, index=target.index)


def run(px: pd.DataFrame, band: float, budget: float, target_vol: float = 0.10,
        vol_window: int = 60, lookbacks=(126, 252), vol_floor_pct: float = 0.20,
        per_market_cap: float = 0.40) -> dict:
    rets = px.pct_change(fill_method=None)
    N = px.shape[1]
    sig = sum(np.sign(px / px.shift(lb) - 1.0) for lb in lookbacks) / len(lookbacks)
    vol = rets.rolling(vol_window).std() * np.sqrt(252)
    floor = vol.expanding(min_periods=252).quantile(vol_floor_pct)
    vol_used = vol.where(floor.isna(), np.maximum(vol, floor))

    notional_held = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    traded_ct = pd.Series(0.0, index=px.index)
    traded_usd = pd.Series(0.0, index=px.index)
    for etf, notl in BASKET:
        if etf not in px.columns:
            continue
        expo = (sig[etf] * (budget * target_vol / np.sqrt(N)) / vol_used[etf]).clip(
            -per_market_cap * budget, per_market_cap * budget)
        ct = contract_path((expo / notl).fillna(0.0), band)
        notional_held[etf] = ct * notl
        d = ct.diff().abs().fillna(0.0)
        traded_ct += d
        traded_usd += d * notl

    # P&L: yesterday's held notional earns today's return (shift avoids lookahead)
    gross_ret = (notional_held.shift(1) * rets.fillna(0.0)).sum(axis=1) / budget
    cost = (traded_ct * COMMISSION + traded_usd * HALF_SPREAD_BPS / 1e4) / budget
    net = (gross_ret - cost).dropna()
    s = summary_stats(net)
    yrs = (net.index[-1] - net.index[0]).days / 365.25
    return {"band": band, "sharpe": s["sharpe"], "ret": s["ann_return"], "vol": s["ann_vol"],
            "maxdd": s["max_drawdown"], "skew": float(net.skew()),
            "ct_yr": traded_ct.sum() / yrs, "usd_yr": traded_usd.sum() / yrs,
            "cost_yr": float(cost.sum() / yrs)}


def main() -> None:
    from data import download_ohlcv
    etfs = [e for e, _ in BASKET]
    print(f"loading {len(etfs)} proxies …")
    px = download_ohlcv(etfs, "2011-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))[
        "adj_close"].dropna(how="all", axis=1).sort_index()

    for budget in (100_000, 200_000):
        print("\n" + "=" * 100)
        print(f"HYSTERESIS BAND SWEEP — ${budget:,} budget, contract-level, net of "
              f"${COMMISSION}/ct + {HALF_SPREAD_BPS}bp")
        print("=" * 100)
        print(f"  {'band':>6} {'Sharpe':>8} {'return':>8} {'vol':>7} {'maxDD':>8} {'skew':>7} | "
              f"{'ct/yr':>7} {'$ turn/yr':>12} {'cost/yr':>9}")
        print("  " + "-" * 94)
        rows = []
        for band in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25):
            r = run(px, band, float(budget))
            rows.append(r)
            tag = "   <- today (== 0.0)" if band == 0.5 else ""
            print(f"  {band:>6.2f} {r['sharpe']:>+8.3f} {r['ret']:>+8.2%} {r['vol']:>7.2%} "
                  f"{r['maxdd']:>+8.2%} {r['skew']:>+7.2f} | {r['ct_yr']:>7.0f} "
                  f"{r['usd_yr']:>11,.0f}$ {r['cost_yr']:>8.2%}{tag}")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / f"hysteresis_{budget}.csv", index=False)
    print(f"\n  wrote {OUT}/hysteresis_*.csv")
    print("\n  band 0.5 MUST equal band 0.0 — round() changes exactly when |target-n| >= 0.5.")
    print("  Adopt a band only if Sharpe/maxDD/skew improve; turnover alone is not a reason.")


if __name__ == "__main__":
    main()
