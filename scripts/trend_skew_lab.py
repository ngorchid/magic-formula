"""Skew lab — sweep trend-overlay signal/vol-targeting variants to chase POSITIVE skew.

The live overlay (strategies/trend_futures/overlay.py) comes out with NEGATIVE skew, which
is backwards for trend-following (trend should be positive-skew = many small losses, rare big
wins = crisis alpha). Two suspects: (1) the binary np.sign signal caps winners at the same
size as dawdlers, (2) reactive 60d vol-targeting shrinks positions right as a trend spikes vol.

This harness reimplements the overlay pipeline with knobs so we can A/B variants over the SAME
7-market live basket and SEE the Sharpe<->skew tradeoff. Non-destructive: does not import or
modify overlay.py. Run: python scripts/trend_skew_lab.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from data import download_ohlcv

# The 7 markets actually traded live, via their ETF proxies (ES/ZN/GC/HG/CL/6E/6A).
LIVE7 = ["SPY", "IEF", "GLD", "CPER", "USO", "FXE", "FXA"]


@dataclass
class Variant:
    name: str
    signal_mode: str = "sign"        # "sign" (binary blend) | "cont" (vol-scaled tanh)
    lookbacks: tuple[int, ...] = (63, 126, 252)
    vol_window: int = 60             # per-name inverse-vol sizing window
    pvol_window: int = 60            # portfolio vol-target window (rolling mode)
    pvol_mode: str = "rolling"       # "rolling" (adaptive) | "constant" (cosmetic full-sample rescale)
    cont_scale: float = 1.0          # tanh saturation for continuous signal
    target_vol: float = 0.10
    rebalance: str = "W-FRI"
    max_leverage: float = 5.0
    half_spread_bps: float = 2.0


def _signal(prices: pd.DataFrame, rets: pd.DataFrame, v: Variant) -> pd.DataFrame:
    """Position direction+strength in [-1, +1]."""
    if v.signal_mode == "sign":
        # Binary sign-blend: throws away magnitude (the current live behaviour).
        return sum(np.sign(prices / prices.shift(lb) - 1.0) for lb in v.lookbacks) / len(v.lookbacks)
    # Continuous: risk-adjusted momentum (return / its expected horizon vol), averaged over
    # lookbacks, squashed through tanh so it stays bounded but SCALES with trend strength.
    vol_ann = (rets.rolling(v.vol_window).std() * np.sqrt(252)).replace(0.0, np.nan)
    zs = []
    for lb in v.lookbacks:
        mom = prices / prices.shift(lb) - 1.0
        horizon_vol = vol_ann * np.sqrt(lb / 252.0)   # expected size of an lb-horizon move
        zs.append(mom / horizon_vol)
    z = sum(zs) / len(zs)
    return np.tanh(z / v.cont_scale)


def run_variant(prices: pd.DataFrame, v: Variant) -> tuple[pd.Series, float]:
    rets = prices.pct_change(fill_method=None)
    N = prices.shape[1]

    signal = _signal(prices, rets, v).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Inverse-vol risk parity: each market targets ~target_vol/sqrt(N) standalone risk.
    vol = (rets.rolling(v.vol_window).std() * np.sqrt(252)).replace(0.0, np.nan)
    raw = signal * (v.target_vol / np.sqrt(N)) / vol
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Weekly rebalance, ffill, lag 1 day (no lookahead).
    rebal = prices.resample(v.rebalance).last().index.intersection(prices.index)
    target = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    target.loc[rebal] = raw.loc[rebal]
    weights = target.ffill().fillna(0.0).shift(1).fillna(0.0)

    # Portfolio vol target — the skew lever. "rolling" = adaptive (kills skew by cutting
    # winners mid-trend). "constant" = single full-sample rescale so exposure GROWS in strong
    # trends; the rescale is cosmetic (skew & Sharpe are scale-invariant) so no lookahead in shape.
    unscaled = (weights * rets.fillna(0.0)).sum(axis=1)
    if v.pvol_mode == "constant":
        realized = unscaled.std() * np.sqrt(252)
        scale = pd.Series(v.target_vol / realized if realized else 1.0, index=weights.index)
    else:
        pvol = (unscaled.rolling(v.pvol_window).std() * np.sqrt(252)).replace(0.0, np.nan)
        scale = (v.target_vol / pvol).clip(upper=v.max_leverage).ffill().fillna(1.0)
    weights = weights.mul(scale, axis=0)
    gross = weights.abs().sum(axis=1)
    weights = weights.mul((v.max_leverage / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0), axis=0)

    gross_ret = (weights * rets.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().fillna(weights.abs()).sum(axis=1)
    net_ret = gross_ret - turnover * (v.half_spread_bps / 1e4)
    return net_ret, float(weights.abs().sum(axis=1).mean())


def main(start: str = "2011-01-01", end: str | None = None) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"loading {len(LIVE7)}-market live basket {LIVE7} …")
    prices = download_ohlcv(LIVE7, start, end)["adj_close"].dropna(how="all", axis=1).sort_index()
    spy = download_ohlcv(["SPY"], start, end)["adj_close"]["SPY"].pct_change(fill_method=None)

    variants = [
        Variant("V0 baseline (live)"),
        # Combine sweep-2 winners: drop the fast 63d drag + rebalance daily.
        Variant("A slow252 weekly",        lookbacks=(252,)),
        Variant("B slow252 daily",         lookbacks=(252,), rebalance="D"),
        Variant("C 126+252 weekly",        lookbacks=(126, 252)),
        Variant("D 126+252 daily",         lookbacks=(126, 252), rebalance="D"),
        Variant("E full-blend daily",      rebalance="D"),
        Variant("F 126+252 d + cont",      lookbacks=(126, 252), rebalance="D", signal_mode="cont"),
    ]

    rows = []
    series = {}
    for v in variants:
        net, gl = run_variant(prices, v)
        series[v.name] = net
        common = net.replace(0.0, np.nan).dropna().index.intersection(spy.dropna().index)
        r = net.reindex(common).fillna(0.0)
        s = summary_stats(r)
        corr = r.corr(spy.reindex(common).fillna(0.0))
        skew = r.skew()
        m_spy = (1 + spy.reindex(common).fillna(0.0)).resample("ME").prod() - 1
        m_trd = (1 + r).resample("ME").prod() - 1
        worst = m_spy <= m_spy.quantile(0.10)
        rows.append((v.name, s["ann_return"], s["ann_vol"], s["sharpe"], s["max_drawdown"],
                     skew, corr, m_trd[worst].mean(), gl))

    print("\n" + "=" * 96)
    print(f"TREND SKEW LAB  ({start}→{end}, 7-market live basket, net)")
    print("=" * 96)
    hdr = f"  {'variant':22s} {'ret':>7s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'SKEW':>6s} {'corrSPY':>8s} {'crisis':>7s} {'grossLv':>7s}"
    print(hdr)
    print("  " + "-" * 92)
    for name, ret, vol, sh, dd, sk, co, cr, gl in rows:
        print(f"  {name:22s} {ret:>+7.2%} {vol:>7.2%} {sh:>+7.2f} {dd:>+8.2%} {sk:>+6.2f} {co:>+8.2f} {cr:>+7.2%} {gl:>6.1f}x")

    out = ROOT / "results" / "trend_overlay"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(series).to_csv(out / "skew_lab_variants.csv")
    print(f"\n  wrote {out}/skew_lab_variants.csv")


if __name__ == "__main__":
    main()
