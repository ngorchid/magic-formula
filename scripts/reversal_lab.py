"""Reversal lab — daily cross-sectional short-term reversal, market-neutral, cost-stressed.

The medallion-style short-horizon idea: uninformed flow pushes stocks off fair value over
a few days; a liquidity provider leans against it and gets paid as they revert. The naive
version (buy the raw 5-day losers) is decayed and eaten by costs. Two refinements keep it
alive, both already in this repo:

  1. RESIDUAL reversal — revert the move that is LEFT after removing market (beta) and sector
     moves (`neutralize`), not the raw return. You're not just buying the sector that fell.
  2. NEWS filter — moves on a volume spike are likely news-driven and DRIFT; moves on quiet
     volume are noise and REVERT. We proxy "news" by a volume spike over the lookback window
     (no free earnings calendar at 500-name scale) and mute those names.

Construction: daily, dollar/beta/sector-neutral z-score book, 1-day lag (signal at close t,
held t+1), gross leverage 1. Costs via the repo's spread+sqrt-impact model on real ADV, stressed
at 1x / 2x / 3x. The whole question at this horizon is whether the edge survives costs — so the
table leads with net Sharpe at three cost levels, exactly like the trend cost-stress test.

Run: python scripts/reversal_lab.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import LinearCostModel, summary_stats
from combination import cs_zscore, winsorize
from data import download_ohlcv, sp500_sectors, sp500_tickers
from signals import short_term_reversal
from strategies.equity_mn.neutralize import neutralize, rolling_beta


@dataclass
class Variant:
    name: str
    lookback: int = 5
    horizons: tuple[int, ...] = ()   # if set, blend sign-flipped reversal over these lookbacks
    beta_neutral: bool = True
    sector_neutral: bool = True
    news_filter: bool = False        # mute names with a volume spike over the lookback
    news_z: float = 3.0              # volume-spike threshold (in vol z-units)
    smooth: int = 1                  # average the target book over N days (cuts turnover)
    decile: float = 0.0              # keep only the top/bottom `decile` fraction by |signal| (0 = all)
    band: float = 0.0                # no-trade band: skip a name's trade if |target-held| < band*avg_wt
    inv_vol: bool = False            # risk-weight: divide signal by each name's realized vol
    vix_scale: bool = False          # scale daily gross by VIX vs its median (lean in when vol high)


def build_weights(v: Variant, prices, volume, betas, sectors,
                  idio_vol=None, vix=None, eligible=None) -> pd.DataFrame:
    # Reversal signal: single lookback, or a blend of horizons (each z-scored then averaged).
    horizons = v.horizons or (v.lookback,)
    parts = [cs_zscore(winsorize(short_term_reversal(prices, lookback=h).reindex_like(prices),
                                 0.01, 0.99)) for h in horizons]
    z = sum(parts) / len(parts)

    # Restrict to point-in-time index members (survivorship fix), if provided.
    if eligible is not None:
        z = z.where(eligible.reindex_like(z).fillna(False), 0.0)

    # Residualize (the refinement): strip beta + sector so we revert idiosyncratic moves.
    z = neutralize(z, betas=betas if v.beta_neutral else None,
                   sectors=sectors if v.sector_neutral else None)

    # Risk-weight: down-weight high-vol names so risk (not just signal) is balanced.
    if v.inv_vol and idio_vol is not None:
        z = z.div(idio_vol.reindex_like(z).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # News proxy: mute names whose volume spiked over the lookback (likely news -> drift).
    if v.news_filter:
        vol_z = (volume - volume.rolling(63).mean()) / volume.rolling(63).std()
        spiked = (vol_z > v.news_z).rolling(v.lookback).max().fillna(0.0) > 0
        z = z.where(~spiked, 0.0)

    # Concentrate: keep only the most extreme signals (top/bottom `decile` fraction per day).
    if v.decile > 0:
        lo = z.quantile(v.decile, axis=1)
        hi = z.quantile(1.0 - v.decile, axis=1)
        z = z.where(z.le(lo, axis=0) | z.ge(hi, axis=0), 0.0)

    # Dollar-neutral book at gross leverage 1: demean, then normalize by gross.
    z = z.sub(z.mean(axis=1), axis=0)
    w = z.div(z.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    if v.smooth > 1:
        w = w.rolling(v.smooth).mean().fillna(0.0)

    # No-trade band: only move a name when the target shifts more than band*avg_weight,
    # a cheap proxy for the optimizer's turnover penalty (skip churn that can't pay its cost).
    if v.band > 0:
        tgt = w.values
        held = np.zeros(tgt.shape[1])
        thr = v.band / max(1, int((np.abs(tgt) > 0).sum(axis=1).mean()))  # band * avg name weight
        held_rows = np.empty_like(tgt)
        for i in range(tgt.shape[0]):
            move = np.abs(tgt[i] - held) > thr
            held = np.where(move, tgt[i], held)
            held_rows[i] = held
        w = pd.DataFrame(held_rows, index=w.index, columns=w.columns)

    # VIX conditioning: lean in when vol is elevated (liquidity provision pays more).
    if v.vix_scale and vix is not None:
        scale = (vix / vix.rolling(252, min_periods=60).median()).clip(0.5, 2.0)
        w = w.mul(scale.reindex(w.index).ffill().fillna(1.0), axis=0)

    return w.shift(1).fillna(0.0)          # lag 1 day: signal at close t, held t+1


def evaluate(v: Variant, prices, volume, betas, sectors, notional=1_000_000.0,
             idio_vol=None, vix=None, eligible=None):
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    w = build_weights(v, prices, volume, betas, sectors, idio_vol, vix, eligible)
    gross = (w * rets).sum(axis=1)

    dw = w.diff().abs().fillna(w.abs())
    turnover = dw.sum(axis=1)
    adv = (prices * volume).rolling(21).mean().reindex_like(w).ffill()
    adv = adv.fillna(adv.median().median())

    out = {"gross": gross, "turnover": float(turnover.mean()),
           "net_beta": float((w * betas.reindex_like(w)).sum(axis=1).mean()),
           "gross_lev": float(w.abs().sum(axis=1).mean())}
    # Scenarios by EXECUTION STYLE (spread), impact held/stressed:
    #   moc   = closing auction / limit: ~0.5bp spread (you provide liquidity)
    #   cross = market-order taker:       2.5bp spread
    #   stress= taker + 2x impact
    for tag, spread, impact in (("moc", 0.5, 10.0), ("cross", 2.5, 10.0), ("stress", 2.5, 20.0)):
        cm = LinearCostModel(half_spread_bps=spread, impact_coef_bps=impact)
        out[f"net_{tag}"] = gross - cm.charge(dw * notional, adv) / notional
    return out


def _sh(r, common):
    return summary_stats(r.reindex(common).fillna(0.0))["sharpe"]


def _moc_sharpe(v, common, **kw):
    r = evaluate(v, **kw)
    return r, _sh(r["net_moc"], common)


def main(start="2011-01-01", end=None):
    import yfinance as yf
    from data.universe import sp500_pit_eligible, sp500_pit_universe  # noqa: E402

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sp500_tickers()
    print(f"loading {len(tickers)} S&P 500 names + SPY, {start}->{end} …")
    panel = download_ohlcv(sorted(set(tickers + ["SPY"])), start, end)
    prices_full = panel["adj_close"].dropna(how="all", axis=1)
    volume = panel["volume"].reindex_like(prices_full)
    bench = prices_full["SPY"].pct_change(fill_method=None)
    prices = prices_full.drop(columns=["SPY"], errors="ignore")
    volume = volume.drop(columns=["SPY"], errors="ignore")
    rets = prices.pct_change(fill_method=None)
    betas = rolling_beta(rets, bench, window=252)
    sectors = sp500_sectors().reindex(prices.columns)
    idio_vol = rets.rolling(20).std()                              # for inverse-vol weighting
    vix = (yf.download("^VIX", start=start, end=end, auto_adjust=True,
                       progress=False)["Close"].squeeze() / 100.0)  # for VIX conditioning
    # Point-in-time membership mask (survivorship fix) over the current price panel.
    try:
        elig = sp500_pit_eligible(prices.index, list(prices.columns))
    except Exception as e:  # noqa: BLE001
        print(f"  [pit eligibility unavailable: {type(e).__name__}] — skipping PIT check"); elig = None
    print(f"  {prices.shape[1]} names with data")

    kw = dict(prices=prices, volume=volume, betas=betas, sectors=sectors,
              idio_vol=idio_vol, vix=vix)

    # --- Section 1: Sharpe levers (all net-of-MOC-cost) ---
    variants = [
        Variant("V3 baseline (5d,2d)", news_filter=True, smooth=2),
        Variant("L1 multi-horizon", horizons=(1, 3, 5, 10), news_filter=True, smooth=2),
        Variant("L2 + inverse-vol", horizons=(1, 3, 5, 10), news_filter=True, smooth=2, inv_vol=True),
        Variant("L3 + VIX conditioning", horizons=(1, 3, 5, 10), news_filter=True, smooth=2,
                inv_vol=True, vix_scale=True),
    ]
    print("\n" + "=" * 100)
    print(f"REVERSAL LAB — SHARPE LEVERS  ({start}->{end}, S&P500, daily, net of MOC cost)")
    print("=" * 100)
    hdr = (f"  {'variant':22s} {'grossSh':>7s} {'MOC':>6s} {'cross':>6s} {'stress':>6s} "
           f"{'annMOC':>7s} {'turn/day':>8s} {'netBeta':>7s}")
    print(hdr + "\n  " + "-" * 98)
    best = None
    for v in variants:
        r = evaluate(v, **kw)
        common = r["gross"].replace(0.0, np.nan).dropna().index
        gs = _sh(r["gross"], common)
        moc, cross, stress = (_sh(r[f"net_{t}"], common) for t in ("moc", "cross", "stress"))
        ann = summary_stats(r["net_moc"].reindex(common).fillna(0.0))["ann_return"]
        print(f"  {v.name:22s} {gs:>+7.2f} {moc:>+6.2f} {cross:>+6.2f} {stress:>+6.2f} "
              f"{ann:>+7.1%} {r['turnover']:>7.1%} {r['net_beta']:>+7.2f}")
        if best is None or moc > best[1]:
            best = (v, moc, r)
    champ = best[0]
    print(f"\n  champion = {champ.name}  (net-MOC Sharpe {best[1]:+.2f})")

    # --- Section 2: Robustness of the champion ---
    net = best[2]["net_moc"]
    print("\n" + "=" * 100)
    print(f"ROBUSTNESS — {champ.name}")
    print("=" * 100)
    print("  sub-period stability (net-MOC):")
    for lbl, a, b in [("2011-2015", "2011-01-01", "2015-12-31"),
                      ("2016-2020", "2016-01-01", "2020-12-31"),
                      ("2021-2026", "2021-01-01", end)]:
        seg = net.loc[a:b].replace(0.0, np.nan).dropna()
        s = summary_stats(seg.fillna(0.0))
        print(f"    {lbl:11s} Sharpe {s['sharpe']:>+5.2f}  ann {s['ann_return']:>+6.1%}  maxDD {s['max_drawdown']:>+6.1%}")

    print("\n  MOC-cost sensitivity (how fragile is the 0.5bp assumption?):")
    common = best[2]["gross"].replace(0.0, np.nan).dropna().index
    rets_g = best[2]["gross"]
    w = build_weights(champ, prices, volume, betas, sectors, idio_vol, vix)
    dw = w.diff().abs().fillna(w.abs())
    adv = (prices * volume).rolling(21).mean().reindex_like(w).ffill()
    adv = adv.fillna(adv.median().median())
    for spread in (0.5, 1.0, 1.5, 2.5):
        cm = LinearCostModel(half_spread_bps=spread, impact_coef_bps=10.0)
        n = rets_g - cm.charge(dw * 1_000_000.0, adv) / 1_000_000.0
        print(f"    spread {spread:>4.1f}bp  ->  net Sharpe {_sh(n, common):>+5.2f}")

    if elig is not None:
        print("\n  survivorship check (point-in-time index membership per date):")
        r_pit = evaluate(champ, eligible=elig, **kw)
        c2 = r_pit["gross"].replace(0.0, np.nan).dropna().index
        print(f"    current-members  net-MOC Sharpe {best[1]:>+5.2f}")
        print(f"    point-in-time    net-MOC Sharpe {_sh(r_pit['net_moc'], c2):>+5.2f}  "
              f"(residual delisting bias remains — see universe.py)")


if __name__ == "__main__":
    main()
