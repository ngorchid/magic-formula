"""Build a daily SPX option chain with implied vol and delta, from OPRA daily bars.

The raw feed gives traded OHLCV per contract. To select strikes by DELTA (which is how the
options-vrp strategy is specified) we need implied vol, so we invert Black-Scholes on the
closing trade price and derive delta from the same model — internally consistent, which is
what matters for strike selection even if the absolute IV level is slightly off.

TWO MAPPING TRAPS in the raw data, both silent and both fatal if missed:
  1. Databento RECYCLES instrument_id — 81% of ids map to >1 contract over time (up to 221),
     so the id->contract map must be date-aware or you get negative days-to-expiry.
  2. instrument_id is ALSO not stable per contract — 40% of contracts are reassigned a new id
     during their life (one 838 times). **Key contracts by (expiry, cp, strike), never by id.**
     Keying by id made mark-continuity look like 5% when it is really 76%.

DATA LIMITS to keep in view:
  - ohlcv-1d is TRADES ONLY: no bid/ask, so spread cost cannot be measured here, and a
    contract with no trade on a day simply has no bar (~24% of days for a typical short leg).
  - SPX, not SPY, and not the 14-name basket. European, cash-settled, 10x notional.

Run: python scripts/spx_chain.py        # builds + validates the chain, writes chain.parquet
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

DATA = Path.home() / "aktien/trading/data/opra"
OUT = ROOT / "results" / "spx_vrp"

# Universe we actually need: puts and calls near the money, at tradeable tenors. Filtering
# first cuts 4.7M bars to a few hundred thousand and makes the BS inversion cheap.
DTE_MIN, DTE_MAX = 10, 70
MNY_LO, MNY_HI = -0.30, 0.15        # strike/spot - 1
DIV_YIELD = 0.018                    # SPX ~1.8%; shifts all deltas consistently


def bs_price(S, K, T, r, q, sigma, cp):
    """Black-Scholes with continuous dividend yield. cp=+1 call, -1 put."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return cp * (S * np.exp(-q * T) * norm.cdf(cp * d1)
                     - K * np.exp(-r * T) * norm.cdf(cp * d2))


def bs_vega(S, K, T, r, q, sigma):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, q, cp, n_iter=60):
    """Vectorised Newton with a bisection safety net.

    Newton alone fails on deep-OTM options where vega ~ 0, so bracket [1e-3, 5.0] and fall
    back to bisection wherever Newton leaves the bracket or stalls.
    """
    lo = np.full_like(price, 1e-3)
    hi = np.full_like(price, 5.0)
    # intrinsic check: a price below intrinsic has no solution
    intrinsic = np.maximum(cp * (S * np.exp(-q * T) - K * np.exp(-r * T)), 0.0)
    ok = (price > intrinsic + 1e-6) & (T > 0) & (price > 0)
    sigma = np.where(ok, 0.20, np.nan)
    for _ in range(n_iter):
        p = bs_price(S, K, T, r, q, sigma, cp)
        diff = p - price
        v = bs_vega(S, K, T, r, q, sigma)
        # tighten the bracket
        hi = np.where(diff > 0, np.minimum(hi, sigma), hi)
        lo = np.where(diff < 0, np.maximum(lo, sigma), lo)
        step = np.where(v > 1e-8, diff / v, 0.0)
        nxt = sigma - step
        bad = ~np.isfinite(nxt) | (nxt <= lo) | (nxt >= hi)
        sigma = np.where(bad, 0.5 * (lo + hi), nxt)
    sigma = np.where(ok & (np.abs(bs_price(S, K, T, r, q, sigma, cp) - price) < 0.5), sigma, np.nan)
    return sigma


def bs_delta(S, K, T, r, q, sigma, cp):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return cp * np.exp(-q * T) * norm.cdf(cp * d1)


def load_rates() -> pd.Series:
    """3-month T-bill from FRED, keyless, as the discount rate."""
    import requests, io
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO", timeout=60)
    s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0]
    return (pd.to_numeric(s, errors="coerce") / 100.0).ffill()


def build_chain() -> pd.DataFrame:
    import yfinance as yf
    bars = pd.read_parquet(DATA / "bars.parquet")
    spot = yf.download("^GSPC", start="2013-01-01", auto_adjust=False, progress=False)["Close"]
    if hasattr(spot, "columns"):
        spot = spot.iloc[:, 0]
    spot.index = pd.to_datetime(spot.index)
    rates = load_rates()

    df = bars.copy()
    df["spot"] = df.date.map(spot)
    df["r"] = df.date.map(rates)
    df = df.dropna(subset=["spot", "r"])
    df["mny"] = df.strike / df.spot - 1.0
    df = df[df.dte.between(DTE_MIN, DTE_MAX) & df.mny.between(MNY_LO, MNY_HI)]

    # KEY BY CONTRACT — see the module docstring on why instrument_id must not be used
    df["contract"] = (df.expiry.dt.strftime("%y%m%d") + df.cp
                      + (df.strike * 1000).astype("int64").astype(str))
    df = df.drop_duplicates(["date", "contract"])
    print(f"  filtered to {len(df):,} bars ({df.date.nunique():,} days, "
          f"{df.contract.nunique():,} contracts)")

    cp = np.where(df.cp.values == "C", 1.0, -1.0)
    T = df.dte.values / 365.0
    df["iv"] = implied_vol(df.close.values, df.spot.values, df.strike.values,
                           T, df.r.values, DIV_YIELD, cp)
    df["delta"] = bs_delta(df.spot.values, df.strike.values, T, df.r.values,
                           DIV_YIELD, df.iv.values, cp)
    print(f"  IV solved for {df.iv.notna().mean():.1%} of bars")
    return df


def validate(df: pd.DataFrame) -> None:
    """The load-bearing check: our ATM 30d IV must track VIX. If it doesn't, the inversion
    is wrong and every delta downstream is wrong with it."""
    import yfinance as yf
    vix = yf.download("^VIX", start="2013-01-01", auto_adjust=False, progress=False)["Close"]
    if hasattr(vix, "columns"):
        vix = vix.iloc[:, 0]
    atm = df[(df.dte.between(25, 35)) & (df.mny.abs() < 0.01) & df.iv.notna()]
    daily = atm.groupby("date").iv.median() * 100
    common = daily.index.intersection(vix.index)
    a, b = daily.reindex(common), vix.reindex(common)
    print("\n" + "=" * 74)
    print("VALIDATION — our ATM ~30d IV vs VIX")
    print("=" * 74)
    print(f"  overlapping days   : {len(common):,}")
    print(f"  correlation        : {a.corr(b):+.4f}")
    print(f"  mean ours / VIX    : {a.mean():.2f} / {b.mean():.2f}   (diff {a.mean()-b.mean():+.2f})")
    print(f"  median abs error   : {(a-b).abs().median():.2f} vol points")
    print(f"  90th pct abs error : {(a-b).abs().quantile(.9):.2f}")
    ok = a.corr(b) > 0.95 and (a - b).abs().median() < 3.0
    print(f"\n  {'PASS' if ok else 'FAIL'} — inversion is {'sound' if ok else 'NOT trustworthy'}")


def main() -> None:
    print("Building SPX chain with IV + delta...")
    df = build_chain()
    validate(df)
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["date", "expiry", "dte", "cp", "strike", "close", "volume",
            "spot", "r", "mny", "iv", "delta", "contract"]
    df[cols].to_parquet(OUT / "chain.parquet", index=False)
    print(f"\n  wrote {OUT}/chain.parquet ({len(df):,} rows)")


if __name__ == "__main__":
    main()
