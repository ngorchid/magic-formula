"""Does the widened European universe change the magic-formula BOOK VOL, and so its prior?

THE QUESTION. `run_paper.py:64` sets `VOL_PRIOR = 0.190`, derived from the authoritative
S&P 500 point-in-time backtest. The circuit-breaker levels are sigmas of that number
(`BreakerLevels.from_vol`, SIGMAS 1.2/2.0/2.8), so at 19.0% they sit at 22.8 / 38.0 / 53.2%
drawdown. On 2026-08-28 the universe went 764 -> 964 names by adding all of Europe, ~200 of
them quoted in non-EUR currencies and held UNHEDGED. `risk_guard.blended_vol` shrinks live
realised vol toward the prior, so the recorded history cannot react to a configuration change
for months -- the prior is the only input that can.

WHY THIS IS NOT A BACKTEST. `run_best_magic.py` loads fundamentals with `sources=("edgar",)`,
which is US filings only, and the European names have no point-in-time membership data either.
The 19.0% came from a US PIT series precisely because that is what is backtestable. Re-running
"the backtest on the new universe" is therefore not available, and pretending otherwise would
produce a number with a survivorship-biased factor signal baked into it.

WHAT THIS MEASURES INSTEAD. The prior is a claim about the volatility of a 30-name equal-weight
book. That is a property of the OPPORTUNITY SET -- the constituents' own vols and their
correlations -- far more than of which factor picked them. So this draws random equal-weight
30-name books from each universe arm and measures the distribution of realised annualised vol:

  arm 1  US only                 (what the 19.0% prior describes)
  arm 2  US + eurozone           (the universe as it stood before 2026-08-28)
  arm 3  US + all Europe         (the universe as it stands now)

Returns are converted to USD, so a European name carries its unhedged currency move exactly as
the live book does. FX scale cancels in a return series, so pence-vs-pounds does not matter
here (it very much does for market cap -- see _MINOR_UNITS in paper/live_data.py).

HOW TO READ IT. The LEVEL of arm 1 is the validity check: if random 30-name US books do not
land near the 19.0% the PIT backtest produced, the method is not measuring the right thing and
the deltas should be ignored. The DELTAS between arms are the answer, and they are far more
robust than the levels -- survivorship bias, factor tilt and sample period hit all three arms
alike and largely difference out.

⚠ This deliberately does NOT model the factor tilt. A value/quality screen selects cheaper,
more defensive names than a random draw, so the absolute levels here should run somewhat above
the strategy's own vol. That is a level effect, and the question is a delta.

Run: python3 scripts/magic_vol_prior_universe_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.universe import (european_eur_tickers,  # noqa: E402
                           european_non_eur_tickers, sp500_tickers)

START, END = "2015-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
TOP_N = 30            # cfg.top_n
N_DRAWS = 400         # random books per arm
US_SAMPLE = 220       # sampling the pool is statistically identical for a vol/correlation study
EU_SAMPLE = 200
NEU_SAMPLE = 160
SEED = 7

# Quote currency -> yfinance FX pair. Only the RATE matters (scale cancels in returns), so GBp
# and GBP share a pair here; that is safe for returns and would be a 100x bug for market cap.
FX_PAIR = {".L": "GBPUSD=X", ".SW": "CHFUSD=X", ".ST": "SEKUSD=X",
           ".CO": "DKKUSD=X", ".OL": "NOKUSD=X"}
EUR_PAIR = "EURUSD=X"
_EURO_SFX = (".DE", ".PA", ".AS", ".MC", ".MI", ".BR", ".HE", ".IR", ".LS")


def _fx_series(cal: pd.DatetimeIndex) -> dict[str, pd.Series]:
    pairs = sorted({EUR_PAIR, *FX_PAIR.values()})
    raw = yf.download(pairs, start=START, end=END, auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return {p: px[p].reindex(cal).ffill().bfill() for p in pairs if p in px.columns}


def _to_usd(px: pd.DataFrame, fx: dict[str, pd.Series]) -> pd.DataFrame:
    """Local-currency prices -> USD, so each name carries its own unhedged FX move."""
    out = px.copy()
    for t in px.columns:
        if "." not in t:
            continue                                   # US, already USD
        sfx = "." + t.rsplit(".", 1)[1]
        pair = FX_PAIR.get(sfx, EUR_PAIR if sfx in _EURO_SFX else None)
        if pair and pair in fx:
            out[t] = px[t] * fx[pair]
    return out


def _book_vols(rets: pd.DataFrame, pool: list[str], rng: np.random.Generator) -> np.ndarray:
    """Annualised vol of N_DRAWS random equal-weight TOP_N books, monthly rebalanced.

    Monthly rebalancing back to equal weight matches the live sleeve (rebalance "ME",
    inv_vol_clip pinned to (1.0, 1.0) since 2026-08-28). Because weights are reset each month
    and never drift far, the daily book return is well approximated by the cross-sectional mean
    of its members' returns -- which is what is computed here.
    """
    pool = [t for t in pool if t in rets.columns]
    out = []
    for _ in range(N_DRAWS):
        names = list(rng.choice(pool, size=min(TOP_N, len(pool)), replace=False))
        r = rets[names].mean(axis=1)
        r = r[r.notna()]
        if len(r) > 500:
            out.append(float(r.std() * np.sqrt(252)))
    return np.array(out)


def main() -> None:
    rng = np.random.default_rng(SEED)
    us_all = sp500_tickers()
    eu_all = european_eur_tickers()
    neu_all = european_non_eur_tickers()

    us = list(rng.choice(us_all, size=min(US_SAMPLE, len(us_all)), replace=False))
    eu = list(rng.choice(eu_all, size=min(EU_SAMPLE, len(eu_all)), replace=False))
    neu = list(rng.choice(neu_all, size=min(NEU_SAMPLE, len(neu_all)), replace=False))

    print(f"[load] {len(us)} US + {len(eu)} eurozone + {len(neu)} non-eurozone, {START}..{END}")
    raw = yf.download(us + eu + neu, start=START, end=END, auto_adjust=True,
                      progress=False, threads=True)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px.dropna(how="all", axis=1).ffill()
    # Require a real history, else a recent listing contributes a short, noisy vol estimate.
    px = px.loc[:, px.notna().sum() >= 1000]

    fx = _fx_series(px.index)
    usd = _to_usd(px, fx)
    rets = usd.pct_change(fill_method=None)
    rets = rets.where(rets.abs() < 0.5)                 # drop split/data artefacts

    us = [t for t in us if t in rets.columns]
    eu = [t for t in eu if t in rets.columns]
    neu = [t for t in neu if t in rets.columns]
    print(f"[ok]   usable: {len(us)} US, {len(eu)} eurozone, {len(neu)} non-eurozone "
          f"over {len(rets)} days\n")

    arms = {
        "US only  (what 19.0% describes)": us,
        "US + eurozone  (pre-2026-08-28)": us + eu,
        "US + ALL Europe  (now)": us + eu + neu,
        "Europe only (reference)": eu + neu,
    }

    print("=" * 96)
    print(f"ANNUALISED VOL OF A RANDOM EQUAL-WEIGHT {TOP_N}-NAME BOOK, USD returns "
          f"({N_DRAWS} draws/arm)")
    print("=" * 96)
    print(f"  {'arm':34s} {'names':>6s} {'mean':>7s} {'median':>7s} {'p10':>7s} {'p90':>7s} "
          f"{'vs US':>8s}")
    res = {}
    for label, pool in arms.items():
        v = _book_vols(rets, pool, np.random.default_rng(SEED))
        res[label] = v
        base = res.get("US only  (what 19.0% describes)")
        d = f"{np.mean(v) - np.mean(base):+7.2%}" if base is not None and label != list(arms)[0] \
            else "   —"
        print(f"  {label:34s} {len(pool):6d} {np.mean(v):7.2%} {np.median(v):7.2%} "
              f"{np.percentile(v, 10):7.2%} {np.percentile(v, 90):7.2%} {d:>8s}")

    us_v = res["US only  (what 19.0% describes)"]
    now_v = res["US + ALL Europe  (now)"]
    delta = float(np.mean(now_v) - np.mean(us_v))

    # Paired by draw index: both arms use the same seed, so the same draw number is a matched
    # comparison rather than two independent samples.
    n = min(len(us_v), len(now_v))
    d = now_v[:n] - us_v[:n]
    t = d.mean() / d.std() * np.sqrt(n) if d.std() > 0 else 0.0

    print("\n" + "=" * 96)
    print("WHAT THIS IMPLIES FOR VOL_PRIOR")
    print("=" * 96)
    print(f"  paired delta (all-Europe minus US-only):  {delta:+.2%}  t={t:+.1f}")
    print(f"  relative change:                          {delta / np.mean(us_v):+.1%}")
    print(f"\n  current VOL_PRIOR                         19.00%")
    print(f"  scaled by the measured relative change    {0.190 * (1 + delta / np.mean(us_v)):6.2%}")

    from risk_guard import BreakerLevels  # noqa: E402
    for v, lab in ((0.190, "as set"), (0.190 * (1 + delta / np.mean(us_v)), "implied")):
        lv = BreakerLevels.from_vol(v)
        print(f"    vol {v:6.2%} ({lab:7s}) -> derisk {lv.derisk:6.2%}  "
              f"reduce_only {lv.reduce_only:6.2%}  halt {lv.halt:6.2%}")

    print("\n  NB the LEVEL of the US arm is the validity check, not a result: if it is far from")
    print("  19.0% the method is mismeasuring and the deltas should not be trusted. A random")
    print("  draw has no value/quality tilt, so it is expected to sit somewhat ABOVE the")
    print("  strategy's own vol.")

    out = ROOT / "results" / "vol_prior"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: pd.Series(v) for k, v in res.items()}).to_csv(out / "book_vols.csv",
                                                                  index=False)
    print(f"\n  wrote {out}/book_vols.csv")


if __name__ == "__main__":
    main()
