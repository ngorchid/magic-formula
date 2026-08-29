"""Out-of-sample test of the put hedge on 1990-2013, WITHOUT buying option data.

THE PROBLEM. Every put-hedge figure in Appendix D rests on 2013-04 to 2026-08, because that is
where the OPRA data starts -- and Databento sells no earlier OPRA history, so the obvious fix is
unavailable. That sample contains exactly ONE fast crash (COVID) and no sustained high-vol
crisis, and 94% of its rolls expire worthless with a single December 2019 put supplying most of
the payoff. It is precisely the situation that killed the vol-spike hedge, which looked
excellent on 2012-2026 and failed on 1993-2011.

THE METHOD. Price synthetic puts from VIX (available since 1990) and validate the pricer against
the real OPRA quotes where they overlap.

The essential detail is SKEW. Pricing off raw VIX values a 1.5-sigma put at **24% of its true
cost** -- skew is most of the premium, not a correction to it. Backing implied vol out of the
real 2013-2026 quotes gives a stable, fittable relationship:

    IV / VIX = 1.555 - 0.717 x VIX      (n=54, residual sd 0.085)

i.e. the skew ratio is ~1.49 at VIX 15 and ~1.35 at VIX 20, flattening as vol rises, which is
the documented shape. Applying it to 1990-2013 gives a genuine out-of-sample test.

⚠ FOUR LIMITS, none of which changes the direction of the result:
  * The pricer reproduces COST well (2.65% synthetic vs 2.88% real, -8%) but understates PAYOFF
    (1.54% vs 2.30%, -33%), because fixed 63-day rolls do not align with real expiries and the
    payoff is dominated by a handful of events whose timing matters. Correcting the OOS payoff
    upward by the same 33% moves net from -2.58% to -2.49% -- immaterial.
  * The skew model is fitted on 2013-2026 and assumed stable back to 1990. This is the largest
    assumption; index skew steepened after 1987 and again after 2008.
  * No bid-ask or commission, so every figure FLATTERS the hedge.
  * VIX itself is backfilled before 2003 on the current methodology.

Run: python3 scripts/put_hedge_oos_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKEW_A, SKEW_B = 1.555, -0.717      # IV/VIX = A + B*VIX, fitted on real OPRA 2013-2026
HOLD = 63                            # trading days ~ one quarter
PERIODS = [("1990-01-01", "1999-12-31", "1990-1999"),
           ("2000-01-01", "2002-12-31", "2000-02 dotcom"),
           ("2003-01-01", "2006-12-31", "2003-06 calm"),
           ("2007-01-01", "2009-12-31", "2007-09 GFC"),
           ("2010-01-01", "2013-03-31", "2010-2013"),
           ("1990-01-01", "2013-03-31", "FULL OOS"),
           ("2013-04-01", "2026-08-01", "2013-26 in-sample")]


def bs_put(S: float, K: float, T: float, sig: float, r: float = 0.02) -> float:
    if T <= 0 or sig <= 0:
        return max(0.0, K - S)
    d1 = (np.log(S / K) + (r + sig * sig / 2) * T) / (sig * np.sqrt(T))
    return K * np.exp(-r * T) * norm.cdf(-(d1 - sig * np.sqrt(T))) - S * norm.cdf(-d1)


def run(spx: pd.Series, vix: pd.Series, s_: str, e_: str, mode: str, par: float) -> tuple:
    idx = spx.loc[s_:e_].index
    costs, pays, otm = [], [], []
    i = 0
    while i < len(idx) - 1:
        t = idx[i]
        S, v = float(spx.loc[t]), float(vix.loc[t]) / 100
        j = min(i + HOLD, len(idx) - 1)
        T = (idx[j] - t).days / 365.0
        if T <= 0:
            break
        K = S * (1 - par * v * np.sqrt(T)) if mode == "sigma" else S * (1 - par)
        iv = v * max(SKEW_A + SKEW_B * v, 1.0)
        costs.append(bs_put(S, K, T, iv) / S)
        pays.append(max(0.0, K - float(spx.loc[idx[j]])) / S)
        otm.append(1 - K / S)
        i = j
    y = (idx[-1] - idx[0]).days / 365.25
    return sum(costs) / y, sum(pays) / y, (sum(pays) - sum(costs)) / y, float(np.mean(otm))


def main() -> None:
    spx = yf.download("^GSPC", start="1990-01-01", end="2026-09-01",
                      auto_adjust=False, progress=False)["Close"]
    vix = yf.download("^VIX", start="1990-01-01", end="2026-09-01",
                      auto_adjust=False, progress=False)["Close"]
    if isinstance(spx, pd.DataFrame):
        spx = spx.iloc[:, 0]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix = vix.reindex(spx.index).ffill()

    print("=" * 100)
    print("PUT HEDGE OUT OF SAMPLE — synthetic prices, skew calibrated on real OPRA quotes")
    print("=" * 100)
    print(f"  skew model  IV/VIX = {SKEW_A:.3f} {SKEW_B:+.3f} x VIX\n")
    print(f"  {'period':>20s} | {'1.5 SIGMA-SCALED':^33s} | {'FIXED 10% OTM':^33s}")
    print(f"  {'':>20s} | {'cost':>7s} {'payoff':>7s} {'net':>8s} {'avgOTM':>7s} | "
          f"{'cost':>7s} {'payoff':>7s} {'net':>8s} {'avgOTM':>7s}")
    for s_, e_, lbl in PERIODS:
        a1, b1, n1, m1 = run(spx, vix, s_, e_, "sigma", 1.5)
        a2, b2, n2, m2 = run(spx, vix, s_, e_, "pct", 0.10)
        print(f"  {lbl:>20s} | {a1:7.2%} {b1:7.2%} {n1:+8.2%} {m1:7.1%} | "
              f"{a2:7.2%} {b2:7.2%} {n2:+8.2%} {m2:7.1%}")

    print("\n" + "=" * 100)
    print("WHAT THIS SAYS")
    print("=" * 100)
    print("  1. The hedge costs ~4.5x MORE out of sample: net -2.58%/yr on 1990-2013 against")
    print("     -0.57%/yr from the real 2013-2026 quotes. The favourable in-sample figure was")
    print("     one December 2019 put, not an expectation.")
    print("  2. VOL-SCALING PAYS NOTHING IN A SUSTAINED CRISIS. Through 2007-09 the 1.5-sigma")
    print("     hedge paid 0.00% while costing 2.88%/yr: with VIX at 40-60 the strike sits")
    print("     ~19-30% below spot, and the market rarely falls that far WITHIN ONE QUARTER even")
    print("     in a 57% peak-to-trough collapse. The strike runs away exactly when it is needed")
    print("     close. For a depth-first objective that is disqualifying.")
    print("  3. Vol-scaling still beats fixed-% out of sample (-2.58% vs -5.18%), so that")
    print("     ranking survives — but fixed-% is the one that actually paid in the GFC (2.72%),")
    print("     at a net cost of -8.28%/yr. Neither is attractive.")
    print("\n  Same shape of failure as the vol-spike hedge: excellent on the recent sample,")
    print("  which contains only FAST crashes, and poor once slow grinds are included.")


if __name__ == "__main__":
    main()
