"""How are the book's drawdowns DISTRIBUTED -- is the max a freak, or routine?

THE QUESTION. Every risk decision so far has been argued against a single number: maxDD -32.8%
for the magic sleeve, -34.1% for magic+trend. That number is ONE realisation of a random
process. It cannot distinguish "a once-in-a-lifetime event you should not design around" from
"the third of four similar episodes, and another is due". Those imply opposite actions, and the
put-hedge analysis in Appendix D leaned on the -32.8% figure throughout without ever asking
which it was.

FOUR VIEWS, because no single statistic answers it.

1. EPISODE DECOMPOSITION. Every distinct peak-to-trough-to-recovery episode: depth, how long
   the fall took, how long recovery took. This is what "regular or freak" actually means -- if
   the second-worst episode is half the worst, the max is an outlier; if it is 90% of it, the
   max is simply the largest of a recurring class.

2. EXCEEDANCE FREQUENCY / RETURN PERIOD. How many episodes breached 10%, 15%, 20%...? Expressed
   as "once every N years", which is the form a capital decision actually needs.

3. TIME UNDER WATER, and the shape of it. maxDD says nothing about duration, yet a 20%
   drawdown lasting three years is a different problem from a 30% one recovered in four months
   -- the first is what makes people abandon a strategy.

4. BLOCK BOOTSTRAP of maxDD. The decisive one. Resampling in blocks (preserving short-horizon
   autocorrelation and volatility clustering) gives the DISTRIBUTION of maxDD this return
   process generates over a sample of this length. If the realised -32.8% sits near the median,
   it is the normal outcome and will recur. If it sits at the 95th percentile, the sample was
   unlucky and the honest planning number is lower. Either way it replaces a point estimate
   with a range.

⚠ The bootstrap destroys the true ordering of returns, so it understates the depth of slow
grinds that depend on persistent negative drift beyond one block. Treat its output as a FLOOR
on plausible maxDD, not a ceiling. Block length is swept for exactly this reason.

Run: python3 scripts/drawdown_profile_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAGIC = ROOT / "results" / "best_magic" / "best_sp500_pit_all.csv"
TREND = ROOT / "results" / "trend_overlay" / "trend_overlay_net.csv"
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
N_BOOT = 4000
SEED = 11


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1.0


def episodes(r: pd.Series, min_depth: float = 0.05) -> pd.DataFrame:
    """Distinct peak -> trough -> recovery episodes deeper than `min_depth`.

    An episode ENDS only when the prior peak is regained, so overlapping dips inside one
    unrecovered decline count once rather than as several -- otherwise a single long grind is
    reported as a series of small drawdowns and the tail looks tamer than it is.
    """
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    under = eq < peak * (1 - 1e-12)
    out, i, n = [], 0, len(eq)
    idx = eq.index
    while i < n:
        if not under.iloc[i]:
            i += 1
            continue
        j = i
        while j < n and under.iloc[j]:
            j += 1
        seg = eq.iloc[i:j]
        p = float(peak.iloc[i])
        depth = float(seg.min() / p - 1.0)
        if -depth >= min_depth:
            t = int(seg.values.argmin())
            out.append({
                "start": idx[i], "trough": seg.index[t],
                "end": idx[j] if j < n else pd.NaT,
                "depth": depth,
                "fall_m": (seg.index[t] - idx[i]).days / 30.44,
                "recover_m": ((idx[j] - seg.index[t]).days / 30.44) if j < n else np.nan,
                "total_m": ((idx[j] - idx[i]).days / 30.44) if j < n else np.nan,
            })
        i = j
    return pd.DataFrame(out).sort_values("depth")


def bootstrap_maxdd(r: pd.Series, block: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Distribution of maxDD from circular block resampling (preserves local dependence)."""
    rng = np.random.default_rng(seed)
    x = r.values
    m = len(x)
    nb = int(np.ceil(m / block))
    starts = rng.integers(0, m, size=(n, nb))
    offs = np.arange(block)
    out = np.empty(n)
    for k in range(n):
        path = x[(starts[k][:, None] + offs[None, :]).ravel() % m][:m]
        eq = np.cumprod(1 + path)
        out[k] = (eq / np.maximum.accumulate(eq)).min() - 1.0
    return out


def report(name: str, r: pd.Series) -> None:
    dd = drawdown_series(r)
    ep = episodes(r)
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    print("\n" + "=" * 100)
    print(f"{name}   ({r.index[0].date()} -> {r.index[-1].date()}, {yrs:.1f} years)")
    print("=" * 100)

    print("\n  1. EPISODES deeper than 5% (peak -> trough -> full recovery)\n")
    print(f"     {'depth':>7s} {'start':>12s} {'trough':>12s} {'recovered':>12s} "
          f"{'fall':>7s} {'recover':>8s} {'total':>7s}")
    for _, e in ep.head(8).iterrows():
        rec = e["end"].date() if pd.notna(e["end"]) else "NOT YET"
        print(f"     {e['depth']:7.1%} {e['start'].date()!s:>12s} {e['trough'].date()!s:>12s} "
              f"{rec!s:>12s} {e['fall_m']:6.1f}m {e['recover_m']:7.1f}m {e['total_m']:6.1f}m")
    if len(ep) > 8:
        print(f"     ... and {len(ep) - 8} more between 5% and {-ep.iloc[7]['depth']:.0%}")

    worst = ep["depth"].values
    print(f"\n     worst {worst[0]:.1%} | 2nd {worst[1]:.1%} | 3rd {worst[2]:.1%}"
          f"   -> 2nd is {worst[1] / worst[0]:.0%} of the worst")
    print("     (near 100% = the max is the largest of a RECURRING class, not a freak)")

    print("\n  2. EXCEEDANCE — how often is a drawdown of at least X reached?\n")
    print(f"     {'depth':>7s} {'episodes':>9s} {'once every':>12s}")
    for t in THRESHOLDS:
        c = int((ep["depth"] <= -t).sum())
        per = f"{yrs / c:.1f} yr" if c else "never"
        print(f"     {t:7.0%} {c:9d} {per:>12s}")

    uw = float((dd < -0.01).mean())
    print(f"\n  3. TIME UNDER WATER: {uw:.0%} of days more than 1% below the prior peak")
    print(f"     median episode length {ep['total_m'].median():.1f} months, "
          f"longest {ep['total_m'].max():.1f} months")
    ulcer = float(np.sqrt((dd ** 2).mean()))
    print(f"     Ulcer index {ulcer:.2%} (RMS drawdown; penalises deep AND long)")
    print(f"     Pain index  {float(-dd.mean()):.2%} (average drawdown at any moment)")
    cdar = float(dd[dd <= dd.quantile(0.05)].mean())
    print(f"     CDaR(5%)    {cdar:.1%} (mean of the worst 5% of days)")

    print("\n  4. BOOTSTRAP — what maxDD does this return process GENERATE?\n")
    print(f"     {'block':>7s} {'p50':>8s} {'p90':>8s} {'p95':>8s} {'p99':>8s} "
          f"{'realised percentile':>20s}")
    real = float(dd.min())
    for block in (5, 21, 63):
        b = bootstrap_maxdd(r, block)
        pct = float((b > real).mean())     # share of paths LESS deep than realised
        print(f"     {block:5d}d {np.percentile(b, 50):8.1%} {np.percentile(b, 10):8.1%} "
              f"{np.percentile(b, 5):8.1%} {np.percentile(b, 1):8.1%} {pct:19.0%}")
    print(f"\n     realised maxDD {real:.1%}. A percentile near 50% means the max is the")
    print("     TYPICAL outcome for this process and should be planned for, not treated")
    print("     as a freak. Near 95%+ means the sample was unlucky.")


def main() -> None:
    mf = pd.read_csv(MAGIC, index_col=0, parse_dates=True)["net_return"]
    tr = pd.read_csv(TREND, index_col=0, parse_dates=True)["trend"]
    idx = mf.index.intersection(tr.index)
    mf_a, tr_a = mf.reindex(idx).fillna(0.0), tr.reindex(idx).fillna(0.0)
    report("MAGIC FORMULA sleeve", mf_a)
    report("BOOK: magic + trend (each at 1.0x NAV, so they stack)", mf_a + tr_a)


if __name__ == "__main__":
    main()
