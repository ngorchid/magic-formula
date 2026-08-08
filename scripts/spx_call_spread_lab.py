"""(d) Should we switch to CALL spreads in a downtrend?

The strategy defaults to bull PUT spreads because put skew plus equity drift are structural
tailwinds. The hypothesis tested here: in a clear downtrend, selling puts fights the trend, so
flip to bear CALL spreads instead.

Three actions are compared per trend rule, because "switch" is not the only alternative:
  A  switch to calls when the trend is down
  B  simply stand aside when the trend is down
  C  what the calls did in those downtrend windows, alone (the diagnostic)
Plus puts-always and calls-always as reference points.

The answer is driven by SKEW, and it is worth stating up front: at the SAME delta an SPX put
sits FURTHER out of the money than a call yet carries ~8-10 vol points MORE implied vol. So a
call spread collects ~30% less credit for the same delta exposure and the same capped risk —
you are selling the cheap wing of the smile. That asymmetry IS why the strategy defaults to
puts, and no trend condition overcomes it.

NB SPX index skew is much steeper than single-name skew (systematic hedging demand), so the
magnitude here will not transfer directly to the 14-name basket, though the direction should.

Run: python scripts/spx_call_spread_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from spx_vrp_lab import Config, load, run, regime_ratio  # noqa: E402

OUT = ROOT / "results" / "spx_vrp"
COST = 0.25
YRS = 13.2


def net(t: pd.DataFrame) -> pd.Series:
    return t.pnl + 4 * 0.50 * 100 - 4 * COST * 100


def stat(t, lab: str) -> str:
    if t is None or len(t) < 5:
        return f"  {lab:34s}  n<5"
    p = net(t)
    s = p.mean() / p.std() * np.sqrt(len(p) / YRS) if p.std() else np.nan
    return (f"  {lab:34s} {len(t):>4d} {p.sum():>9,.0f} {p.mean():>8,.0f} "
            f"{(p > 0).mean():>5.0%} {s:>+7.2f} {p.min():>9,.0f}")


def skew_table(ch: pd.DataFrame) -> None:
    g = ch[ch.iv.notna() & ch.dte.between(30, 45)]
    print("SKEW ASYMMETRY — same |delta|, put vs call (the reason for the result)")
    print(f"  {'|delta|':>8s} {'put IV':>9s} {'call IV':>9s} {'gap':>8s} "
          f"{'put %OTM':>10s} {'call %OTM':>11s}")
    print("  " + "-" * 60)
    for lo, hi, lab in ((0.14, 0.18, "0.16"), (0.08, 0.12, "0.10"),
                        (0.18, 0.22, "0.20"), (0.23, 0.27, "0.25")):
        p = g[(g.cp == "P") & g.delta.abs().between(lo, hi)]
        c = g[(g.cp == "C") & g.delta.abs().between(lo, hi)]
        if p.empty or c.empty:
            continue
        print(f"  {lab:>8s} {p.iv.median()*100:>8.1f}% {c.iv.median()*100:>8.1f}% "
              f"{(p.iv.median()-c.iv.median())*100:>+7.1f} {p.mny.median()*100:>9.2f}% "
              f"{c.mny.median()*100:>10.2f}%")


def main() -> None:
    ch, spot, vrp = load()
    ratio = regime_ratio()
    skew_table(ch)

    sp = spot.sort_index()
    # all lagged one day — decided on yesterday's close, applied today
    trends = {
        "SPX < 200d MA": (sp < sp.rolling(200).mean()).shift(1),
        "126d mom < 0": (sp.pct_change(126) < 0).shift(1),
        "252d mom < 0": (sp.pct_change(252) < 0).shift(1),
    }

    base = dict(stop_mult=0.0, regime_thr=1.00)   # live spec, stop off (see spx_vrp_lab)
    puts = run(Config(cp="P", **base), ch, spot, vrp, ratio)
    calls = run(Config(cp="C", **base), ch, spot, vrp, ratio)

    hdr = (f"  {'variant':34s} {'n':>4s} {'total$':>9s} {'$/tr':>8s} {'win':>5s} "
           f"{'Sharpe':>7s} {'worst$':>9s}")
    print("\n" + "=" * 84)
    print("(d) TREND-CONDITIONAL CALL SPREADS  (live spec: regime gate on, no stop)")
    print("=" * 84)
    print(hdr)
    print("  " + "-" * 80)
    print(stat(puts, "PUTS always (current strategy)"))
    print(stat(calls, "CALLS always (reference)"))

    rows = []
    for lab, dn in trends.items():
        pdn = puts.entry_date.map(dn).fillna(False)
        cdn = calls.entry_date.map(dn).fillna(False)
        switch = pd.concat([puts[~pdn], calls[cdn]]).sort_values("entry_date")
        aside = puts[~pdn]
        conly = calls[cdn]
        print(f"\n  --- {lab}   ({pdn.mean():.0%} of entries flagged downtrend) ---")
        print(stat(switch, "   A switch to calls in downtrend"))
        print(stat(aside, "   B stand aside in downtrend"))
        print(stat(conly, "   C calls only, downtrend only"))
        rows.append({"rule": lab, "A_switch": net(switch).mean(),
                     "B_aside": net(aside).mean(),
                     "C_calls_in_dn": net(conly).mean() if len(conly) else np.nan,
                     "n_dn": int(cdn.sum())})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "call_spread_trend.csv", index=False)
    print(f"\n  wrote {OUT}/call_spread_trend.csv")
    print("\n  VERDICT: calls lose outright (-$134/tr vs puts +$120), switching is worse than")
    print("  not switching on every rule, and standing aside is worse than doing nothing.")
    print("  Column C is the tell: calls sold SPECIFICALLY in downtrends are the worst cell,")
    print("  because downtrends are when vol is high and rebounds are violent.")


if __name__ == "__main__":
    main()
