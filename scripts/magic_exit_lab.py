"""WHY did a name drop out of the band — and does the reason predict what it does next?

THE IDEA. The band sweep showed that holding dropped names longer does not pay ON AVERAGE. But
"dropped out of the top 45" mixes two very different events, and averaging them together could
easily hide an exploitable split:

  PRICE-DRIVEN   the name got EXPENSIVE because it went UP. Its FCF/EV yield fell while the
                 business is unchanged. This is a winner being sold by construction -- the
                 valuation ranking mechanically penalises a risen price.
  FUNDAMENTAL    the BUSINESS deteriorated. Return on capital fell. Nothing to do with price.

The live rank splits these exactly, with no modelling required:
  fcf_ev_yield          = FCF / (mcap + debt - cash)   -> CONTAINS PRICE
  fcf_return_on_capital = FCF / (NWC + net PP&E)       -> NO PRICE AT ALL

So a drop where ROC held but EV-yield fell is almost purely a price event; a drop where ROC fell
is a business event. If the first group outperforms the second AFTER being sold, the exit rule is
throwing away winners and a conditional hold is worth building.

⚠ MEASURED AS EXCESS OVER THE BOOK, not absolute. Holding a dropped name means NOT buying its
replacement, so the decision-relevant quantity is how the dropped name does relative to the
equal-weighted book it would have been swapped into. An absolute +8% means nothing if the book
made +10%.

⚠ NO LOOKAHEAD: the classification uses only the trailing 252d change in each component's
cross-sectional percentile rank, all of it known at the drop date.

Run: python3 scripts/magic_exit_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from signals.quality import fcf_ev_yield, fcf_return_on_capital   # noqa: E402
from strategies.magic_formula import (EnhancedMagicConfig,        # noqa: E402
                                      enhanced_weights)
from run_best_magic import _load                                   # noqa: E402

HORIZONS = (63, 126, 252)      # ~3m, 6m, 12m
LOOK = 252                     # trailing window for the component-rank change


def main() -> None:
    cfg = EnhancedMagicConfig(use_graham=False)      # LIVE config
    print("[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load(
        "sp500_pit", cfg, "2012-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    w, rank = enhanced_weights(f, mcap, adj, base, cfg)

    # Component percentile ranks (higher = more attractive), on the same calendar.
    ey = fcf_ev_yield(f, mcap).reindex_like(adj).rank(axis=1, pct=True)
    roc = fcf_return_on_capital(f).reindex_like(adj).rank(axis=1, pct=True)
    d_ey = ey - ey.shift(LOOK)
    d_roc = roc - roc.shift(LOOK)

    fwd = {h: (adj.shift(-h) / adj - 1.0) for h in HORIZONS}
    book = {h: pd.Series({d: (fwd[h].loc[d][w.loc[d] > 0]).mean() for d in w.index})
            for h in HORIZONS}                        # equal-weight book return, per date

    # A DROP: held on the previous bar, not held now.
    held_prev, held_now = (w.shift(1) > 0), (w > 0)
    drops = held_prev & ~held_now

    recs = []
    for d in w.index[w.index > adj.index[LOOK]]:
        names = drops.columns[drops.loc[d].values]
        for t in names:
            de, dr = d_ey.at[d, t], d_roc.at[d, t]
            if not (np.isfinite(de) and np.isfinite(dr)):
                continue
            rec = {"date": d, "ticker": t, "d_ey": de, "d_roc": dr}
            for h in HORIZONS:
                v, b = fwd[h].at[d, t], book[h].get(d, np.nan)
                rec[f"ex{h}"] = v - b if np.isfinite(v) and np.isfinite(b) else np.nan
            recs.append(rec)
    ev = pd.DataFrame(recs)
    print(f"  {len(ev):,} drop events, {ev['date'].nunique()} dates\n")

    # Classification. Thresholds are round numbers in PERCENTILE-RANK units, deliberately not
    # tuned -- a rule that needs a tuned threshold to work on this many events is fitted.
    def bucket(r):
        roc_fell = r["d_roc"] <= -0.10
        ey_fell = r["d_ey"] <= -0.10
        if roc_fell and not ey_fell:
            return "ROC fell only (business)"
        if ey_fell and not roc_fell:
            return "EY fell only (price)"
        if ey_fell and roc_fell:
            return "both fell"
        return "neither fell much"

    ev["bucket"] = ev.apply(bucket, axis=1)
    print("=" * 92)
    print(f"FORWARD EXCESS RETURN OF A DROPPED NAME vs THE BOOK — {label}, band {cfg.hold_n}")
    print("=" * 92)
    print(f"  {'why it dropped':30}{'n':>7}" + "".join(f"{'ex' + str(h) + 'd':>12}" for h in HORIZONS)
          + f"{'t by-date':>10}{'t annual':>10}")
    print("  " + "-" * 88)
    order = ["EY fell only (price)", "ROC fell only (business)", "both fell", "neither fell much"]
    for b in order:
        g = ev[ev["bucket"] == b]
        if g.empty:
            continue
        # CLUSTERED BY DATE, not by event. The naive event-level t treats 617 drops as 617
        # independent draws; they sit on 155 dates (same-date drops share a book return) with
        # 252-day windows that overlap heavily -- over 13.4 years there are only ~13 truly
        # independent annual windows. The naive t roughly DOUBLES every number here.
        bd = g.groupby("date")["ex252"].mean().dropna()
        t = bd.mean() / bd.std() * np.sqrt(len(bd)) if len(bd) > 2 and bd.std() else np.nan
        yr = g.groupby(g["date"].dt.year)["ex252"].mean().dropna()
        ty = yr.mean() / yr.std() * np.sqrt(len(yr)) if len(yr) > 2 and yr.std() else np.nan
        print(f"  {b:30}{len(g):>7,}" + "".join(f"{g['ex' + str(h)].mean():>+12.2%}" for h in HORIZONS)
              + f"{t:>+10.1f}{ty:>+10.1f}")
    _bd = ev.groupby("date")["ex252"].mean().dropna()
    _yr = ev.groupby(ev["date"].dt.year)["ex252"].mean().dropna()
    print(f"  {'ALL drops':30}{len(ev):>7,}"
          + "".join(f"{ev['ex' + str(h)].mean():>+12.2%}" for h in HORIZONS)
          + f"{_bd.mean() / _bd.std() * np.sqrt(len(_bd)):>+10.1f}"
          + f"{_yr.mean() / _yr.std() * np.sqrt(len(_yr)):>+10.1f}")
    print("\n  ⚠ NOTHING in this table is significant once clustering is respected. The point")
    print("     estimates are suggestive; the evidence is not there. Do not act on a row alone.")

    a = ev[ev["bucket"] == "EY fell only (price)"]["ex252"].dropna()
    b_ = ev[ev["bucket"] == "ROC fell only (business)"]["ex252"].dropna()
    if len(a) > 2 and len(b_) > 2:
        diff = a.mean() - b_.mean()
        # NB still event-level and so still OPTIMISTIC; it was insignificant even before
        # clustering, which is why it is left as the weaker test rather than sharpened.
        se = np.sqrt(a.var() / len(a) + b_.var() / len(b_))
        print(f"\n  PRICE-driven minus BUSINESS-driven, 252d: {diff:+.2%}  "
              f"(t = {diff / se:+.1f} on n={len(a):,} vs {len(b_):,})")
        print("  This is the whole question: a positive, significant gap means the exit rule is")
        print("  discarding winners and a conditional hold is worth building. Near zero means the")
        print("  band is right to sell both, and the reason it dropped carries no extra signal.")
    ev.to_csv(ROOT / "results" / "magic_exit_events.csv", index=False)


if __name__ == "__main__":
    main()
