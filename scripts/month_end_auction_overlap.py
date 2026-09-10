"""Is the month-end 5y effect really the 5Y/7Y auction snapback, counted twice?

`month_end_extension_lab.py` measures a buy-duration-into-month-end effect strongest at the
5y (t=+4.13). But Treasury schedules the 2Y/5Y/7Y auctions in the LAST WEEK of the month, so
the month-end window T-3 -> T0 can sit directly on top of the post-auction snapback that
`treasury_auction_lab.py` already trades. If so, the two clocks are not independent and
adding Clock 2 to the sleeve's P&L estimate double-counts the same basis points.

This is a bias-of-the-MEAN question, not a width-of-the-interval question, so it matters more
than the block-bootstrap caveat: overlapping events inflate the estimate itself.

The test. For each month-end event, flag whether a NEW-ISSUE 5Y or 7Y auction falls inside
the measurement window (or inside its own T-1..T+3 snapback window, which is the leg the
auction sleeve actually holds). Then split:

    contaminated   auction snapback overlaps the month-end window
    clean          no overlap -- pure extension flow

If the clean subset keeps the effect, the two clocks are separable. If the effect lives only
in the contaminated subset, the month-end 5y is Clock 1 wearing a different hat, and Clock 2
should be run on the 10y/30y only, where auctions fall mid-month.

Usage:  python scripts/month_end_auction_overlap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from month_end_extension_lab import (  # noqa: E402
    build_measures, load_yields, month_end_dates,
)
from treasury_auction_lab import fetch_auctions, parse_tenor  # noqa: E402

ENTRY = 3          # month-end window: T-3 -> T0, matching the lab
SNAP_PRE, SNAP_POST = 1, 3   # the auction sleeve holds T-1 -> T+3


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb)) if va + vb > 0 else np.nan


def main() -> None:
    bp = load_yields()
    measures = build_measures(bp)
    idx = bp.dropna(subset=[10]).index
    me_dates = month_end_dates(idx)

    auc = fetch_auctions()
    auc["tenor"] = auc["securityTerm"].map(parse_tenor)
    auc = auc.dropna(subset=["tenor"])
    new_issue = auc[auc["reopening"].astype(str).str.upper().isin(("NO", "FALSE", "N"))]

    # Auction dates whose SNAPBACK window the auction sleeve would be holding
    short_end = new_issue[new_issue["tenor"].isin([2.0, 5.0, 7.0])]["auctionDate"]
    short_end = pd.DatetimeIndex(sorted(set(short_end)))

    pos = {d: i for i, d in enumerate(idx)}

    rows = []
    for d in me_dates:
        if d not in pos:
            continue
        i = pos[d]
        if i - ENTRY < 0:
            continue
        win_start, win_end = idx[i - ENTRY], idx[i]

        # Does any 2/5/7y new-issue auction's snapback window overlap this month-end window?
        overlap = False
        for a in short_end[(short_end >= win_start - pd.Timedelta(days=14))
                           & (short_end <= win_end + pd.Timedelta(days=3))]:
            if a not in pos:
                nearest = idx[idx.get_indexer([a], method="nearest")[0]]
                a = nearest
            j = pos[a]
            s0 = idx[max(j - SNAP_PRE, 0)]
            s1 = idx[min(j + SNAP_POST, len(idx) - 1)]
            if s0 <= win_end and s1 >= win_start:      # intervals intersect
                overlap = True
                break

        row = {"date": d, "overlap": overlap}
        for name, s in measures.items():
            if win_start in s.index and win_end in s.index:
                row[name] = float(s.loc[win_end] - s.loc[win_start])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    n_ov = int(df["overlap"].sum())
    print("=" * 92)
    print("DO THE TWO CLOCKS OVERLAP?  month-end window T-3 -> T0 vs 2/5/7y auction snapback")
    print("=" * 92)
    print(f"  month-end events: {len(df)}   contaminated by a 2/5/7y new-issue snapback: "
          f"{n_ov} ({n_ov / len(df):.0%})")
    print("  (positive = the hypothesised move: yields FALL into month-end)\n")

    focus = [c for c in ("outright 5y", "outright 10y", "outright 30y",
                         "slope 5s30s", "fly 2s10s30s") if c in df.columns]
    print(f"{'measure':16}{'ALL':>22}{'CLEAN (no auction)':>26}{'CONTAMINATED':>24}"
          f"{'diff t':>9}")
    print(f"{'':16}{'n   mean      t':>22}{'n   mean      t':>26}{'n   mean      t':>24}")
    print("-" * 92)
    for c in focus:
        s = df[c].dropna()
        cl = df.loc[~df["overlap"], c].dropna()
        ct = df.loc[df["overlap"], c].dropna()

        def blk(x):
            if len(x) < 5:
                return f"{len(x):>4} {'—':>8} {'—':>7}"
            t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
            return f"{len(x):>4} {x.mean():>+8.3f} {t:>+7.2f}"

        print(f"{c:16}{blk(s):>22}{blk(cl):>26}{blk(ct):>24}"
              f"{welch_t(cl.values, ct.values):>+9.2f}")

    print("\n  'diff t' tests CLEAN minus CONTAMINATED. Near zero = the two subsets agree,")
    print("  i.e. the month-end effect is NOT just the auction snapback in disguise.")
    print("  Strongly negative = the effect lives in the contaminated events = double counting.")

    out = ROOT / "results" / "month_end"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "overlap_split.csv")
    print(f"\nWrote {out / 'overlap_split.csv'}")


if __name__ == "__main__":
    main()
