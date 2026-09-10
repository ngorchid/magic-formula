"""Is there a month-end extension effect that the auction sleeve does NOT already own?

`month_end_auction_overlap.py` split month-end events by TIME -- does a 2/5/7y snapback window
overlap the month-end window -- and the effect vanished in the clean subset. But that test has
a gap. The index extension is driven by EVERYTHING that settled in the month, including the
mid-month 10y/30y refunding issues, which the auction sleeve trades on a different date (or,
for the 30y, not at all). So a month could be "contaminated" on the 5y clock while its
extension flow is really about the long end -- a genuinely separate trade.

The decisive question is therefore not "did an auction overlap" but:

    IN THE CLEAN MONTHS, WHERE THE SLEEVE HOLDS NOTHING, DOES A BIG EXTENSION STILL MOVE
    YIELDS?

Index extension happens EVERY month-end regardless of auction timing, because issuance settles
monthly. So if the effect is real and separable, clean months with LARGE duration-weighted
issuance must show it. If even those are flat, there is no extension effect to harvest and the
whole month-end result was the auction snapback.

Two cuts:
  * issuance tercile   -- duration-weighted coupon issuance settling that month
  * refunding month    -- did a NEW-ISSUE 10y or 30y settle? (Feb/May/Aug/Nov quarterly
                          refunding, the months with the biggest extension by construction)

Usage:  python scripts/month_end_tenor_split.py
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
    build_measures, issuance_proxy, load_yields, month_end_dates,
)
from treasury_auction_lab import fetch_auctions, parse_tenor  # noqa: E402

ENTRY = 3
SNAP_PRE, SNAP_POST = 1, 3


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 5 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def cell(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 5:
        return f"{len(x):>4} {'—':>9} {'—':>7}"
    return f"{len(x):>4} {x.mean():>+9.3f} {tstat(x):>+7.2f}"


def main() -> None:
    bp = load_yields()
    measures = build_measures(bp)
    idx = bp.dropna(subset=[10]).index
    me_dates = month_end_dates(idx)
    pos = {d: i for i, d in enumerate(idx)}

    auc = fetch_auctions()
    auc["tenor"] = auc["securityTerm"].map(parse_tenor)
    auc = auc.dropna(subset=["tenor"])
    is_new = auc["reopening"].astype(str).str.upper().isin(("NO", "FALSE", "N"))
    new_issue = auc[is_new]

    short_end = pd.DatetimeIndex(sorted(set(
        new_issue[new_issue["tenor"].isin([2.0, 5.0, 7.0])]["auctionDate"])))
    # Months in which a NEW-ISSUE long bond SETTLED -> the big-extension months
    long_settle = new_issue[new_issue["tenor"].isin([10.0, 30.0])]
    long_months = set(pd.to_datetime(long_settle["issueDate"]).dt.to_period("M"))

    iss = issuance_proxy()

    rows = []
    for d in me_dates:
        if d not in pos or pos[d] - ENTRY < 0:
            continue
        i = pos[d]
        w0, w1 = idx[i - ENTRY], idx[i]

        overlap = False
        for a in short_end[(short_end >= w0 - pd.Timedelta(days=20))
                           & (short_end <= w1 + pd.Timedelta(days=8))]:
            if a not in pos:
                a = idx[idx.get_indexer([a], method="nearest")[0]]
            j = pos[a]
            if idx[max(j - SNAP_PRE, 0)] <= w1 and idx[min(j + SNAP_POST, len(idx) - 1)] >= w0:
                overlap = True
                break

        per = pd.Period(d, freq="M")
        row = {"date": d, "overlap": overlap,
               "refunding": per in long_months,
               "issuance": float(iss.get(per, np.nan))}
        for name, s in measures.items():
            if w0 in s.index and w1 in s.index:
                row[name] = float(s.loc[w1] - s.loc[w0])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    ok = df["issuance"].notna()
    df.loc[ok, "iss_hi"] = df.loc[ok, "issuance"] >= df.loc[ok, "issuance"].median()

    clean = df[~df["overlap"]]
    print("=" * 96)
    print("IS THERE AN EXTENSION EFFECT THE AUCTION SLEEVE DOES NOT ALREADY OWN?")
    print("=" * 96)
    print(f"  month-end events {len(df)};  clean of a 2/5/7y snapback: {len(clean)}")
    print("  Extension happens EVERY month-end, so if it is real the CLEAN + BIG-EXTENSION")
    print("  cell must show it. (positive = yields fall into month-end, the hypothesis)\n")

    for m in ("outright 5y", "outright 10y", "outright 30y"):
        if m not in df.columns:
            continue
        print(f"  --- {m} ---")
        print(f"{'':22}{'n      mean       t':>22}{'n      mean       t':>24}")
        print(f"{'':22}{'CLEAN':>22}{'CONTAMINATED':>24}")
        for lbl, mask in (("all", pd.Series(True, index=df.index)),
                          ("big extension", df["iss_hi"] == True),        # noqa: E712
                          ("small extension", df["iss_hi"] == False),     # noqa: E712
                          ("refunding month", df["refunding"] == True),   # noqa: E712
                          ("non-refunding", df["refunding"] == False)):   # noqa: E712
            c = df.loc[(~df["overlap"]) & mask, m]
            k = df.loc[(df["overlap"]) & mask, m]
            print(f"  {lbl:20}{cell(c):>22}{cell(k):>24}")
        print()

    print("  READ: if the CLEAN column is flat even where the extension is largest, then")
    print("  index extension is not separately tradeable here and the whole month-end")
    print("  result was the auction snapback. If CLEAN + big extension is positive and")
    print("  significant, there IS a second clock and it should be traded on those months.")

    out = ROOT / "results" / "month_end"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tenor_split.csv")
    print(f"\nWrote {out / 'tenor_split.csv'}")


if __name__ == "__main__":
    main()
