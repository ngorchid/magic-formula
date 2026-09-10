"""Does a WEAK auction produce a bigger snapback? Pre-committed conditioning test.

The mechanism says the snapback happens because dealers are left holding paper they did not
want and must distribute it. If that is right, the snapback should be biggest exactly when
dealers got stuffed. Three free TreasuryDirect fields measure that:

    dealer_share    primaryDealerAccepted / totalAccepted   dealers are the buyer of last
                    resort, so a high share means end demand did not show up
    bid_to_cover    total tendered / total accepted          low = weak demand
    indirect_share  indirectBidderAccepted / totalAccepted   indirect = foreign/central bank
                    real money; low = weak demand

The true tail (stop-out yield minus the 1pm when-issued) is NOT free -- TreasuryDirect does
not publish WI -- but all three fields above correlate with it, so a composite gets most of
the signal without buying quotes.

⚠ THE DESIGN CONSTRAINT THAT SHAPES THIS WHOLE SCRIPT. There are ~16 new-issue events a year.
A threshold search on that will find something whether or not it is there -- and this repo has
already been burned by exactly that: the OPRA walk-forward picked VRP>5 on the training era,
which scored +0.51 on the holdout, while the theory-chosen VRP>2 scored +1.31. The searched
winner LOST to the pre-specified one.

So the rule is fixed before looking and is not tuned afterwards:

    TRADE the top tercile of dealer_share, ranked WITHIN tenor.

and every result is reported next to the unconditional one, so the conditional LIFT is
visible rather than the conditional level alone. The other two fields are reported as
pre-specified alternatives, not as a menu to pick from.

Terciles are computed three ways; only the first is a verdict (see `flag_top_tercile`):

    rolling     ranked against the trailing 24 auctions in the same tenor -- causal AND
                detrended. THE TEST.
    full        whole-sample quantile -- NOT tradeable, shown to size the look-ahead
    expanding   ranked against all prior history -- BROKEN here, because these measures
                trend hard; kept only so the failure stays reproducible

RESULT (2026-09-10): the hypothesis is BACKWARDS. On the rolling rule, weak auctions produce
a SMALLER snapback (+0.23bp gross, -0.52 NET, t=+0.39) than strong ones (+1.52 gross, +0.77
net, t=+7.92), against an unconditional +1.56/+0.81 at t=+6.59. All three measures agree, and
the lift is negative in every era. Conditioning is REJECTED -- trade every new issue.

Usage:  python scripts/auction_conditioning_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treasury_auction_lab import (  # noqa: E402
    CACHE, Legs, build_measures, clustered_t, event_panel, fetch_yields, leg_pnl, parse_tenor,
)

COST_RT = 0.75          # bp of fly per round trip, same as the main lab
TERCILE = 2.0 / 3.0     # "top tercile" = at or above this quantile
MIN_HIST = 12           # need this many prior auctions in the tenor before ranking
ROLL = 24               # trailing window for the rolling tercile (~3y of 5y auctions)


def fetch_auction_results(start_year: int = 1980, end_year: int = 2026) -> pd.DataFrame:
    """Auction RESULTS with the bidder-composition fields the main lab's cache drops.

    Same endpoint and the same 250-row cap, so it walks year by year and type by type. Cached
    separately rather than changing the main lab's cache, so the existing sleeve's inputs are
    untouched.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"auction_results_{start_year}_{end_year}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["auctionDate"])

    url = "https://www.treasurydirect.gov/TA_WS/securities/search"
    keep = ["cusip", "securityTerm", "auctionDate", "reopening", "offeringAmount",
            "totalAccepted", "totalTendered", "bidToCoverRatio", "highYield",
            "primaryDealerAccepted", "primaryDealerTendered",
            "indirectBidderAccepted", "directBidderAccepted", "somaAccepted"]
    rows = []
    for year in range(start_year, end_year + 1):
        for sec_type in ("Note", "Bond"):
            try:
                r = requests.get(url, params={"format": "json",
                                              "startDate": f"{year}-01-01",
                                              "endDate": f"{year}-12-31",
                                              "dateFieldName": "auctionDate",
                                              "type": sec_type}, timeout=120)
                rows += r.json() if r.text.strip() else []
            except Exception as exc:  # noqa: BLE001
                print(f"  warn {year} {sec_type}: {exc!r}")
        print(f"  fetched {year} ({len(rows)} cumulative)", end="\r")

    df = pd.DataFrame(rows)
    df = df[[c for c in keep if c in df.columns]].copy()
    df["auctionDate"] = pd.to_datetime(df["auctionDate"]).dt.tz_localize(None).dt.normalize()
    for c in keep:
        if c in df.columns and c not in ("cusip", "securityTerm", "auctionDate", "reopening"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["cusip", "auctionDate"]).sort_values("auctionDate")
    df.to_csv(path, index=False)
    print(f"\n  cached {len(df)} auction results -> {path}")
    return df


def add_conditioners(a: pd.DataFrame) -> pd.DataFrame:
    """The three free weak-auction measures. SOMA (Fed rollovers) is removed from the
    denominator: it is not competitive demand, and its share swings enormously across QE
    regimes, which would otherwise make dealer_share look regime-driven rather than
    demand-driven."""
    comp = a["totalAccepted"] - a["somaAccepted"].fillna(0.0)
    a = a.copy()
    a["dealer_share"] = a["primaryDealerAccepted"] / comp
    a["indirect_share"] = a["indirectBidderAccepted"] / comp
    a["bid_to_cover"] = a["bidToCoverRatio"]
    # Signed so that HIGHER = WEAKER auction = the hypothesised bigger snapback
    a["w_dealer"] = a["dealer_share"]
    a["w_btc"] = -a["bid_to_cover"]
    a["w_indirect"] = -a["indirect_share"]
    return a


def flag_top_tercile(a: pd.DataFrame, col: str, mode: str) -> pd.Series:
    """True where `col` is in the top tercile within its tenor.

    ⚠ MODE MATTERS, AND `expanding` IS WRONG HERE. These measures are strongly TRENDING:
    dealer share (ex-SOMA) falls 48.7% -> 12.7% across 2008-2026 as electronic and foreign
    bidding displaced dealer take-down, and indirect share rises 40% -> 65% as its mirror. An
    expanding threshold is therefore set by the early era and essentially never exceeded later
    (it flagged 3 of 284 events). Worse, conditioning on the LEVEL of a trending variable is
    conditioning on the ERA -- the same trap the month-end issuance proxy fell into.

      rolling      rank against the trailing ROLL auctions in the same tenor. Causal AND
                   detrended: "was this auction weak FOR ITS TIME", which is the economically
                   meaningful question. This is the real test.
      full         whole-sample quantile. NOT tradeable; shown only to size the look-ahead.
      expanding    kept so the failure above is reproducible, never for a verdict.
    """
    out = pd.Series(np.nan, index=a.index)
    for _tenor, g in a.groupby("tenor"):
        g = g.sort_values("auctionDate")
        v = g[col]
        if mode == "rolling":
            thr = v.shift(1).rolling(ROLL, min_periods=MIN_HIST).quantile(TERCILE)
        elif mode == "expanding":
            thr = v.shift(1).expanding(min_periods=MIN_HIST).quantile(TERCILE)
        else:
            thr = pd.Series(v.quantile(TERCILE), index=g.index)
        out.loc[g.index] = (v >= thr).where(thr.notna())
    return out


def snapback_stats(pnl: pd.DataFrame, mask: pd.Series | None, label: str) -> dict:
    df = pnl if mask is None else pnl[mask.reindex(pnl.index).fillna(False).astype(bool)]
    df = df.dropna(subset=["snapback"])
    if len(df) < 8:
        return {"label": label, "n": len(df)}
    weeks = pd.to_datetime(df["auctionDate"]).dt.to_period("W")
    mean, se, t = clustered_t(df["snapback"], weeks)
    return {"label": label, "n": len(df), "gross": mean, "net": mean - COST_RT,
            "t": t, "hit": float((df["snapback"] > 0).mean())}


def show(rows: list[dict], title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'':26}{'n':>5}{'gross bp':>10}{'net bp':>9}{'clustered t':>13}{'hit':>7}")
    for r in rows:
        if r.get("gross") is None or "gross" not in r:
            print(f"  {r['label']:26}{r['n']:>5}{'too few events':>32}")
            continue
        print(f"  {r['label']:26}{r['n']:>5}{r['gross']:>+10.3f}{r['net']:>+9.3f}"
              f"{r['t']:>+13.2f}{r['hit']:>7.0%}")


def main() -> None:
    print("Fetching auction results with bidder composition (cached after first run)...")
    res = fetch_auction_results()
    print("Fetching FRED CMT yields...")
    yields = fetch_yields()

    res["tenor"] = res["securityTerm"].map(parse_tenor)
    res = res.dropna(subset=["tenor"])
    res["tenor"] = res["tenor"].astype(int)
    res["reopen"] = res["reopening"].astype(str).str.strip().str.lower().eq("yes")
    res = add_conditioners(res)

    # The sleeve trades NEW ISSUES only (reopenings measured at t = -0.1), and only the
    # tenors it actually holds.
    a = res[(~res["reopen"]) & res["tenor"].isin([5, 10])].copy()
    a["size_z"] = 0.0
    a = a.dropna(subset=["w_dealer"]).sort_values("auctionDate")
    print(f"\n  {len(a)} new-issue 5y/10y auctions with bidder data, "
          f"{a['auctionDate'].min().date()}..{a['auctionDate'].max().date()}")
    print("  by tenor: " + ", ".join(f"{t}y={n}" for t, n in
                                     a["tenor"].value_counts().sort_index().items()))
    print(f"  dealer share (ex-SOMA): median {a['dealer_share'].median():.1%}, "
          f"p10 {a['dealer_share'].quantile(.1):.1%}, p90 {a['dealer_share'].quantile(.9):.1%}")

    measures = build_measures(yields)
    panel = event_panel(measures, a)
    if panel.empty:
        print("\n  PANEL EMPTY."); return
    pnl = leg_pnl(panel[panel["measure"] == "fly"], Legs())
    if pnl.empty:
        print("\n  NO P&L."); return
    pnl = pnl.merge(a[["cusip", "auctionDate", "w_dealer", "w_btc", "w_indirect",
                       "dealer_share"]], on=["cusip", "auctionDate"], how="left")

    print("\n" + "=" * 96)
    print("PRE-COMMITTED RULE: trade the top tercile of DEALER SHARE, ranked within tenor")
    print("=" * 96)
    print(f"  Snapback leg only (T-1 -> T+3), fly, cost {COST_RT:.2f}bp/round trip.")
    print("  The unconditional row is the benchmark — the conditional LIFT is what matters.")

    for mode, tag in (("rolling", f"ROLLING-{ROLL} tercile (causal + detrended — THE TEST)"),
                      ("full", "FULL-SAMPLE tercile (look-ahead — size of the bias)"),
                      ("expanding", "EXPANDING tercile (BROKEN by the trend — see docstring)")):
        flag = flag_top_tercile(a, "w_dealer", mode)
        a[f"top_{mode}"] = flag
        m = pnl.merge(a[["cusip", "auctionDate", f"top_{mode}"]],
                      on=["cusip", "auctionDate"], how="left")[f"top_{mode}"]
        rows = [snapback_stats(pnl, None, "unconditional (all)"),
                snapback_stats(pnl, m == True, "top tercile (WEAK)"),      # noqa: E712
                snapback_stats(pnl, m == False, "rest (STRONG)")]          # noqa: E712
        show(rows, tag)

    # Pre-specified alternatives, reported together so nothing is cherry-picked
    print("\n" + "=" * 96)
    print("THE OTHER TWO FREE MEASURES — reported, not selected from")
    print("=" * 96)
    for col, name in (("w_btc", "low bid-to-cover"), ("w_indirect", "low indirect share")):
        flag = flag_top_tercile(a, col, "rolling")
        a[f"t_{col}"] = flag
        m = pnl.merge(a[["cusip", "auctionDate", f"t_{col}"]],
                      on=["cusip", "auctionDate"], how="left")[f"t_{col}"]
        show([snapback_stats(pnl, m == True, f"top tercile ({name})"),     # noqa: E712
              snapback_stats(pnl, m == False, "rest")],                    # noqa: E712
             f"{name}, rolling tercile")

    # Era split — the standing requirement is non-negative in EVERY era, not one big t-stat
    print("\n" + "=" * 96)
    print("BY ERA — the bar is NON-NEGATIVE EVERYWHERE, not a single large t")
    print("=" * 96)
    flag = flag_top_tercile(a, "w_dealer", "rolling")
    a["top"] = flag
    m = pnl.merge(a[["cusip", "auctionDate", "top"]],
                  on=["cusip", "auctionDate"], how="left")["top"]
    pnl["_top"] = m.values
    yr = pd.to_datetime(pnl["auctionDate"]).dt.year
    eras = [("pre-2008", yr < 2008), ("2008-2014 QE", (yr >= 2008) & (yr <= 2014)),
            ("2015-2021", (yr >= 2015) & (yr <= 2021)), ("2022- QT", yr >= 2022)]
    print(f"  {'era':16}{'uncond n':>10}{'uncond bp':>11}{'top-terc n':>12}"
          f"{'top-terc bp':>13}{'lift':>8}")
    for name, sel in eras:
        u = pnl[sel]["snapback"].dropna()
        c = pnl[sel & (pnl["_top"] == True)]["snapback"].dropna()      # noqa: E712
        if len(u) < 5:
            continue
        lift = (c.mean() - u.mean()) if len(c) >= 5 else np.nan
        print(f"  {name:16}{len(u):>10}{u.mean():>+11.3f}{len(c):>12}"
              f"{(c.mean() if len(c) >= 5 else np.nan):>+13.3f}{lift:>+8.3f}")

    out = ROOT / "results" / "auction_conditioning"
    out.mkdir(parents=True, exist_ok=True)
    pnl.to_csv(out / "conditioned_pnl.csv", index=False)
    print(f"\nWrote {out / 'conditioned_pnl.csv'}")
    print("\n  READ: the conditioning earns its place only if the top-tercile NET beats the")
    print("  unconditional NET by enough to justify trading half as often, AND the lift is")
    print("  non-negative in every era. A lift that exists only full-sample, or only in one")
    print("  era, is the threshold search this design exists to avoid.")


if __name__ == "__main__":
    main()
