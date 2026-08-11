"""Month-end index extension — event study kill switch.

Bond indices rebalance at the close of the LAST BUSINESS DAY of each month. New issuance
enters, sub-1y bonds drop out, everything else ages. The net is almost always an EXTENSION of
index duration, so anyone benchmarked to the index must BUY duration at that close or accrue
tracking error. Trillions track the Bloomberg Agg, so that is a large, price-insensitive,
calendar-bound flow — structurally the same limits-to-arbitrage setup as the auction
concession in `treasury_auction_lab.py`.

DIFFERENT SHAPE FROM THE AUCTION. The auction concession is localised at ONE curve point, so a
butterfly isolates it. Extension demand is general duration demand, so a fly is probably the
wrong instrument. Three expressions are tested:

  outright   10y yield         directional; captures everything but carries full rates risk
  slope      2s10s, 5s30s      if funds buy duration where it is cheapest (the long end),
                               the curve should FLATTEN into month-end and steepen after
  fly        2s10s30s          included only to confirm it is NOT a curvature effect

SIGN CONVENTION: buying pressure -> prices up -> YIELDS DOWN into month-end, reverting after.
For slopes, "flattening" = the long end richening relative to the short end = the 2s10s spread
NARROWING. All series are signed so that POSITIVE = the hypothesised pre-event move, so a real
effect shows the same hump shape as the auction study.

CONDITIONING: the true driver is the projected extension, published mid-month by index
providers — data we do not have. Proxy it with the duration-weighted coupon issuance that
SETTLED during the month, reusing the TreasuryDirect cache from the auction lab. Bigger
issuance -> bigger extension -> bigger effect, if the mechanism is real.

Same caveats as the auction lab: FRED CMT is a fitted daily par curve struck once, not
tradeable prices. And month-end extension is a well-known desk trade, so unlike the auction it
may simply be arbitraged away.

Run: python scripts/month_end_extension_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache" / "treasury"
OUT = ROOT / "results" / "month_end"

# approximate Macaulay duration per coupon tenor, for the issuance-weighted extension proxy
DURATION = {2: 1.9, 3: 2.9, 5: 4.6, 7: 6.3, 10: 8.4, 20: 13.5, 30: 17.5}

ERAS = [
    ("1980-2007 pre-GFC", "1980-01-01", "2007-12-31"),
    ("2008-2014 QE1-3", "2008-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2021 covid QE", "2020-01-01", "2021-12-31"),
    ("2022- QT", "2022-01-01", "2030-12-31"),
]


def load_yields() -> pd.DataFrame:
    path = CACHE / "cmt_yields.csv"
    if not path.exists():
        raise SystemExit("run scripts/treasury_auction_lab.py first to populate the cache")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [int(c) for c in df.columns]
    return df * 100.0        # -> basis points


def build_measures(bp: pd.DataFrame) -> dict[str, pd.Series]:
    """All signed so POSITIVE = the hypothesised move INTO month-end."""
    m: dict[str, pd.Series] = {}
    # buying duration pushes yields DOWN, so negate to make the hypothesis positive
    for t in (5, 10, 30):
        if t in bp:
            m[f"outright {t}y"] = -bp[t]
    # flattening = long end richens vs short end = spread narrows, so negate
    for lo, hi in ((2, 10), (5, 30), (2, 30)):
        if lo in bp and hi in bp:
            m[f"slope {lo}s{hi}s"] = -(bp[hi] - bp[lo])
    # curvature, included to confirm this is NOT a fly effect
    if all(t in bp for t in (2, 10, 30)):
        w1 = (30 - 10) / (30 - 2)
        m["fly 2s10s30s"] = -(bp[10] - w1 * bp[2] - (1 - w1) * bp[30])
    return {k: v.dropna() for k, v in m.items()}


def month_end_dates(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last available trading day of each month, taken from the CMT calendar itself so
    holidays are handled without a separate calendar."""
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).last())


def issuance_proxy() -> pd.Series:
    """Duration-weighted coupon issuance SETTLING each month — proxy for the extension size.

    Index membership follows the ISSUE date, not the auction date, so bonds auctioned late in
    the month may not enter until the following month's rebalance. Grouping on issueDate is
    the right call.
    """
    path = CACHE / "auctions_1980_2026.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    a = pd.read_csv(path, parse_dates=["auctionDate", "issueDate"])
    a["amt"] = pd.to_numeric(a["offeringAmount"], errors="coerce")

    def tenor(term: str) -> float | None:
        if not isinstance(term, str):
            return None
        yrs = mos = 0.0
        parts = term.replace("-", " ").split()
        for i, tok in enumerate(parts):
            if tok.lower().startswith("year") and i:
                yrs = float(parts[i - 1])
            elif tok.lower().startswith("month") and i:
                mos = float(parts[i - 1])
        total = yrs + mos / 12.0
        if total <= 0:
            return None
        near = min(DURATION, key=lambda c: abs(c - total))
        return DURATION[near] if abs(near - total) <= 1.5 else None

    a["dur"] = a["securityTerm"].map(tenor)
    a = a.dropna(subset=["dur", "amt", "issueDate"])
    a["dv"] = a["amt"] * a["dur"]
    g = a.groupby(a["issueDate"].dt.to_period("M"))["dv"].sum()
    return g / 1e9 / 1e3      # $bn * years, scaled


def event_panel(measures: dict[str, pd.Series], dates: list[pd.Timestamp],
                pre: int = 10, post: int = 10) -> pd.DataFrame:
    recs = []
    for name, v in measures.items():
        idx = v.index
        pos = pd.Series(np.arange(len(idx)), index=idx)
        for d in dates:
            i = pos.get(d)
            if i is None:
                continue
            i = int(i)
            if i - pre < 0 or i + post >= len(idx):
                continue
            w = v.iloc[i - pre:i + post + 1]
            if w.isna().any():
                continue
            base = w.iloc[0]
            for k, val in enumerate(w.to_numpy()):
                recs.append((name, d, k - pre, val - base))
    return pd.DataFrame(recs, columns=["measure", "date", "rel_day", "chg_bp"])


def clustered_t(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std() == 0:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(len(x)))


def main() -> None:
    bp = load_yields()
    measures = build_measures(bp)
    dates = month_end_dates(bp.dropna(subset=[10]).index)
    panel = event_panel(measures, dates)
    if panel.empty:
        print("no events")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "event_panel.csv", index=False)

    print("=" * 100)
    print(f"MONTH-END EXTENSION — mean cumulative change from T-10 (bp), {panel['date'].nunique()} month-ends")
    print("  T0 = last business day of the month.  POSITIVE = the hypothesised pre-event move")
    print("  (yields falling / curve flattening).  A real effect = rise into T0, revert after.")
    print("=" * 100)
    shape = panel.pivot_table(index="rel_day", columns="measure", values="chg_bp", aggfunc="mean")
    cols = list(shape.columns)
    print(f"  {'rel_day':>8s} " + "".join(f"{c[:12]:>13s}" for c in cols))
    for day, row in shape.iterrows():
        mark = "  <-- rebalance" if day == 0 else ""
        print(f"  {day:>8d} " + "".join(f"{v:>+13.3f}" for v in row.values) + mark)

    # ---- trade P&L. Hypothesis: hold INTO month-end, exit after the reversion.
    print("\n" + "=" * 100)
    print("TRADE — enter T-3, exit T0 (ride the flow in), and the reversion T0 -> T+3")
    print("=" * 100)
    print(f"  {'measure':16s} {'n':>4s} | {'INTO (T-3->T0)':>22s} | {'REVERSION (T0->T+3)':>22s}")
    print(f"  {'':16s} {'':>4s} | {'mean bp':>10s} {'t':>10s} | {'mean bp':>10s} {'t':>10s}")
    print("  " + "-" * 82)
    wide = panel.pivot_table(index=["measure", "date"], columns="rel_day", values="chg_bp")
    for name in cols:
        w = wide.loc[name].dropna(subset=[-3, 0, 3])
        into = w[0] - w[-3]
        rev = -(w[3] - w[0])
        print(f"  {name:16s} {len(w):>4d} | {into.mean():>+10.3f} {clustered_t(into):>+10.2f} | "
              f"{rev.mean():>+10.3f} {clustered_t(rev):>+10.2f}")

    # ---- does it scale with issuance, as the mechanism requires?
    iss = issuance_proxy()
    if not iss.empty:
        print("\n" + "=" * 100)
        print("MECHANISM CHECK — bigger duration issuance should mean a bigger extension")
        print("=" * 100)
        print(f"  {'measure':16s} {'high issuance':>26s} {'low issuance':>26s}")
        print(f"  {'':16s} {'into':>12s} {'t':>13s} {'into':>12s} {'t':>13s}")
        print("  " + "-" * 70)
        for name in cols:
            w = wide.loc[name].dropna(subset=[-3, 0])
            per = pd.PeriodIndex(w.index, freq="M")
            v = iss.reindex(per).to_numpy()
            into = (w[0] - w[-3]).to_numpy()
            ok = ~np.isnan(v)
            if ok.sum() < 40:
                continue
            hi = v[ok] > np.median(v[ok])
            a, b = into[ok][hi], into[ok][~hi]
            print(f"  {name:16s} {a.mean():>+12.3f} {a.mean()/a.std()*np.sqrt(len(a)):>+13.2f} "
                  f"{b.mean():>+12.3f} {b.mean()/b.std()*np.sqrt(len(b)):>+13.2f}")

    # ---- era stability
    best = max(cols, key=lambda c: abs(clustered_t((wide.loc[c].dropna(subset=[-3, 0]))[0]
                                                   - (wide.loc[c].dropna(subset=[-3, 0]))[-3]) or 0))
    print("\n" + "=" * 100)
    print(f"BY ERA — {best} (strongest measure), into-move T-3 -> T0")
    print("=" * 100)
    w = wide.loc[best].dropna(subset=[-3, 0])
    into = (w[0] - w[-3])
    for label, lo, hi in ERAS:
        g = into[(into.index >= lo) & (into.index <= hi)]
        if len(g) >= 12:
            print(f"  {label:24s} n={len(g):>4d}  mean {g.mean():>+7.3f}bp  t={clustered_t(g):>+6.2f}")

    print(f"\n  wrote {OUT}/event_panel.csv")
    print("\n  CAVEATS: CMT is a fitted daily par curve, not tradeable prices. Month-end")
    print("  extension is a well-known desk trade, so unlike the auction concession it may")
    print("  simply be arbitraged away — that is exactly what this test is for.")


if __name__ == "__main__":
    main()
