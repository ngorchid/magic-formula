"""Treasury auction-cycle event study — is the pre-auction concession still there?

The thesis (Lou, Yan & Zhang, RFS 2013): Treasury is a price-insensitive, calendar-bound
seller. Dealers are REQUIRED to bid and know they will own duration at 1:01pm, so they
pre-hedge by selling the sector in the days before; real-money demand withdraws from the
secondary market and pools at the auction. Both push the auctioned sector's yield UP into
the event ("concession"). After the auction, as dealers distribute inventory, it reverses
("snapback"). It survives despite being fully anticipated because absorbing the supply
needs balance sheet, and balance sheet is the scarce thing being priced.

This script is the KILL SWITCH, not a backtest. If the average fly does not show the
characteristic hump around T0, there is nothing to build and we stop here.

Three measures per event, all in basis points:
  outright  raw change in the auctioned tenor's yield        (contaminated by the rate level)
  fly       2*belly - wing_lo - wing_hi, market convention   (what you'd trade in futures)
  resid     belly hedged on both wings by trailing OLS       (no 2:1:1 assumption)

The hedge ratios for `resid` are fit on a trailing window that ENDS BEFORE the event window
opens, so an auction never contributes to its own hedge.

Standard errors are clustered by calendar week: the 2y/5y/7y auctions land on consecutive
days in the same week and their +/-10d windows overlap almost completely, so unclustered
t-stats here are badly overstated.

Data: TreasuryDirect TA_WS (auction calendar, 1980-) + FRED constant-maturity yields. Both
keyless. NB CMT is a fitted par curve struck once daily, not executable futures prices — a
positive result here is an existence proof that justifies buying real futures data, nothing
more.

Run: python scripts/treasury_auction_lab.py
"""
from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache" / "treasury"
OUT = ROOT / "results" / "treasury_auction"

FRED_TENORS = [1, 2, 3, 5, 7, 10, 20, 30]

# Nearest neighbours available on the CMT curve. The 30y has no wing beyond it, so its
# "fly" is really a linear extrapolation off 10s/20s — flagged in the output.
WINGS = {2: (1, 3), 3: (2, 5), 5: (3, 7), 7: (5, 10), 10: (7, 20), 20: (10, 30), 30: (10, 20)}

CANONICAL = [2, 3, 5, 7, 10, 20, 30]

# Auction -> futures contract whose CTD actually sits at that point on the curve. ZN is a
# ~7y instrument in practice, NOT a 10y; the 10y auction wants TN.
CONTRACT = {2: "ZT", 3: "(Z3N, illiquid)", 5: "ZF", 7: "ZN", 10: "TN", 20: "ZB", 30: "UB"}

ERAS = [
    ("1980-2007 pre-GFC", "1980-01-01", "2007-12-31"),
    ("2008-2014 QE1-3", "2008-01-01", "2014-12-31"),
    ("2015-2019 normalisation", "2015-01-01", "2019-12-31"),
    ("2020-2021 covid QE", "2020-01-01", "2021-12-31"),
    ("2022- QT / heavy supply", "2022-01-01", "2030-12-31"),
]


# ----------------------------------------------------------------------------- data


def fetch_auctions(start_year: int = 1980, end_year: int = 2026) -> pd.DataFrame:
    """Auction calendar from TreasuryDirect. The API hard-caps at 250 rows per call and
    ignores pagesize, so we walk the /search endpoint one year and one type at a time."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"auctions_{start_year}_{end_year}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["auctionDate", "announcementDate", "issueDate"])

    url = "https://www.treasurydirect.gov/TA_WS/securities/search"
    rows = []
    for year in range(start_year, end_year + 1):
        for sec_type in ("Note", "Bond"):
            params = {
                "format": "json",
                "startDate": f"{year}-01-01",
                "endDate": f"{year}-12-31",
                "dateFieldName": "auctionDate",
                "type": sec_type,
            }
            try:
                r = requests.get(url, params=params, timeout=120)
                data = r.json() if r.text.strip() else []
            except Exception as exc:  # noqa: BLE001 - network flake shouldn't kill the run
                print(f"  warn {year} {sec_type}: {exc!r}")
                continue
            rows += data
        print(f"  fetched {year} ({len(rows)} cumulative)", end="\r")

    df = pd.DataFrame(rows)
    keep = ["cusip", "securityType", "securityTerm", "auctionDate", "announcementDate",
            "issueDate", "reopening", "offeringAmount", "totalAccepted", "bidToCoverRatio",
            "highYield"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in ("auctionDate", "announcementDate", "issueDate"):
        df[c] = pd.to_datetime(df[c]).dt.tz_localize(None).dt.normalize()
    for c in ("offeringAmount", "totalAccepted", "bidToCoverRatio", "highYield"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["cusip", "auctionDate"]).sort_values("auctionDate")
    df.to_csv(path, index=False)
    print(f"\n  cached {len(df)} auctions -> {path}")
    return df


def parse_tenor(term: str) -> float | None:
    """'9-Year 11-Month' -> 10.0. Reopenings carry a stub term slightly shorter than the
    original issue, so snap to the nearest canonical point; drop anything further than
    0.75y away (odd 1980s terms, cash-management issues)."""
    if not isinstance(term, str):
        return None
    years = months = 0.0
    parts = term.replace("-", " ").split()
    for i, tok in enumerate(parts):
        low = tok.lower()
        if low.startswith("year") and i:
            years = float(parts[i - 1])
        elif low.startswith("month") and i:
            months = float(parts[i - 1])
    total = years + months / 12.0
    if total <= 0:
        return None
    nearest = min(CANONICAL, key=lambda c: abs(c - total))
    return float(nearest) if abs(nearest - total) <= 0.75 else None


def fetch_yields() -> pd.DataFrame:
    """Daily constant-maturity Treasury yields (percent) from FRED, keyless CSV endpoint."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "cmt_yields.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [int(c) for c in df.columns]  # CSV round-trip stringifies the tenors
        return df

    out = {}
    for t in FRED_TENORS:
        sid = f"DGS{t}"
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=120)
        s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0]
        out[t] = pd.to_numeric(s, errors="coerce")
        print(f"  fetched {sid} ({s.notna().sum()} obs)", end="\r")
    df = pd.DataFrame(out).sort_index()
    df = df.dropna(how="all")
    df.to_csv(path)
    print(f"\n  cached yields {df.index.min().date()}..{df.index.max().date()} -> {path}")
    return df


# ------------------------------------------------------------------- signal construction


def build_measures(yields: pd.DataFrame, window: int = 250) -> dict[int, pd.DataFrame]:
    """Per tenor: outright / fly / resid, all in bp. `resid` uses trailing-OLS hedge ratios
    lagged by 11 business days so an event never informs its own hedge."""
    bp = yields * 100.0
    measures = {}
    for tenor, (lo, hi) in WINGS.items():
        cols = [tenor, lo, hi]
        if any(c not in bp.columns for c in cols):
            continue
        sub = bp[cols].dropna()
        if len(sub) < window + 50:
            continue
        belly, wlo, whi = sub[tenor], sub[lo], sub[hi]

        fly = 2.0 * belly - wlo - whi

        # Rolling OLS of d(belly) on d(wings); hedge ratios lagged past the event window.
        d = sub.diff()
        y, x1, x2 = d[tenor], d[lo], d[hi]
        cov11 = x1.rolling(window).var()
        cov22 = x2.rolling(window).var()
        cov12 = x1.rolling(window).cov(x2)
        cov1y = x1.rolling(window).cov(y)
        cov2y = x2.rolling(window).cov(y)
        det = cov11 * cov22 - cov12**2
        b1 = ((cov1y * cov22 - cov2y * cov12) / det).shift(11)
        b2 = ((cov2y * cov11 - cov1y * cov12) / det).shift(11)
        resid_d = y - b1 * x1 - b2 * x2
        resid = resid_d.cumsum()

        measures[tenor] = pd.DataFrame(
            {"outright": belly, "fly": fly, "resid": resid}
        ).dropna(subset=["outright", "fly"])
    return measures


def event_panel(measures: dict[int, pd.DataFrame], auctions: pd.DataFrame,
                pre: int = 10, post: int = 10) -> pd.DataFrame:
    """Long panel: one row per (auction, relative business day, measure), value = change in
    bp from the T-`pre` baseline. Positive = the belly cheapened relative to its wings."""
    recs = []
    for tenor, m in measures.items():
        idx = m.index
        pos_of = pd.Series(np.arange(len(idx)), index=idx)
        rows = auctions[auctions["tenor"] == tenor]
        for _, a in rows.iterrows():
            adate = a["auctionDate"]
            loc = pos_of.get(adate)
            if loc is None:  # auction on a CMT holiday; snap to next available day
                after = idx[idx > adate]
                if len(after) == 0:
                    continue
                loc = int(pos_of[after[0]])
            loc = int(loc)
            if loc - pre < 0 or loc + post >= len(idx):
                continue
            win = m.iloc[loc - pre: loc + post + 1]
            for col in ("outright", "fly", "resid"):
                if col not in win or win[col].isna().any():
                    continue
                base = win[col].iloc[0]
                for k, val in enumerate(win[col].to_numpy()):
                    recs.append((tenor, a["cusip"], adate, a["reopen"], a["size_z"],
                                 col, k - pre, val - base))
    return pd.DataFrame(recs, columns=["tenor", "cusip", "auctionDate", "reopen",
                                       "size_z", "measure", "rel_day", "chg_bp"])


# ------------------------------------------------------------------------------- stats


def clustered_t(x: pd.Series, groups: pd.Series) -> tuple[float, float, float]:
    """Mean, cluster-robust SE, t-stat. Clusters = calendar weeks (overlapping windows)."""
    x = x.dropna()
    groups = groups.reindex(x.index)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    mean = x.mean()
    e = x - mean
    g_sums = e.groupby(groups).sum()
    G = len(g_sums)
    if G < 2:
        return mean, np.nan, np.nan
    var = (g_sums**2).sum() / n**2 * (G / (G - 1))
    se = float(np.sqrt(var))
    return float(mean), se, float(mean / se) if se > 0 else np.nan


@dataclass
class Legs:
    """Entry/pivot/exit in business days from the auction.

    CRITICAL TIMING: competitive bidding closes at 1:00pm ET but the CMT curve is struck at
    ~3:30pm, so day T0's observation is ALREADY POST-AUCTION. The last clean pre-auction
    print is T-1. Pivoting on T0 straddles the reversal and mixes the two legs together —
    the empirical shape confirms this, with the concession peaking at T-1 and rolling over
    on the auction-day print.
    """
    entry: int = -5      # short the belly as the concession builds
    pivot: int = -1      # last pre-auction observation: cover the short, flip long
    exit: int = 3        # snapback runs a few days as dealers distribute


def leg_pnl(panel: pd.DataFrame, legs: Legs) -> pd.DataFrame:
    """Per-event P&L in bp of fly for each leg.

      concession  short belly T_entry -> T_pivot   pnl = +(fly[pivot] - fly[entry])
      snapback    long  belly T_pivot -> T_exit    pnl = -(fly[exit]  - fly[pivot])
      combined    both, flipping at the pivot

    Sign convention: fly up = belly cheapened vs wings = short-belly position gains.
    """
    wide = panel.pivot_table(index=["tenor", "cusip", "auctionDate", "reopen", "size_z", "measure"],
                             columns="rel_day", values="chg_bp")
    need = [legs.entry, legs.pivot, legs.exit]
    if any(d not in wide.columns for d in need):
        return pd.DataFrame()
    wide = wide.dropna(subset=need)
    out = wide[need].copy()
    out.columns = ["at_entry", "at_pivot", "at_exit"]
    out["concession"] = out["at_pivot"] - out["at_entry"]
    out["snapback"] = -(out["at_exit"] - out["at_pivot"])
    out["combined"] = out["concession"] + out["snapback"]
    return out.reset_index()


def stat_block(df: pd.DataFrame, label: str, cost_rt: float) -> dict:
    """Gross/net mean bp + clustered t for each leg. Costs: 1 round trip per single leg,
    2 for the combined (entry 0.5 + flip 1.0 + exit 0.5)."""
    weeks = df["auctionDate"].dt.to_period("W")
    row = {"cut": label, "n": len(df)}
    for leg, rts in (("concession", 1), ("snapback", 1), ("combined", 2)):
        mean, se, t = clustered_t(df[leg], weeks)
        row[f"{leg}_gross"] = mean
        row[f"{leg}_net"] = mean - cost_rt * rts
        row[f"{leg}_t"] = t
        row[f"{leg}_ir"] = mean / df[leg].std() * np.sqrt(len(df)) if df[leg].std() > 0 else np.nan
    return row


def print_table(rows: list[dict], title: str) -> None:
    if not rows:
        print(f"\n  (no data for {title})")
        return
    if title:
        print("\n" + "=" * 104)
        print(title)
        print("=" * 104)
    print(f"  {'cut':30s} {'n':>5s} | {'concession':>18s} | {'snapback':>18s} | {'combined':>18s}")
    print(f"  {'':30s} {'':>5s} | {'gross':>6s}{'net':>6s}{'t':>6s} | "
          f"{'gross':>6s}{'net':>6s}{'t':>6s} | {'gross':>6s}{'net':>6s}{'t':>6s}")
    print("  " + "-" * 100)
    for r in rows:
        line = f"  {r['cut']:30s} {r['n']:>5d} |"
        for leg in ("concession", "snapback", "combined"):
            line += (f" {r[f'{leg}_gross']:>+6.2f}{r[f'{leg}_net']:>+6.2f}"
                     f"{r[f'{leg}_t']:>+6.2f} |")
        print(line)
    print("\n  units = bp of fly. t is clustered by auction week. net subtracts "
          f"round-trip cost.")


def plot_event_study(panel: pd.DataFrame, measure: str = "fly") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = panel[panel["measure"] == measure]
    tenors = [t for t in CANONICAL if t in set(sub["tenor"])]
    ncol = 3
    nrow = int(np.ceil((len(tenors) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, tenor in zip(axes, tenors):
        g = sub[sub["tenor"] == tenor]
        m = g.groupby("rel_day")["chg_bp"].mean()
        n_ev = g["cusip"].nunique()
        se = g.groupby("rel_day")["chg_bp"].sem()
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(0, color="crimson", lw=0.9, ls="--")
        ax.plot(m.index, m.values, color="#1f4e79", lw=1.8)
        ax.fill_between(m.index, m - 2 * se, m + 2 * se, color="#1f4e79", alpha=0.15)
        ax.set_title(f"{tenor}y  ({CONTRACT[tenor]})  n={n_ev}", fontsize=10)
        ax.set_ylabel("bp vs T-10", fontsize=8)
        ax.tick_params(labelsize=8)

    pooled = axes[len(tenors)]
    m = sub.groupby("rel_day")["chg_bp"].mean()
    se = sub.groupby("rel_day")["chg_bp"].sem()
    pooled.axhline(0, color="0.7", lw=0.8)
    pooled.axvline(0, color="crimson", lw=0.9, ls="--")
    pooled.plot(m.index, m.values, color="black", lw=2.0)
    pooled.fill_between(m.index, m - 2 * se, m + 2 * se, color="black", alpha=0.15)
    pooled.set_title(f"ALL TENORS POOLED  n={sub['cusip'].nunique()}", fontsize=10)
    pooled.tick_params(labelsize=8)
    for ax in axes[len(tenors) + 1:]:
        ax.axis("off")
    for ax in axes[-ncol:]:
        ax.set_xlabel("business days from auction", fontsize=8)

    fig.suptitle(f"Treasury auction cycle — mean cumulative change in {measure} "
                 f"(bp, +ve = belly cheapens vs wings)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"event_study_{measure}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


# -------------------------------------------------------------------------------- main


def main() -> None:
    cost_rt = 0.75   # bp of fly per round trip — the ~$100 on a 6-contract fly at $135/bp
    legs = Legs()

    print("Fetching auction calendar (cached after first run)...")
    auctions = fetch_auctions()
    print("Fetching FRED CMT yields...")
    yields = fetch_yields()

    auctions["tenor"] = auctions["securityTerm"].map(parse_tenor)
    auctions = auctions.dropna(subset=["tenor"])
    auctions["tenor"] = auctions["tenor"].astype(int)
    auctions["reopen"] = auctions["reopening"].astype(str).str.strip().str.lower().eq("yes")

    # Size surprise vs the trailing median for the SAME tenor — known at announcement (T-7),
    # so it is legitimately usable to condition the concession leg.
    auctions = auctions.sort_values("auctionDate")
    med = (auctions.groupby("tenor")["offeringAmount"]
           .transform(lambda s: s.shift(1).rolling(8, min_periods=4).median()))
    auctions["size_z"] = (auctions["offeringAmount"] / med - 1.0).fillna(0.0)

    print(f"\n  {len(auctions)} auctions mapped to canonical tenors, "
          f"{auctions['auctionDate'].min().date()}..{auctions['auctionDate'].max().date()}")
    print("  by tenor: " + ", ".join(f"{t}y={n}" for t, n in
                                     auctions['tenor'].value_counts().sort_index().items()))

    measures = build_measures(yields)
    print(f"  built measures for tenors: {sorted(measures)}")

    panel = event_panel(measures, auctions)
    if panel.empty:
        print("\n  PANEL EMPTY — nothing to study.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "event_panel.csv", index=False)

    # ---- the kill switch: does the hump exist at all?
    print("\n" + "=" * 104)
    print("EVENT STUDY — mean cumulative change from T-10 (bp), pooled across tenors")
    print("=" * 104)
    shape = panel.pivot_table(index="rel_day", columns="measure", values="chg_bp", aggfunc="mean")
    print(f"  {'rel_day':>8s} " + "".join(f"{c:>11s}" for c in shape.columns))
    for day, row in shape.iterrows():
        mark = "  <-- auction" if day == 0 else ""
        print(f"  {day:>8d} " + "".join(f"{v:>+11.3f}" for v in row.values) + mark)

    # The shape BY ERA is the real diagnostic. A P&L table can be dragged around by a few
    # outliers; if the hump itself inverts in a sub-period, the mechanism is not stable.
    fly_panel = panel[panel["measure"] == "fly"]
    print("\n" + "=" * 104)
    print("EVENT SHAPE BY ERA (fly, bp from T-10) — is the hump stable?")
    print("=" * 104)
    days = [-10, -5, -3, -1, 0, 1, 2, 3, 5, 10]
    print(f"  {'era':30s} {'n':>5s} " + "".join(f"{('T'+str(d)):>8s}" for d in days))
    print("  " + "-" * 100)
    for label, lo, hi in ERAS:
        g = fly_panel[(fly_panel["auctionDate"] >= lo) & (fly_panel["auctionDate"] <= hi)]
        if g.empty:
            continue
        m = g.groupby("rel_day")["chg_bp"].mean()
        print(f"  {label:30s} {g['cusip'].nunique():>5d} "
              + "".join(f"{m.get(d, np.nan):>+8.2f}" for d in days))

    for meas in ("fly", "resid"):
        if meas in set(panel["measure"]):
            plot_event_study(panel, meas)

    # ---- trade P&L, sliced
    pnl = leg_pnl(panel[panel["measure"] == "fly"], legs)
    if pnl.empty:
        print("\n  no complete event windows for the fly measure.")
        return

    rows = [stat_block(pnl, "ALL (fly)", cost_rt)]
    resid_pnl = leg_pnl(panel[panel["measure"] == "resid"], legs)
    if not resid_pnl.empty:
        rows.append(stat_block(resid_pnl, "ALL (resid, OLS-hedged)", cost_rt))
    out_pnl = leg_pnl(panel[panel["measure"] == "outright"], legs)
    if not out_pnl.empty:
        rows.append(stat_block(out_pnl, "ALL (outright, unhedged)", cost_rt))
    print_table(rows, f"TRADE P&L — entry T{legs.entry:+d}, auction T0, exit T{legs.exit:+d}")

    rows = [stat_block(g, f"{int(t)}y  ({CONTRACT[int(t)]})", cost_rt)
            for t, g in pnl.groupby("tenor") if len(g) >= 12]
    print_table(rows, "BY TENOR (fly)")

    # Same cut on the OLS-hedged residual. If the fly number is much bigger than the resid
    # number, the gap is residual slope/curve exposure the 2:1:1 weights leave behind — a
    # factor tilt that happened to pay, not the auction effect.
    if not resid_pnl.empty:
        rows = [stat_block(g, f"{int(t)}y  ({CONTRACT[int(t)]})", cost_rt)
                for t, g in resid_pnl.groupby("tenor") if len(g) >= 12]
        print_table(rows, "BY TENOR (resid, OLS-hedged) — how much survives a clean hedge?")

    rows = []
    for label, lo, hi in ERAS:
        g = pnl[(pnl["auctionDate"] >= lo) & (pnl["auctionDate"] <= hi)]
        if len(g) >= 12:
            rows.append(stat_block(g, label, cost_rt))
    print_table(rows, "BY ERA (fly) — does it survive out of sample?")

    # ---- mechanism checks. If the effect is real these should line up; if it shows up
    # but ignores them, we are probably looking at noise.
    rows = []
    for label, mask in (("new issue", ~pnl["reopen"]), ("reopening", pnl["reopen"])):
        g = pnl[mask]
        if len(g) >= 12:
            rows.append(stat_block(g, label, cost_rt))
    big = pnl["size_z"] > pnl["size_z"].median()
    for label, mask in ((f"size > median", big), ("size <= median", ~big)):
        g = pnl[mask]
        if len(g) >= 12:
            rows.append(stat_block(g, label, cost_rt))
    print_table(rows, "MECHANISM CHECKS (fly) — bigger supply shock should mean bigger concession")

    # The one cut worth taking seriously. Every filter here was justified BEFORE seeing any
    # results, so it is not a fishing expedition:
    #   new issues only  - reopenings are a smaller supply shock (mechanism)
    #   2y / 5y / 10y    - the only auctions with a liquid future at the right curve point
    #   2015 onward      - QE years had the Fed absorbing the supply shock (regime)
    #   resid measure    - no 2:1:1 assumption, cleanest hedge (construction)
    print("\n" + "=" * 104)
    print("PRE-SPECIFIED TRADEABLE CUT — new issues, 2y/5y/10y, 2015+, both measures")
    print("=" * 104)
    rows = []
    for label, src in (("fly (as traded)", pnl), ("resid (clean hedge)", resid_pnl)):
        if src.empty:
            continue
        g = src[(~src["reopen"]) & (src["tenor"].isin([2, 5, 10]))
                & (src["auctionDate"] >= "2015-01-01")]
        if len(g) >= 12:
            rows.append(stat_block(g, label, cost_rt))
        for t in (2, 5, 10):
            gt = g[g["tenor"] == t]
            if len(gt) >= 12:
                rows.append(stat_block(gt, f"  {t}y {CONTRACT[t]} — {label.split()[0]}", cost_rt))
    print_table(rows, "")

    # Window sensitivity. NOT for picking the best cell — for checking the effect is a broad
    # plateau rather than one lucky (entry, exit) pair.
    print("\n" + "=" * 104)
    print("WINDOW SENSITIVITY — combined net bp (pivot fixed at T-1, cost charged)")
    print("=" * 104)
    exits = [1, 2, 3, 5, 7, 10]
    print(f"  {'entry':>7s} " + "".join(f"{('exit T+'+str(e)):>10s}" for e in exits))
    for entry in (-10, -7, -5, -3, -2):
        line = f"  {('T'+str(entry)):>7s} "
        for ex in exits:
            p = leg_pnl(fly_panel, Legs(entry=entry, pivot=-1, exit=ex))
            line += f"{(p['combined'].mean() - 2 * cost_rt):>+10.2f}" if not p.empty else f"{'':>10s}"
        print(line)

    pnl.to_csv(OUT / "event_pnl.csv", index=False)
    print(f"\n  wrote {OUT}/event_pnl.csv, {OUT}/event_panel.csv")

    print("\n  READ THIS BEFORE BELIEVING ANY OF IT:")
    print("   - CMT is a fitted par curve struck once daily, not tradeable futures prices.")
    print("   - 30y 'fly' extrapolates off 10s/20s (no wing beyond 30y) — treat separately.")
    print("   - t-stats cluster by week; overlapping windows still leave them optimistic.")


if __name__ == "__main__":
    main()
