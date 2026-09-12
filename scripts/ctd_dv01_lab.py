"""Re-derive futures DV01s from the cheapest-to-deliver, and check them against the market.

WHY. `auction_settlement_yf.py` found the real futures deliver about HALF the P&L the CMT
model predicts on the 5y fly and nothing on the 10y. The prime suspect is the static DV01
table in the deployed sleeve's `contracts.py`, which its own docstring flags as estimates
that "drift with yield levels ... re-check them if yields move a long way from ~4%". The
2015-2026 sample spans 2% to 5%, and the sleeve SIZES on those numbers -- so if they are
wrong, the live book is mis-sized, not merely mis-backtested.

A Treasury future does not track a bond. It tracks the CHEAPEST-TO-DELIVER out of a basket,
and its sensitivity is

    futures DV01  =  CTD DV01 / conversion factor

Both terms move: the CTD switches as yields cross the 6% notional coupon and as the curve
reshapes, and the conversion factor is fixed per (bond, contract) but differs across the
basket. That is exactly the drift the static table cannot capture.

TWO INDEPENDENT DERIVATIONS, because either alone is easy to get wrong:

  ANALYTICAL  build the deliverable basket from TreasuryDirect (coupon + maturity), compute
              CME conversion factors from the published formula, price each candidate off the
              CMT curve, pick the CTD, and divide its DV01 by the CF.
  EMPIRICAL   regress d(futures price) on d(CMT yield) over a trailing window. The slope IS
              the realised DV01, whatever the CTD happens to be doing. It needs no basket, no
              conversion factor and no assumption -- and it is what the sleeve actually
              experiences.

If the two agree, the analytical machinery is right and we can trust it forward. Where they
disagree, the empirical one is the one that pays the P&L.

Usage:  python scripts/ctd_dv01_lab.py
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

from treasury_auction_lab import CACHE, fetch_yields  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "treasury-auction"))
from treasury_auction.contracts import CONTRACTS  # noqa: E402

# CME deliverable grids: (min remaining years, max remaining years, max ORIGINAL term years)
BASKET = {
    "ZT": (1 + 9 / 12, 2.0, 5.25),
    "ZF": (4 + 2 / 12, 5.25, 5.25),
    "ZN": (6.5, 10.0, 10.0),
    "TN": (9 + 5 / 12, 10.0, 10.0),
    "ZB": (15.0, 25.0, 99.0),
    "UB": (25.0, 99.0, 99.0),
}
FACE = {"ZT": 200_000.0, "ZF": 100_000.0, "ZN": 100_000.0,
        "TN": 100_000.0, "ZB": 100_000.0, "UB": 100_000.0}
LEG_CMT = {"ZT": 2, "ZF": 5, "ZN": 7, "TN": 10, "ZB": 20}
NOTIONAL_COUPON = 0.06


def fetch_securities() -> pd.DataFrame:
    """Every note/bond ever auctioned, with coupon and maturity. Deduped by cusip, since a
    reopening shares its original's cusip, coupon and maturity."""
    path = CACHE / "securities_coupons.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["maturityDate", "issueDate"])
    url = "https://www.treasurydirect.gov/TA_WS/securities/search"
    rows = []
    for year in range(1990, 2027):
        for t in ("Note", "Bond"):
            try:
                r = requests.get(url, params={"format": "json",
                                              "startDate": f"{year}-01-01",
                                              "endDate": f"{year}-12-31",
                                              "dateFieldName": "auctionDate",
                                              "type": t}, timeout=120)
                rows += r.json() if r.text.strip() else []
            except Exception as exc:  # noqa: BLE001
                print(f"  warn {year} {t}: {exc!r}")
        print(f"  fetched {year} ({len(rows)})", end="\r")
    df = pd.DataFrame(rows)
    keep = ["cusip", "securityTerm", "originalSecurityTerm", "interestRate",
            "maturityDate", "issueDate"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["interestRate"] = pd.to_numeric(df["interestRate"], errors="coerce")
    for c in ("maturityDate", "issueDate"):
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["interestRate", "maturityDate"])
    df = df.drop_duplicates(subset=["cusip"]).sort_values("maturityDate")
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  cached {len(df)} securities -> {path}")
    return df


def term_years(s: str) -> float:
    """'9-Year 11-Month' -> 9.917."""
    if not isinstance(s, str):
        return np.nan
    y = m = 0.0
    p = s.replace("-", " ").split()
    for i, tok in enumerate(p):
        if tok.lower().startswith("year") and i:
            y = float(p[i - 1])
        elif tok.lower().startswith("month") and i:
            m = float(p[i - 1])
    return y + m / 12.0 if (y or m) else np.nan


def conversion_factor(coupon: float, first_delivery: pd.Timestamp,
                      maturity: pd.Timestamp) -> float:
    """CME conversion factor: the bond's price per $1 at a 6% yield, maturity rounded DOWN
    to the nearest quarter measured from the first day of the delivery month.

        a = 1/1.03^(z/6)                      b = (c/2)(6-z)/6
        cc = 1/1.03^(2n)   if z < 7, else 1/1.03^(2n+1)
        d  = (c/0.06)(1-cc)                   CF = a[c/2 + cc + d] - b
    """
    months = (maturity.year - first_delivery.year) * 12 + (maturity.month - first_delivery.month)
    months = max(months, 0)
    n = months // 12                     # whole years
    z = ((months - 12 * n) // 3) * 3     # months, rounded DOWN to a quarter
    c = coupon
    a = 1.0 / (1.03 ** (z / 6.0))
    b = (c / 2.0) * ((6.0 - z) / 6.0)
    cc = 1.0 / (1.03 ** (2 * n)) if z < 7 else 1.0 / (1.03 ** (2 * n + 1))
    d = (c / NOTIONAL_COUPON) * (1.0 - cc)
    return a * (c / 2.0 + cc + d) - b


def price_and_dv01(coupon: float, years: float, ytm: float) -> tuple[float, float]:
    """Clean-ish price per 100 and DV01 in price-points per bp, semi-annual, flat yield."""
    n = max(int(round(years * 2)), 1)
    cpn = coupon * 100.0 / 2.0
    y = ytm / 2.0

    def pv(yy: float) -> float:
        t = np.arange(1, n + 1)
        return float((cpn / (1 + yy) ** t).sum() + 100.0 / (1 + yy) ** n)

    p = pv(y)
    bump = 0.0001 / 2.0
    return p, (pv(y - bump) - pv(y + bump)) / 2.0


def curve_yield(row: pd.Series, years: float) -> float:
    """Linear interpolation of the CMT curve (decimal) at `years`."""
    ts = np.array([t for t in (1, 2, 3, 5, 7, 10, 20, 30) if t in row.index],
                  dtype=float)
    vs = np.array([row[int(t)] for t in ts], dtype=float)
    ok = np.isfinite(vs)
    if ok.sum() < 2:
        return np.nan
    return float(np.interp(years, ts[ok], vs[ok])) / 100.0


def analytical_dv01(symbol: str, asof: pd.Timestamp, secs: pd.DataFrame,
                    yrow: pd.Series) -> dict | None:
    """CTD and implied futures DV01 for the front contract as of `asof`."""
    lo, hi, max_orig = BASKET[symbol]
    # Front quarterly delivery month
    q = [3, 6, 9, 12]
    m = next((x for x in q if x >= asof.month), None)
    year = asof.year if m else asof.year + 1
    m = m or 3
    first_delivery = pd.Timestamp(year=year, month=m, day=1)

    s = secs.copy()
    s["rem"] = (s["maturityDate"] - first_delivery).dt.days / 365.25
    s["orig"] = s["originalSecurityTerm"].map(term_years).fillna(
        s["securityTerm"].map(term_years))
    cand = s[(s["rem"] >= lo) & (s["rem"] <= hi) & (s["orig"] <= max_orig)
             & (s["issueDate"] <= asof) & (s["maturityDate"] > asof)]
    if cand.empty:
        return None

    rows = []
    for _, b in cand.iterrows():
        cf = conversion_factor(float(b["interestRate"]) / 100.0, first_delivery,
                               b["maturityDate"])
        if not np.isfinite(cf) or cf <= 0:
            continue
        ytm = curve_yield(yrow, float(b["rem"]))
        if not np.isfinite(ytm):
            continue
        p, dv = price_and_dv01(float(b["interestRate"]) / 100.0, float(b["rem"]), ytm)
        rows.append({"cusip": b["cusip"], "coupon": float(b["interestRate"]),
                     "maturity": b["maturityDate"], "rem": float(b["rem"]),
                     "cf": cf, "price": p, "dv01_pts": dv, "adj": p / cf})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # CTD proxy: cheapest on an adjusted-price basis (ignores financing, which is second-order
    # for a DV01 estimate and needs a repo curve we do not have).
    ctd = df.loc[df["adj"].idxmin()]
    fut_dv01 = ctd["dv01_pts"] / ctd["cf"] * FACE[symbol] / 100.0
    return {"symbol": symbol, "n_basket": len(df), "ctd_cusip": ctd["cusip"],
            "ctd_coupon": ctd["coupon"], "ctd_rem": ctd["rem"], "cf": ctd["cf"],
            "dv01": fut_dv01, "delivery": first_delivery.strftime("%Y-%m")}


def empirical_dv01(px: pd.DataFrame, ybp: pd.DataFrame, symbol: str,
                   window: str) -> tuple[float, float, int]:
    """Regress d(futures price) on d(CMT yield). Slope -> $ per bp. Returns (dv01, R2, n)."""
    cmt = LEG_CMT[symbol]
    if symbol not in px.columns or cmt not in ybp.columns:
        return np.nan, np.nan, 0
    d = pd.concat([px[symbol].diff().rename("dp"),
                   ybp[cmt].diff().rename("dy")], axis=1).dropna()
    d = d.loc[window:]
    d = d[(d["dy"].abs() > 0) & (d["dp"].abs() < 5)]
    if len(d) < 100:
        return np.nan, np.nan, len(d)
    slope, icept = np.polyfit(d["dy"], d["dp"], 1)
    pred = slope * d["dy"] + icept
    r2 = 1 - ((d["dp"] - pred) ** 2).sum() / ((d["dp"] - d["dp"].mean()) ** 2).sum()
    # slope is price-points per bp; negative because price falls as yield rises
    return float(-slope * FACE[symbol] / 100.0), float(r2), len(d)


def main() -> None:
    import yfinance as yf
    print("Fetching Treasury securities with coupons (cached after first run)...")
    secs = fetch_securities()
    yields = fetch_yields()
    ybp = yields * 100.0
    asof = yields.dropna(how="all").index[-1]
    yrow = yields.loc[asof]
    print(f"  {len(secs)} securities; curve as of {asof.date()}")
    print("  CMT: " + ", ".join(f"{t}y {yrow[t]:.2f}%" for t in (2, 5, 7, 10, 20, 30)
                                if t in yrow.index and np.isfinite(yrow[t])))

    yfmap = {"ZT": "ZT=F", "ZF": "ZF=F", "ZN": "ZN=F", "TN": "TN=F", "ZB": "ZB=F"}
    raw = yf.download(list(yfmap.values()), start="2004-01-01", end="2026-09-12",
                      auto_adjust=False, progress=False, group_by="column")["Close"]
    px = raw.rename(columns={v: k for k, v in yfmap.items()}).sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()

    print("\n" + "=" * 100)
    print("ANALYTICAL — cheapest-to-deliver today, and the DV01 it implies")
    print("=" * 100)
    print(f"  {'leg':5}{'basket':>8}{'CTD coupon':>12}{'CTD rem yrs':>13}{'conv factor':>13}"
          f"{'DV01 $/bp':>12}{'static':>9}{'ratio':>8}")
    ana = {}
    for sym in ("ZT", "ZF", "ZN", "TN", "ZB"):
        r = analytical_dv01(sym, asof, secs, yrow)
        if not r:
            print(f"  {sym:5}{'basket empty':>8}")
            continue
        ana[sym] = r["dv01"]
        st = CONTRACTS[sym].dv01
        print(f"  {sym:5}{r['n_basket']:>8}{r['ctd_coupon']:>11.3f}%{r['ctd_rem']:>13.2f}"
              f"{r['cf']:>13.4f}{r['dv01']:>12.1f}{st:>9.0f}{r['dv01'] / st:>8.2f}")

    print("\n" + "=" * 100)
    print("EMPIRICAL — regression of d(futures price) on d(CMT yield)")
    print("=" * 100)
    for label, start in (("full 2005-", "2005-01-01"), ("2015-", "2015-01-01"),
                         ("2022- (QT)", "2022-01-01")):
        print(f"\n  {label}")
        print(f"  {'leg':5}{'n':>7}{'R2':>8}{'DV01 $/bp':>12}{'static':>9}{'ratio':>8}"
              f"{'analytical':>12}")
        for sym in ("ZT", "ZF", "ZN", "TN", "ZB"):
            dv, r2, n = empirical_dv01(px, ybp, sym, start)
            if not np.isfinite(dv):
                continue
            st = CONTRACTS[sym].dv01
            a = ana.get(sym, np.nan)
            print(f"  {sym:5}{n:>7}{r2:>8.2f}{dv:>12.1f}{st:>9.0f}{dv / st:>8.2f}"
                  f"{a:>12.1f}")

    print("\n  RATIO < 1 means the static table OVERSTATES the contract's sensitivity, so the")
    print("  sleeve's modelled P&L per bp is too big AND its live position is sized too large.")
    print("  The empirical column is the one that pays: it needs no basket, no conversion")
    print("  factor and no CTD assumption, and it measures what the contract actually did.")

    out = ROOT / "results" / "ctd_dv01"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": s, "analytical": v, "static": CONTRACTS[s].dv01}
                  for s, v in ana.items()]).to_csv(out / "dv01.csv", index=False)
    print(f"\nWrote {out / 'dv01.csv'}")


if __name__ == "__main__":
    main()
