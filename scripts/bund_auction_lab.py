"""Does the Treasury auction concession TRANSFER to German Bunds?

We validated the mechanism in US Treasuries (`treasury_auction_lab.py`): dealers are required
to bid, pre-hedge by selling the sector, and the auctioned point cheapens into the auction then
reverts. This asks whether the same thing happens at the Bundesrepublik's auctions.

WHY IT MIGHT: identical structure — a price-insensitive, calendar-bound sovereign seller and a
mandated primary-dealer group (the Bund Issues Auction Group) that must bid.

WHY IT MIGHT NOT: the Bundesbank RETAINS part of every auction for secondary-market operations
(the "Marktpflegequote"), so dealers are not stuffed with the full size the way US primaries
are. If the effect is really about forced inventory absorption, retention should blunt it.

Data, both keyless:
  yields    ECB AAA euro-area government spot curve (Svensson-fitted, daily). Same caveat as
            FRED CMT — a fitted curve, not tradeable prices — so this is an existence test.
  auctions  Deutsche Finanzagentur "Auction Results since 1999" XLSX.

Tradeable expression (Eurex, and NB Lynx charges EUR 2.00/contract vs USD 4.00 for US futures):
  Bobl 5Y auction  -> 2s5s10s  = FGBS / FGBM / FGBL
  Bund 10Y auction -> 5s10s30s = FGBM / FGBL / FGBX
Unlike the US (where ZN's CTD sits at ~7y, not 10y), the Eurex contracts sit close to their
nominal points, so the mapping is cleaner.

Run: python scripts/bund_auction_lab.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache" / "bund"
OUT = ROOT / "results" / "bund_auction"

ECB_URL = ("https://data-api.ecb.europa.eu/service/data/YC/"
           "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{n}Y?format=csvdata")
DE_AUCTIONS = ("https://www.deutsche-finanzagentur.de/fileadmin/user_upload/"
               "Institutionelle-investoren/auktionen/emissionshistorie_en.xlsx")

TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]

# auctioned segment -> (short wing, belly, long wing) on the ECB curve
WINGS = {2: (1, 2, 5), 5: (2, 5, 10), 10: (5, 10, 30), 30: (10, 30, None)}
FUTURES = {2: "FGBS", 5: "FGBM", 10: "FGBL", 30: "FGBX"}


def fetch_yields() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "ecb_aaa_curve.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [int(c) for c in df.columns]
        return df
    out = {}
    for n in TENORS:
        r = requests.get(ECB_URL.format(n=n), timeout=120)
        d = pd.read_csv(io.StringIO(r.text))
        s = pd.Series(pd.to_numeric(d["OBS_VALUE"], errors="coerce").to_numpy(),
                      index=pd.to_datetime(d["TIME_PERIOD"]))
        out[n] = s[~s.index.duplicated()]
        print(f"  ECB {n:>2}Y: {s.notna().sum()} obs", end="\r")
    df = pd.DataFrame(out).sort_index().dropna(how="all")
    df.to_csv(path)
    print(f"\n  cached ECB curve {df.index.min().date()}..{df.index.max().date()}")
    return df


def fetch_auctions() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "de_auctions.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    raw = requests.get(DE_AUCTIONS, timeout=180).content
    (CACHE / "de_auctions.xlsx").write_bytes(raw)
    df = pd.read_excel(io.BytesIO(raw), header=None)
    # header block is rows 7-10; data starts where col 0 is a running integer
    df = df[pd.to_numeric(df[0], errors="coerce").notna()].copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(df[1], errors="coerce"),
        "isin": df[2].astype(str),
        "bond": df[3].astype(str).str.strip(),
        "segment_raw": df[6].astype(str).str.strip(),
        "volume": pd.to_numeric(df[7], errors="coerce"),
        "process": df[9].astype(str).str.strip(),
        "retention": pd.to_numeric(df[17], errors="coerce"),
    }).dropna(subset=["date"])

    def seg(s: str) -> float | None:
        try:
            v = float(str(s).upper().replace("Y", "").strip())
        except ValueError:
            return None
        return min(WINGS, key=lambda c: abs(c - v)) if v > 0 else None

    out["segment"] = out["segment_raw"].map(seg)
    out = out.dropna(subset=["segment"])
    out["segment"] = out["segment"].astype(int)
    out.to_csv(path, index=False)
    print(f"  cached {len(out)} German auctions "
          f"{out.date.min().date()}..{out.date.max().date()}")
    return out


def fly(bp: pd.DataFrame, lo: int, b: int, hi: int) -> pd.Series:
    """Slope-neutral, distance-weighted — same construction as the US lab, so the numbers
    are directly comparable (belly-equivalent bp)."""
    w1 = (hi - b) / (hi - lo)
    return (bp[b] - w1 * bp[lo] - (1 - w1) * bp[hi]).dropna()


def snapback(v: pd.Series, dates, piv: int = -1, ex: int = 3) -> np.ndarray:
    idx = v.index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    out = []
    for d in dates:
        i = pos.get(d)
        if i is None:
            nxt = idx[idx > d]
            if not len(nxt):
                continue
            i = pos[nxt[0]]
        i = int(i)
        if i + piv < 0 or i + ex >= len(idx):
            continue
        w = v.iloc[i + piv:i + ex + 1]
        if w.isna().any():
            continue
        out.append(-(w.iloc[-1] - w.iloc[0]))     # long belly: profit when the fly falls
    return np.array(out)


def tstat(x: np.ndarray) -> float:
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() else np.nan


def main() -> None:
    print("Fetching ECB AAA curve...")
    bp = fetch_yields() * 100.0
    print("Fetching Finanzagentur auction history...")
    au = fetch_auctions()
    au = au[au["date"] >= bp.index.min()]

    print(f"\n  curve {bp.index.min().date()}..{bp.index.max().date()}, "
          f"{len(au)} auctions in range")
    print("  by segment: " + ", ".join(
        f"{int(k)}y={v}" for k, v in au["segment"].value_counts().sort_index().items()))

    # --- do German auctions cluster the way US ones do? (drove the 2y being untradeable there)
    print("\n" + "=" * 96)
    print("AUCTION CLUSTERING — days from each auction to the nearest auction of the wing tenor")
    print("=" * 96)
    for seg in sorted(au["segment"].unique()):
        if seg not in WINGS or WINGS[seg][2] is None:
            continue
        lo, _, hi = WINGS[seg]
        d0 = sorted(au[au["segment"] == seg]["date"])
        line = f"  {int(seg):>2}y belly:"
        for tag, wing in (("lo", lo), ("hi", hi)):
            dw = sorted(au[au["segment"] == wing]["date"])
            if not dw or not d0:
                line += f"   {tag}={wing}y n/a"
                continue
            gaps = [min(abs((x - h).days) for h in dw) for x in d0]
            line += f"   {tag}={wing}y median {np.median(gaps):>4.0f}d"
        print(line)

    # --- the kill switch
    print("\n" + "=" * 96)
    print("SNAPBACK T-1 -> T+3, by segment (belly-equivalent bp; US comparison in brackets)")
    print("=" * 96)
    US = {5: "US 5y: +0.99 t=+7.5", 10: "US 10y: +1.37 t=+8.1"}
    print(f"  {'seg':>4s} {'future':7s} {'fly':10s} {'n':>4s} {'gross bp':>9s} {'sd':>6s} "
          f"{'t':>6s}   {'US benchmark':22s}")
    print("  " + "-" * 92)
    results = {}
    for seg in sorted(WINGS):
        lo, b, hi = WINGS[seg]
        if hi is None or any(t not in bp.columns for t in (lo, b, hi)):
            continue
        rows = au[au["segment"] == seg]
        if len(rows) < 20:
            continue
        v = fly(bp, lo, b, hi)
        r = snapback(v, list(rows["date"]))
        if len(r) < 20:
            continue
        results[seg] = r
        print(f"  {seg:>3}y {FUTURES[seg]:7s} {f'{lo}s{b}s{hi}s':10s} {len(r):>4d} "
              f"{r.mean():>+9.2f} {r.std():>6.2f} {tstat(r):>+6.2f}   {US.get(seg,''):22s}")

    if not results:
        print("\n  nothing testable")
        return

    # --- shape: is there a hump at all?
    print("\n" + "=" * 96)
    print("EVENT SHAPE — mean cumulative change from T-5 (bp). Should RISE into T-1, fall after")
    print("=" * 96)
    days = list(range(-5, 6))
    print(f"  {'seg':>4s} " + "".join(f"{('T'+str(d)):>8s}" for d in days))
    for seg in results:
        lo, b, hi = WINGS[seg]
        v = fly(bp, lo, b, hi)
        idx = v.index
        pos = pd.Series(np.arange(len(idx)), index=idx)
        paths = []
        for d in au[au["segment"] == seg]["date"]:
            i = pos.get(d)
            if i is None:
                nxt = idx[idx > d]
                if not len(nxt):
                    continue
                i = pos[nxt[0]]
            i = int(i)
            if i - 5 < 0 or i + 5 >= len(idx):
                continue
            w = v.iloc[i - 5:i + 6]
            if w.isna().any():
                continue
            paths.append(w.to_numpy() - w.iloc[0])
        if paths:
            m = np.mean(paths, axis=0)
            print(f"  {seg:>3}y " + "".join(f"{x:>+8.2f}" for x in m))

    # --- the mechanism check that decides it: retention
    print("\n" + "=" * 96)
    print("MECHANISM — Bundesbank RETENTION should BLUNT the effect")
    print("=" * 96)
    print("  If the concession is compensation for forced dealer inventory, auctions where the")
    print("  Bundesbank retained MORE (leaving dealers less to absorb) should show LESS effect.")
    print(f"\n  {'seg':>4s} {'high retention':>22s} {'low retention':>22s}")
    print(f"  {'':4s} {'gross':>10s} {'t':>11s} {'gross':>10s} {'t':>11s}")
    print("  " + "-" * 50)
    for seg in results:
        lo, b, hi = WINGS[seg]
        v = fly(bp, lo, b, hi)
        rows = au[au["segment"] == seg].copy()
        rows["ret_pct"] = rows["retention"] / rows["volume"]
        rows = rows.dropna(subset=["ret_pct"])
        if len(rows) < 40:
            continue
        med = rows["ret_pct"].median()
        hi_r = snapback(v, list(rows[rows.ret_pct > med]["date"]))
        lo_r = snapback(v, list(rows[rows.ret_pct <= med]["date"]))
        if len(hi_r) < 15 or len(lo_r) < 15:
            continue
        print(f"  {seg:>3}y {hi_r.mean():>+10.2f} {tstat(hi_r):>+11.2f} "
              f"{lo_r.mean():>+10.2f} {tstat(lo_r):>+11.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{k}y": pd.Series(v) for k, v in results.items()}).to_csv(
        OUT / "snapback_pnl.csv", index=False)
    print(f"\n  wrote {OUT}/snapback_pnl.csv")
    print("\n  CAVEAT: the ECB AAA curve is Svensson-fitted euro-area AAA, not Bund cash or")
    print("  futures prices — an existence test, exactly like CMT was for the US.")


if __name__ == "__main__":
    main()
