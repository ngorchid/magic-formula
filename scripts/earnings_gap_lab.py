"""Earnings-gap continuation (Appendix H.11): follow the opening gap on an earnings release, sector-hedged.

THE IDEA. Buy-on-gap (H.10) split cleanly in two: since 2020, gap-downs without news recover over
the day, gap-downs on news fall a further ~225bp. Post-earnings drift is the best-documented anomaly
that has not decayed. So: on the first session after an earnings release, go WITH the opening gap,
hedged with the stock's sector ETF, and hold for a few days.

EVENTS. Every 8-K with Item 2.02 ("Results of Operations") from EDGAR, Aug 2004 onwards, timed by
its acceptance time:
  before 09:30 ET  -> reaction day = that day (the release is known at the open)
  after  16:00 ET  -> reaction day = the next trading day
  09:30-16:00      -> excluded (the open does not contain the news; counted, not traded)
NB EDGAR's submissions JSON reports acceptanceDateTime as ET + twice the UTC offset, labelled "Z"
(checked 2026-10-17 against the filing index pages: JPM 06:30 ET shows as 14:30Z, AAPL 16:30 EST
as 02:30Z). Converting UTC->New York twice recovers ET; _to_et() asserts both known cases.

THE RULE (fixed before the first run):
  gap = open_d / close_{d-1} - 1 on the reaction day d; z = gap / sd of the previous 90 daily
  close-to-close returns. Trade if |z| >= 1, in the direction of the gap: long the stock and short
  the sector ETF on a gap up, the reverse on a gap down, equal dollars. Enter at the open of d,
  exit at the close of d + h, h in {0, 1, 3, 5} trading days.
  Not tradable, so excluded: raw price < $5, 20-day average dollar volume < $5m, and SHORTS on a
  gap of -10% or worse (Rule 201: a stock 10% down cannot be shorted on a downtick).

COSTS, as a fraction of the stock leg's notional, $5k per order:
  stock half-spread by 20-day average dollar volume: >= $500m 2bp, $100-500m 4bp, $20-100m 8bp,
  $5-20m 15bp; paid twice at the gapped open (spreads are wide there) and once at the close;
  IB tiered commission $0.0035/share, $0.35 minimum, both sides; ETF leg 1bp per side;
  borrow 0.3%/yr on a short stock leg.

CONTROL. Gaps of the same size (|z| >= 1) on days NOT within 3 trading days of an earnings
reaction day, for stocks whose earnings dates are known on that day (inside their 8-K history),
same rule and costs. If earnings gaps continue and other gaps revert, the two must differ.

UNIVERSES.
  A  S&P 500, point in time (fja05680): a stock is eligible only on days it was a member.
     Former members whose ticker SEC no longer maps to a CIK have no events (coverage reported).
  B  current S&P 400 + 600 (mid and small caps), where drift is documented as strongest.
     SURVIVORSHIP-BIASED: today's members only. Firms that failed after bad news are missing,
     which understates continuation on the short side, so B is if anything conservative there.
  Sector = current GICS sector from today's lists (SPDR sector ETF; IYR/VOX before XLRE/XLC
  existed); unknown (former members) -> SPY.

PRIMARY TEST (fixed before the first run; everything else is descriptive):
  Universe A, both directions, h = 3, net of costs, design window 2005-2022.
  PASS = mean net > 0 with date-clustered t >= 2, AND net > 0 in 2020-2022 on its own, AND the
  event mean beats the control mean (t >= 2).
HOLDOUT. 2023 onwards is not printed unless --holdout is passed, once, after the verdict.

Usage:
  python scripts/earnings_gap_lab.py fetch        # EDGAR 8-K Item 2.02 events (cached)
  python scripts/earnings_gap_lab.py run [--holdout]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.loaders import download_ohlcv  # noqa: E402
from data.universe import (sp1500_constituents, sp500_pit_eligible,  # noqa: E402
                           sp500_pit_universe)

START, END = "2004-01-01", "2026-09-01"
DESIGN_END, HOLDOUT_START = "2022-12-31", "2023-01-01"
PERIODS = {"2005-2012": ("2005-01-01", "2012-12-31"), "2013-2019": ("2013-01-01", "2019-12-31"),
           "2020-2022": ("2020-01-01", "2022-12-31")}
HOLDOUT = {"2023+ (holdout)": (HOLDOUT_START, END)}
Z_MIN, SD_WIN, ADV_WIN = 1.0, 90, 20
HORIZONS = (0, 1, 3, 5)
PRIMARY_H = 3
MIN_PX, MIN_ADV, SSR_GAP = 5.0, 5e6, -0.10
ORDER, COMM_SH, COMM_MIN, ETF_HS_BP, BORROW = 5_000.0, 0.0035, 0.35, 1.0, 0.003
CONTROL_GAP_DAYS = 3
EV_CACHE = ROOT / "data" / "cache" / "earnings_8k"
OUT = ROOT / "results" / "earnings_gap"
SECTOR_ETF = {"Information Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
              "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
              "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
              "Real Estate": "IYR", "Communication Services": "VOX"}


# ------------------------------------------------------------------------------------ EDGAR
def _ua() -> str:
    ua = os.getenv("SEC_USER_AGENT")
    env = ROOT / ".env"
    if not ua and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("SEC_USER_AGENT="):
                ua = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not ua:
        raise SystemExit("SEC_USER_AGENT not set")
    return ua


SESSION = requests.Session()


def _get(url: str) -> dict:
    for attempt in range(4):
        time.sleep(0.12)                               # under SEC's 10 requests/s
        r = SESSION.get(url, headers={"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"},
                        timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return {}
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return {}


def _to_et(stamp: str) -> pd.Timestamp:
    """EDGAR's acceptanceDateTime -> naive ET (see the module note: convert UTC->NY twice)."""
    t = pd.Timestamp(stamp)
    t = t.tz_convert("America/New_York").tz_localize(None).tz_localize("UTC")
    return t.tz_convert("America/New_York").tz_localize(None)


assert _to_et("2026-07-14T14:30:38.000Z") == pd.Timestamp("2026-07-14 06:30:38")   # JPM, EDT
assert _to_et("2026-01-30T02:30:33.000Z") == pd.Timestamp("2026-01-29 16:30:33")   # AAPL, EST


def cik_map() -> dict[str, int]:
    data = _get("https://www.sec.gov/files/company_tickers.json")
    out = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}
    out.update({"MMC": 62709, "XOM": 34088, "K": 55067, "WBA": 1618921})
    return out


def _filings(block: dict) -> pd.DataFrame:
    keys = ("form", "filingDate", "acceptanceDateTime", "items")
    if not block or "form" not in block:
        return pd.DataFrame(columns=keys)
    return pd.DataFrame({k: block.get(k, [""] * len(block["form"])) for k in keys})


def fetch_cik(cik: int) -> pd.DataFrame:
    path = EV_CACHE / f"{cik}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    frames = [_filings(sub.get("filings", {}).get("recent", {}))]
    for f in sub.get("filings", {}).get("files", []):
        if f.get("filingTo", "9999") >= "2004-08-01":
            frames.append(_filings(_get(f"https://data.sec.gov/submissions/{f['name']}")))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["form"] == "8-K") & df["items"].astype(str).str.contains("2.02", regex=False)]
    df = df[df["filingDate"] >= "2004-08-01"].reset_index(drop=True)
    EV_CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def universe_tickers() -> tuple[list[str], list[str], pd.Series]:
    a = sp500_pit_universe(START, END)
    c = sp1500_constituents()
    b = sorted(c.loc[c["tier"].isin(["mid", "small"]), "ticker"].tolist())
    sectors = c.set_index("ticker")["sector"]
    return a, b, sectors


def fetch() -> None:
    a, b, _ = universe_tickers()
    cmap = cik_map()
    names = sorted(set(a) | set(b))
    mapped = [t for t in names if t in cmap]
    print(f"{len(names)} tickers, {len(mapped)} map to a CIK; fetching 8-K Item 2.02 filings …")
    failed = []
    for k, t in enumerate(mapped, 1):
        try:
            fetch_cik(cmap[t])
        except requests.RequestException as e:
            failed.append(t)
            print(f"  {t}: {e.__class__.__name__}; skipped (rerun fetch to retry)")
        if k % 200 == 0:
            print(f"  {k}/{len(mapped)}", flush=True)
    print(f"done; {len(failed)} failed: {failed}")


def events(tickers: list[str], cmap: dict[str, int], cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """One row per (ticker, reaction day); plus counts of what was dropped."""
    rows, n_intraday, n_total = [], 0, 0
    for t in tickers:
        if t not in cmap or not (EV_CACHE / f"{cmap[t]}.parquet").exists():
            continue
        df = pd.read_parquet(EV_CACHE / f"{cmap[t]}.parquet")
        for stamp in df["acceptanceDateTime"]:
            n_total += 1
            et = _to_et(stamp)
            hm = et.hour * 60 + et.minute
            day = et.normalize()
            if hm < 9 * 60 + 30:
                i = cal.searchsorted(day, side="left")
            elif hm >= 16 * 60:
                i = cal.searchsorted(day, side="right")
            else:
                n_intraday += 1
                continue
            if i < len(cal):
                rows.append((t, cal[i]))
    ev = pd.DataFrame(rows, columns=["ticker", "day"]).drop_duplicates()
    # several 2.02 filings within a week (amendments, pre-announcements): keep the first
    ev = ev.sort_values(["ticker", "day"])
    keep = ev.groupby("ticker")["day"].diff().dt.days.fillna(99) > 7
    return ev[keep].reset_index(drop=True), {"filings": n_total, "intraday": n_intraday}


# -------------------------------------------------------------------------------------- run
def half_spread_bp(adv: pd.DataFrame) -> pd.DataFrame:
    hs = pd.DataFrame(np.nan, index=adv.index, columns=adv.columns)
    for lo, bp in ((5e6, 15.0), (20e6, 8.0), (100e6, 4.0), (500e6, 2.0)):
        hs = hs.mask(adv >= lo, bp)
    return hs


def build(tickers: list[str], sectors: pd.Series, member: str, cmap: dict[str, int]) -> pd.DataFrame:
    panel = download_ohlcv(tickers, START, END)
    close_raw = panel["close"].sort_index()
    f = (panel["adj_close"].reindex_like(close_raw) / close_raw).where(lambda x: x > 0)
    o = panel["open"].reindex_like(close_raw) * f
    c = close_raw * f
    vol = panel["volume"].reindex_like(close_raw)
    keep = c.notna().sum() > SD_WIN
    o, c, close_raw, vol = o.loc[:, keep], c.loc[:, keep], close_raw.loc[:, keep], vol.loc[:, keep]
    cal = c.index

    etfs = sorted(set(SECTOR_ETF.values()) | {"SPY"})
    ep = download_ohlcv(etfs, START, END)
    ef = ep["adj_close"] / ep["close"]
    eo, ec = (ep["open"] * ef).reindex(cal), ep["adj_close"].reindex(cal)
    etf_of = {t: SECTOR_ETF.get(str(sectors.get(t, "")), "SPY") for t in c.columns}
    EO = pd.DataFrame({t: eo[etf_of[t]] for t in c.columns})
    EC = pd.DataFrame({t: ec[etf_of[t]] for t in c.columns})

    elig = (sp500_pit_eligible(cal, list(c.columns)).astype(bool) if member == "pit"
            else pd.DataFrame(True, index=cal, columns=c.columns))
    sd = c.pct_change(fill_method=None).rolling(SD_WIN, min_periods=SD_WIN).std().shift(1)
    gap = o / c.shift(1) - 1.0
    z = gap / sd
    adv = (close_raw * vol).rolling(ADV_WIN, min_periods=ADV_WIN).mean().shift(1)
    hs = half_spread_bp(adv)
    raw_open = o / f.reindex_like(o).loc[:, o.columns]
    sgn = np.sign(gap)
    tradable = (elig & (z.abs() >= Z_MIN) & (close_raw.shift(1) >= MIN_PX) & (adv >= MIN_ADV)
                & hs.notna() & ~((sgn < 0) & (gap <= SSR_GAP)))

    ev, counts = events(list(c.columns), cmap, cal)
    E = pd.DataFrame(False, index=cal, columns=c.columns)
    for t, d in ev.itertuples(index=False):
        E.at[d, t] = True
    near = E.astype(float).rolling(2 * CONTROL_GAP_DAYS + 1, center=True, min_periods=1).max() > 0
    span = pd.DataFrame(False, index=cal, columns=c.columns)
    for t, g in ev.groupby("ticker"):
        span.loc[g["day"].min():g["day"].max(), t] = True
    ctl = tradable & span & ~near
    evm = tradable & E
    n_cov = len({t for t in c.columns if t in cmap})
    print(f"  {c.shape[1]} stocks with prices, {n_cov} with a CIK; 8-K 2.02 filings {counts['filings']}, "
          f"intraday releases excluded {counts['intraday']}; reaction days {len(ev)}, "
          f"tradable (|z|>={Z_MIN:g}) {int(evm.values.sum())}; control gaps {int(ctl.values.sum())}")

    comm = 2 * np.maximum(COMM_MIN, COMM_SH * ORDER / raw_open) / ORDER
    rows = []
    for h in HORIZONS:
        r_stock = c.shift(-h) / o - 1.0
        r_etf = EC.shift(-h) / EO - 1.0
        gross = sgn * (r_stock - r_etf)
        cost = (3 * hs / 1e4 + comm + 2 * ETF_HS_BP / 1e4
                + (sgn < 0) * BORROW * (h + 1) / 252)
        for kind, mask in (("event", evm), ("control", ctl)):
            m = mask & gross.notna()
            s = pd.DataFrame({"gross": gross.where(m).stack(), "cost": cost.where(m).stack(),
                              "side": sgn.where(m).stack(), "z": z.where(m).stack()})
            s["h"], s["kind"] = h, kind
            rows.append(s)
    out = pd.concat(rows)
    out.index.names = ["day", "ticker"]
    out["net"] = out["gross"] - out["cost"]
    return out.reset_index()


def _t_clustered(df: pd.DataFrame, col: str) -> float:
    """t of the trade-weighted mean, standard error clustered by reaction day (events bunch in
    earnings season, so trades on one day are not independent)."""
    x = df[col]
    dev = (x - x.mean()).groupby(df["day"]).sum()
    se = np.sqrt((dev ** 2).sum()) / len(x)
    return x.mean() / se if len(dev) > 2 and se > 0 else np.nan


def report(tr: pd.DataFrame, arm: str, periods: dict) -> dict:
    print(f"\n=== {arm} ===")
    hdr = f"{'kind':8}{'side':6}{'h':>3}{'period':>17}{'n':>7}{'gross bp':>10}{'net bp':>9}{'t(net)':>8}{'win':>6}"
    print(hdr)
    res = {}
    allp = {"design 2005-2022": ("2005-01-01", DESIGN_END), **periods}
    for kind in ("event", "control"):
        for h in HORIZONS:
            for side, sv in (("both", None), ("long", 1.0), ("short", -1.0)):
                if side != "both" and h != PRIMARY_H:
                    continue
                for p, (a, b) in allp.items():
                    x = tr[(tr.kind == kind) & (tr.h == h) & (tr.day >= a) & (tr.day <= b)]
                    if sv is not None:
                        x = x[x.side == sv]
                    if len(x) < 5:
                        continue
                    t = _t_clustered(x, "net")
                    res[(kind, side, h, p)] = (x["net"].mean(), t, len(x))
                    print(f"{kind:8}{side:6}{h:>3}{p:>17}{len(x):>7}{x.gross.mean() * 1e4:>+10.1f}"
                          f"{x.net.mean() * 1e4:>+9.1f}{t:>+8.2f}{(x.net > 0).mean():>6.0%}")
            print()
    return res


def verdict(tr: pd.DataFrame, res: dict) -> None:
    p = "design 2005-2022"
    m, t, n = res[("event", "both", PRIMARY_H, p)]
    m20 = res.get(("event", "both", PRIMARY_H, "2020-2022"), (np.nan, np.nan, 0))[0]
    x = tr[(tr.h == PRIMARY_H) & (tr.day <= DESIGN_END)]
    e, c_ = x[x.kind == "event"]["net"], x[x.kind == "control"]["net"]
    td = (e.mean() - c_.mean()) / np.sqrt(e.var() / len(e) + c_.var() / len(c_))
    ok = (m > 0 and t >= 2) and m20 > 0 and td >= 2
    print(f"PRIMARY (A, both sides, h={PRIMARY_H}, net, 2005-2022): mean {m * 1e4:+.1f}bp, t {t:+.2f}, n {n}; "
          f"2020-2022 {m20 * 1e4:+.1f}bp; event minus control {(e.mean() - c_.mean()) * 1e4:+.1f}bp "
          f"(t {td:+.2f})  ->  {'PASS' if ok else 'FAIL'}")


def run(holdout: bool) -> None:
    a, b, sectors = universe_tickers()
    cmap = cik_map()
    OUT.mkdir(parents=True, exist_ok=True)
    periods = {**PERIODS, **(HOLDOUT if holdout else {})}
    results = {}
    for arm, tick, member in (("A  S&P 500, point in time", a, "pit"),
                              ("B  S&P 400+600, current members (survivorship-biased)", b, "all")):
        print(f"[{arm}] building …")
        tr = build(tick, sectors, member, cmap)
        if not holdout:
            tr = tr[tr.day <= DESIGN_END]
        tr.to_parquet(OUT / f"trades_{arm[0]}{'_holdout' if holdout else ''}.parquet")
        results[arm] = (tr, report(tr, arm, periods))
    tr, res = results["A  S&P 500, point in time"]
    print()
    verdict(tr, res)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    if stage == "fetch":
        fetch()
    else:
        run("--holdout" in sys.argv)
