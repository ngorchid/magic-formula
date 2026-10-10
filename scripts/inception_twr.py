"""Inception-to-date account TWR (IBKR's own figure) and the S&P 500 over the same window.

WHY. The book summary's "Account TWR" is the daily Flex statement's ROLLING 7-day window. The
since-inception figure needs a statement over the whole life of the account; the Flex Web
Service gives one for the same query when SendRequest carries fd/td (from/to date, yyyymmdd).
Verified 2026-10-10: fd=20260813 td=20261009 -> ChangeInNAV twr 1.4596% (= 50,729.78 / 50,000
deposited; no other flows), and a td on a Sunday is clamped by IB to the Friday.

RANGE LIMIT. One request covers at most 365 days. Longer histories are split into consecutive
segments and their TWRs chained, (1+r1)(1+r2)...-1 -- which is how a TWR composes.

BENCHMARK. ^SP500TR (S&P 500 Total Return: dividends reinvested, no fees) from yfinance, close on
the inception day -> close on IB's toDate. The account held only cash on 13 Aug (deposit day, no
trades), so both series start from the 13 Aug close. The account TWR is net of all costs and
covers every sleeve (equities, futures overlay, options), so this is a reference, not a
like-for-like index.

STORAGE (inside the IB-records folder, NOT raw/ or tables/: the ITD statement repeats what the
daily statements already archive, and its ChangeInNAV must not enter tables/change_in_nav.csv,
where it would replace the weekly headline):
  itd/latest.xml        the most recent ITD statement, overwritten each run
  itd/twr_history.csv   append-only, one row per run -- the book summary reads the last row

Run standalone: python scripts/inception_twr.py   (download_ib_records.py also calls update())
"""
from __future__ import annotations

import csv
import os
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

INCEPTION = os.getenv("ACCOUNT_INCEPTION", "20260813")   # the day the $50,000 deposit landed
MAX_DAYS = 365                                            # Flex Web Service fd/td limit
BENCH = "^SP500TR"
FIELDS = ["asof", "fromDate", "toDate", "twr", "startingValue", "endingValue",
          "depositsWithdrawals", "segments", "bench", "bench_from", "bench_to", "bench_ret"]


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def segments(start: date, end: date) -> list[tuple[date, date]]:
    """Consecutive [from, to] windows of at most MAX_DAYS days covering start..end."""
    out, s = [], start
    while s <= end:
        e = min(end, s + timedelta(days=MAX_DAYS - 1))
        out.append((s, e))
        s = e + timedelta(days=1)
    return out


def change_in_nav(data: bytes, account: str) -> dict:
    """The single ChangeInNAV row of a statement, checked for the account and a TWR."""
    root = ET.fromstring(data)
    accts = {s.get("accountId") for s in root.findall(".//FlexStatement")}
    if account and accts != {account}:
        raise ValueError(f"statement is for {accts}, expected {account}")
    rows = root.findall(".//ChangeInNAV")
    if len(rows) != 1 or rows[0].get("twr") in (None, ""):
        raise ValueError(f"expected one ChangeInNAV with a twr, got {len(rows)}")
    return dict(rows[0].attrib)


def chain(rets_pct: list[float]) -> float:
    g = 1.0
    for r in rets_pct:
        g *= 1 + r / 100
    return (g - 1) * 100


def itd(fetch, start: date, end: date, account: str) -> tuple[dict, bytes]:
    """fetch(fd, td) -> statement bytes. Returns (summary row, the last segment's statement)."""
    navs, data = [], b""
    for s, e in segments(start, end):
        data = fetch(s.strftime("%Y%m%d"), e.strftime("%Y%m%d"))
        navs.append(change_in_nav(data, account))
    return {"fromDate": navs[0]["fromDate"], "toDate": navs[-1]["toDate"],
            "twr": round(chain([float(n["twr"]) for n in navs]), 6),
            "startingValue": navs[0].get("startingValue", ""), "endingValue": navs[-1].get("endingValue", ""),
            "depositsWithdrawals": round(sum(float(n.get("depositsWithdrawals") or 0) for n in navs), 2),
            "segments": len(navs)}, data


def bench_return(closes: list[tuple[str, float]], frm: str, to: str) -> tuple[str, str, float] | None:
    """Return over the last close on/before frm -> the last close on/before to (dates yyyymmdd).
    None when either end is missing or the window is empty."""
    closes = sorted(closes)
    a = [c for c in closes if c[0] <= frm]
    b = [c for c in closes if c[0] <= to]
    if not a or not b or b[-1][0] <= a[-1][0]:
        return None
    return a[-1][0], b[-1][0], round((b[-1][1] / a[-1][1] - 1) * 100, 6)


def sp500_closes(frm: str, to: str) -> list[tuple[str, float]]:
    import yfinance as yf
    h = yf.Ticker(BENCH).history(start=(_d(frm) - timedelta(days=10)).isoformat(),
                                 end=(_d(to) + timedelta(days=1)).isoformat(), auto_adjust=True)["Close"]
    return [(ix.strftime("%Y%m%d"), float(v)) for ix, v in h.items() if v == v]


def update(records: Path, fetch, account: str, today: date | None = None, closes=sp500_closes) -> dict:
    """Compute the ITD row, write itd/latest.xml + append itd/twr_history.csv. The benchmark is
    best effort: a failed price download leaves its columns blank, never the TWR."""
    today = today or date.today()
    row, data = itd(fetch, _d(INCEPTION), today - timedelta(days=1), account)
    row = {"asof": today.isoformat(), **row, "bench": BENCH}
    try:
        b = bench_return(closes(row["fromDate"], row["toDate"]), row["fromDate"], row["toDate"])
    except Exception as e:  # noqa: BLE001
        print(f"benchmark unavailable: {type(e).__name__}: {e}")
        b = None
    row.update(dict(zip(("bench_from", "bench_to", "bench_ret"), b)) if b else
               {"bench_from": "", "bench_to": "", "bench_ret": ""})
    out = records / "itd"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.xml").write_bytes(data)
    hist = out / "twr_history.csv"
    new = not hist.exists()
    with open(hist, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def latest(records: Path) -> dict | None:
    """The last row of itd/twr_history.csv, or None."""
    p = records / "itd" / "twr_history.csv"
    try:
        with open(p, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("twr") not in (None, "")]
        return rows[-1] if rows else None
    except (OSError, csv.Error):
        return None


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import download_ib_records as d  # loads the live .env
    tok, qid = os.getenv("FLEX_TOKEN"), os.getenv("FLEX_QUERY_ID")
    r = update(d.RECORDS, lambda fd, td: d.download(tok, qid, extra={"fd": fd, "td": td}), d.EXPECT_ACCOUNT)
    print(r)
