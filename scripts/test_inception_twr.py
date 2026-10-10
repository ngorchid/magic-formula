"""Tests for scripts/inception_twr.py: the inception-to-date account TWR (IBKR, via fd/td Flex
requests) and the S&P 500 total-return benchmark over the same window.
scripts/mutate_inception_twr.py seeds the faults. No IB, no network.

Run: python scripts/test_inception_twr.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass
import inception_twr as it  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        return f"crashed: {type(e).__name__}: {e}"


def stmt(frm, to, twr, start="0", end="50729.78", dep="50000", acct="U27760647", n=1) -> bytes:
    nav = "".join(f'<ChangeInNAV accountId="{acct}" fromDate="{frm}" toDate="{to}" startingValue="{start}" '
                  f'endingValue="{end}" depositsWithdrawals="{dep}" twr="{twr}"/>' for _ in range(n))
    return (f'<FlexQueryResponse><FlexStatements count="1"><FlexStatement accountId="{acct}" fromDate="{frm}" '
            f'toDate="{to}">{nav}</FlexStatement></FlexStatements></FlexQueryResponse>').encode()


print("SEGMENTS (Flex fd/td covers at most 365 days)")
s = safe(it.segments, date(2026, 8, 13), date(2026, 10, 9))
check("a short history is one request", s == [(date(2026, 8, 13), date(2026, 10, 9))], str(s))
s = safe(it.segments, date(2026, 8, 13), date(2027, 8, 13))
check("366 days -> two windows of 365 + 1, contiguous",
      s == [(date(2026, 8, 13), date(2027, 8, 12)), (date(2027, 8, 13), date(2027, 8, 13))], str(s))
s = safe(it.segments, date(2026, 8, 13), date(2028, 9, 1))
check("no window exceeds 365 days and none overlaps or leaves a gap",
      isinstance(s, list) and all((e - b).days + 1 <= 365 for b, e in s)
      and all((s[i + 1][0] - s[i][1]).days == 1 for i in range(len(s) - 1)) and s[-1][1] == date(2028, 9, 1), str(s))

print("\nCHAIN")
check("TWRs compose multiplicatively: +10% then -10% = -1%", abs(it.chain([10, -10]) + 1.0) < 1e-9,
      str(it.chain([10, -10])))
check("one segment is returned as is", abs(it.chain([1.4596]) - 1.4596) < 1e-9, "")

print("\nCHANGE IN NAV")
row = safe(it.change_in_nav, stmt("20260813", "20261009", "1.459564966"), "U27760647")
check("the statement's twr is read", isinstance(row, dict) and row.get("twr") == "1.459564966", str(row))
check("another account's statement is refused",
      "crashed: ValueError" in str(safe(it.change_in_nav, stmt("20260813", "20261009", "1", acct="U999"), "U27760647")), "")
check("no twr -> refused (never read as 0%)",
      "crashed: ValueError" in str(safe(it.change_in_nav, stmt("20260813", "20261009", ""), "U27760647")), "")
check("two ChangeInNAV rows -> refused (ambiguous)",
      "crashed: ValueError" in str(safe(it.change_in_nav, stmt("20260813", "20261009", "1", n=2), "U27760647")), "")

print("\nITD OVER SEVERAL SEGMENTS")
calls = []


def fetch(fd, td):
    calls.append((fd, td))
    return {"20260813": stmt("20260813", "20270812", "10", start="0", end="55000", dep="50000"),
            "20270813": stmt("20270813", "20270915", "-10", start="55000", end="49500", dep="0")}[fd]


r = safe(it.itd, fetch, date(2026, 8, 13), date(2027, 9, 15), "U27760647")
ok = isinstance(r, tuple)
check("each segment is requested with its own fd/td",
      calls == [("20260813", "20270812"), ("20270813", "20270915")], str(calls))
check("segment TWRs are chained (+10%, -10% -> -1%)", ok and abs(r[0]["twr"] + 1.0) < 1e-6, str(r)[:200])
check("the window runs from the first segment's start to the last segment's end",
      ok and (r[0]["fromDate"], r[0]["toDate"]) == ("20260813", "20270915"), str(r)[:200])
check("ending value is the last segment's; deposits are summed",
      ok and r[0]["endingValue"] == "49500" and r[0]["depositsWithdrawals"] == 50000 and r[0]["segments"] == 2,
      str(r)[:300])

print("\nBENCHMARK")
closes = [("20260812", 100.0), ("20260813", 110.0), ("20261008", 120.0), ("20261009", 121.0), ("20261012", 200.0)]
b = it.bench_return(closes, "20260813", "20261009")
check("from the inception-day close to the toDate close", b == ("20260813", "20261009", 10.0), str(b))
b = it.bench_return(closes, "20260815", "20261011")
check("weekend ends roll back to the last close on/before (never a later close)",
      b == ("20260813", "20261009", 10.0), str(b))
check("no close on/before the start -> None", it.bench_return(closes, "20260801", "20261009") is None, "")
check("no close after the start -> None (not a 0% return)",
      it.bench_return([("20260813", 110.0)], "20260813", "20261009") is None, "")

print("\nUPDATE + LATEST")
tmp = Path(tempfile.mkdtemp())
one = lambda fd, td: stmt("20260813", "20261009", "1.459564966", end="50729.78")  # noqa: E731
r1 = safe(it.update, tmp, one, "U27760647", today=date(2026, 10, 10),
          closes=lambda f, t: [("20260813", 17460.76), ("20261009", 17525.23)])
check("the row has the account TWR and the S&P 500 TR over the same dates",
      isinstance(r1, dict) and r1["fromDate"] == "20260813" and r1["toDate"] == "20261009"
      and r1["bench_from"] == "20260813" and r1["bench_to"] == "20261009" and abs(r1["bench_ret"] - 0.369221) < 1e-5,
      str(r1))
_lx = tmp / "itd" / "latest.xml"
check("the latest ITD statement is kept", _lx.exists() and _lx.read_bytes() == one("", ""), "")


def boom(f, t):
    raise RuntimeError("no network")


r2 = safe(it.update, tmp, one, "U27760647", today=date(2026, 10, 11), closes=boom)
check("a failed benchmark download leaves its columns blank, the TWR is still written",
      isinstance(r2, dict) and r2["bench_ret"] == "" and r2["twr"] == round(1.459564966, 6), str(r2))
with open(tmp / "itd" / "twr_history.csv", encoding="utf-8") as f:
    hist = list(csv.DictReader(f))
check("history is append-only, one header, one row per run",
      [h["asof"] for h in hist] == ["2026-10-10", "2026-10-11"], str(hist))
lt = safe(it.latest, tmp)
check("latest() is the last row", isinstance(lt, dict) and lt["asof"] == "2026-10-11", str(lt))
check("no history -> None", safe(it.latest, tmp / "nothing") is None, "")
r3 = safe(it.update, tmp, lambda fd, td: stmt(fd, td, "0") if td == "20261009" else b"<bad/>", "U27760647",
          today=date(2026, 10, 10), closes=boom)
check("the asof day itself is not requested (IB's statement ends yesterday)",
      isinstance(r3, dict) and r3["toDate"] == "20261009", str(r3))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} inception-TWR checks behaved as expected")
