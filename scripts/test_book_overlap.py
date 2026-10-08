"""Tests for three book-summary lines (2026-10-07): names held by both sleeves, IBKR's account TWR
as the headline, and sleeve percentages stated against their disclosed base.
scripts/mutate_book_overlap.py seeds the faults. No IB, no network.

Run: python scripts/test_book_overlap.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass
import book_summary as bs  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


tmp = Path(tempfile.mkdtemp())
magic, opt = tmp / "magic.json", tmp / "opt.json"
magic.write_text(json.dumps({"inception_date": "2026-08-17", "inception_nav": 50000.0,
                             "positions": [{"ticker": "XOM", "shares": 12}, {"ticker": "SAP.DE", "shares": 3},
                                           {"ticker": "AAPL", "shares": 5}],
                             "nav_history": [{"date": "2026-10-06", "nav": 50000.0},
                                             {"date": "2026-10-07", "nav": 50200.0}]}))
opt.write_text(json.dumps({"inception_date": "2026-10-04",
                           "open_spreads": [{"ticker": "XOM", "short_strike": 110, "long_strike": 105, "contracts": 3},
                                            {"ticker": "IWM", "short_strike": 263, "long_strike": 256, "contracts": 2},
                                            {"ticker": "AAPL", "short_strike": 200, "long_strike": 190, "contracts": 1,
                                             "assigned_contracts": 1},
                                            {"ticker": "SAP", "short_strike": 150, "long_strike": 140, "contracts": 1}],
                           "nav_history": [["2026-10-06", 0.0], ["2026-10-07", -150.0]]}))

def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        return f"crashed: {type(e).__name__}: {e}"


print("NAMES HELD BY BOTH SLEEVES")
line = safe(bs.overlap_line, magic, opt)
check("XOM and AAPL are listed with both sleeves' holdings", "XOM (magic 12 sh · VRP 110/105P x3)" in line
      and "AAPL (magic 5 sh · VRP 200/190P x1 ASSIGNED)" in line, line)
check("a name only one sleeve holds (IWM) is not listed", "IWM" not in line, line)
check("the yfinance exchange suffix is dropped (magic SAP.DE ~ VRP SAP)", "SAP (magic 3 sh" in line, line)
empty = tmp / "none.json"
empty.write_text(json.dumps({"open_spreads": [{"ticker": "IWM"}]}))
check("no overlap -> 'none'", safe(bs.overlap_line, magic, empty) == "Names held by both sleeves: none",
      safe(bs.overlap_line, magic, empty))
bad = tmp / "bad.json"
bad.write_text("{oops")
check("an unreadable state -> says unknown, never crashes", "unknown" in safe(bs.overlap_line, magic, bad), "")

print("\nIBKR TWR HEADLINE")
rec = tmp / "records"
(rec / "tables").mkdir(parents=True)
with open(rec / "tables" / "change_in_nav.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["fromDate", "toDate", "twr"])
    w.writeheader()
    w.writerows([{"fromDate": "20260929", "toDate": "20261005", "twr": "0.40"},
                 {"fromDate": "20260930", "toDate": "20261006", "twr": "1.25"},
                 {"fromDate": "20261001", "toDate": "20261006", "twr": "9.99"}])
bs.RECORDS_DIR = rec
twr = bs.ibkr_twr()
check("the most recent statement's TWR is read (latest toDate, widest window)",
      twr == (1.25, "20260930", "20261006"), str(twr))
bs.RECORDS_DIR = tmp / "missing"
check("no archive -> None (headline says unavailable)", bs.ibkr_twr() is None, "")

print("\nIN THE EMAIL")
bs.RECORDS_DIR = rec
bs.MAGIC_STATE, bs.OPTIONS_STATE, bs.TREND_STATE = magic, opt, tmp / "no-trend.json"
bs.NETLIQ_HIST = tmp / "netliq.json"
bs._ib_margin = lambda: {"NetLiquidation": 50_100.0, "ExcessLiquidity": 20_000.0,
                         "FullMaintMarginReq": 10_000.0, "GrossPositionValue": 60_000.0}
os.environ.pop("TREND_BASE", None)
_b = safe(bs.build)
body = _b[1] if isinstance(_b, tuple) else str(_b)
check("the account TWR (IBKR) is the headline", "<b>Account TWR:</b> +1.25% (20260930–20261006, IBKR)" in body,
      re.findall(r"Account TWR[^<]*</b>[^<]*", body))
check("the overlap line is in the email", "Names held by both sleeves: AAPL" in body, "")
check("a sleeve percentage names its disclosed base",
      "of $50,000)" in body, re.findall(r"\([-0-9.]+% of \$[0-9,]+\)", body))

print("\nOPEN ASSIGNMENTS — a second channel, every day")
import numpy as np  # noqa: E402
TODAY = "2026-10-07"
d = lambda n: str(np.busday_offset(np.datetime64(TODAY), -n, roll="backward"))  # noqa: E731
asg = tmp / "asg.json"
asg.write_text(json.dumps({"open_spreads": [
    {"ticker": "XLE", "short_strike": 90, "long_strike": 85, "contracts": 3, "assigned_contracts": 3,
     "assigned_date": TODAY, "assigned_auto": True},
    {"ticker": "NKE", "short_strike": 70, "long_strike": 67.5, "contracts": 2, "assigned_contracts": 2,
     "assigned_date": d(1), "assigned_auto": False},
    {"ticker": "BAC", "short_strike": 40, "long_strike": 37.5, "contracts": 6, "assigned_contracts": 6,
     "assigned_date": d(4), "assigned_auto": True, "assigned_stock_sold": True},
    {"ticker": "IWM", "short_strike": 263, "long_strike": 256, "contracts": 2}]}))
lines = safe(bs.open_assignments, asg, TODAY)
ok = isinstance(lines, list) and len(lines) == 3
sus = tmp / "sus.json"
sus.write_text(json.dumps({"open_spreads": [
    {"ticker": "XLE", "short_strike": 59, "long_strike": 57, "contracts": 8, "assign_suspected": 1,
     "assign_suspected_date": d(2)}]}))
part = tmp / "part.json"
part.write_text(json.dumps({"open_spreads": [
    {"ticker": "IWM", "short_strike": 263, "long_strike": 256, "contracts": 2, "assigned_contracts": 2,
     "assigned_date": TODAY, "assigned_auto": True, "assigned_shares_sold": 120.0,
     "unwind_order": {"leg": "stock", "permId": 77}}]}))
pl = safe(bs.open_assignments, part, TODAY)
check("a partly sold unwind shows the shares left and the working order (decision #15)",
      isinstance(pl, list) and len(pl) == 1 and "80 of 200 IWM shares" in pl[0]
      and "unwind order working at IB (permId 77)" in pl[0], str(pl))
sl = safe(bs.open_assignments, sus, TODAY)
check("a SUSPECTED assignment (short gone, no shares) is listed too, escalating (URGENT day 3)",
      isinstance(sl, list) and len(sl) == 1 and sl[0].startswith("SUSPECTED URGENT day 3")
      and "XLE 59/57P x1" in sl[0] and "not managed or valued" in sl[0], str(sl))
check("one line per assigned spread, none for an intact one", ok and not any("IWM" in x for x in lines), str(lines))
check("escalates with age: OPEN day 1, ESCALATION day 2, URGENT day 5",
      ok and lines[0].startswith("OPEN day 1") and lines[1].startswith("ESCALATION day 2")
      and lines[2].startswith("URGENT day 5"), str(lines))
check("says what is still on and whether it is automatic or MANUAL",
      ok and "300 XLE shares + 3 long 85P open; automatic unwind retrying" in lines[0]
      and "MANUAL unwind needed" in lines[1] and "shares sold, long puts still open" in lines[2], str(lines))
check("an unreadable ledger says UNKNOWN (never 'nothing open')",
      "UNKNOWN" in str(safe(bs.open_assignments, bad, TODAY)), "")
check("no ledger file -> nothing", safe(bs.open_assignments, tmp / "none-here.json", TODAY) == [], "")
bs.OPTIONS_STATE = asg
_b2 = safe(bs.build)
check("the email shows the block AND marks the subject '⚠ ASSIGNED'",
      isinstance(_b2, tuple) and _b2[0].startswith("⚠ ASSIGNED ") and "OPEN ASSIGNMENTS (options-vrp)" in _b2[1],
      str(_b2)[:200])
bs.OPTIONS_STATE = empty                 # an IWM spread, not assigned
_b3 = safe(bs.build)
check("no open assignment -> no marker, no red block, and an EXPLICIT 'none'",
      isinstance(_b3, tuple) and not _b3[0].startswith("⚠") and "OPEN ASSIGNMENTS" not in _b3[1]
      and "Open assignments (options-vrp): none" in _b3[1], str(_b3)[:120])
bs.OPTIONS_STATE = tmp / "not-there.json"
_b4 = safe(bs.build)
check("options ledger missing -> 'UNKNOWN — ledger not found', never 'none'",
      isinstance(_b4, tuple) and "Open assignments (options-vrp): UNKNOWN" in _b4[1]
      and "Open assignments (options-vrp): none" not in _b4[1], str(_b4)[:120])

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} book-summary line checks behaved as expected")
