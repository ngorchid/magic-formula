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

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} book-summary line checks behaved as expected")
