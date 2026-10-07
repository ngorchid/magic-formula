"""Tests for the documented capital bases and the NAV reconciliation line in the book summary.

Drives the REAL build() with synthetic sleeve states and a fake IB account read. No IB, no network.
scripts/mutate_capital_bases.py seeds the faults.

Run: python scripts/test_capital_bases.py
"""
from __future__ import annotations

import importlib
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

for k in ("TREND_BASE", "OPTIONS_BASE", "BOOK_INCEPTION_CAPITAL"):
    os.environ.pop(k, None)
import book_summary as bs  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


CFG = json.loads((ROOT / "config" / "capital_bases.json").read_text())
DOC = (ROOT / "docs" / "capital_bases.md").read_text()

print("DOCUMENTED BASES")
check("config holds the three bases (50k / 75k / 50k)",
      [CFG[s]["amount"] for s in ("magic-formula", "trend-overlay", "options-vrp")] == [50000, 75000, 50000],
      str(CFG))
check("each base has an effective date and says what it represents",
      all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", CFG[s]["effective"]) and CFG[s]["represents"]
          for s in ("magic-formula", "trend-overlay", "options-vrp")), "")
check("the doc's table matches the config (amount + effective date per sleeve)",
      all(f"{CFG[s]['effective']}" in DOC and f"{CFG[s]['amount']:,}" in DOC
          for s in ("magic-formula", "trend-overlay", "options-vrp")), "")
check("the doc has a rule for changes and a change log",
      "## Rule for changes" in DOC and "## Change log" in DOC, "")
check("book_summary reads its bases from the config (no hard-coded defaults)",
      bs.TREND_BASE == CFG["trend-overlay"]["amount"] and bs.OPTIONS_BASE == CFG["options-vrp"]["amount"],
      f"{bs.TREND_BASE} {bs.OPTIONS_BASE}")


def scenario(net_liq, magic_nav=50_600.0, trend=300.0, opt=-100.0, env=None):
    tmp = Path(tempfile.mkdtemp())
    for k in ("TREND_BASE", "OPTIONS_BASE"):
        os.environ.pop(k, None)
    os.environ.update(env or {})
    m = importlib.reload(bs)
    for name, st in (("magic", {"inception_date": "2026-08-17", "inception_nav": 50000.0,
                                "nav_history": [{"date": "2026-10-06", "nav": 50_400.0},
                                                {"date": "2026-10-07", "nav": magic_nav}]}),
                     ("trend", {"inception_date": "2026-09-14",
                                "nav_history": [{"date": "2026-10-06", "total_pnl": 0.0},
                                                {"date": "2026-10-07", "total_pnl": trend}]}),
                     ("opt", {"inception_date": "2026-10-04",
                              "nav_history": [["2026-10-06", 0.0], ["2026-10-07", opt]]})):
        (tmp / f"{name}.json").write_text(json.dumps(st))
    m.MAGIC_STATE, m.TREND_STATE, m.OPTIONS_STATE = (tmp / "magic.json", tmp / "trend.json",
                                                     tmp / "opt.json")
    m.NETLIQ_HIST = tmp / "netliq.json"
    m._ib_margin = (lambda: None) if net_liq is None else (lambda: {
        "NetLiquidation": net_liq, "ExcessLiquidity": 20_000.0, "FullMaintMarginReq": 10_000.0,
        "GrossPositionValue": 60_000.0})
    for k in (env or {}):
        os.environ.pop(k, None)
    return m.build()[1]


print("\nTHE NAV LINE — sleeve ledgers vs the account")
body = scenario(51_000.0)                  # 50,000 + magic 600 + trend 300 + options -100 = 50,800
check("the line spells out NAV = inception + each sleeve's ledger P&L + unattributed",
      "NAV $51,000 = inception $50,000 + magic $+600 + trend $+300 + options $-100 + unattributed $+200"
      in body, re.findall(r"NAV \$[^<]*", body))
check("a small unattributed remainder (0.4% of NAV) is NOT flagged", "do not explain" not in body,
      re.findall(r"NAV \$[^<]*", body))
body = scenario(52_000.0)                  # unattributed 1,200 = 2.3% of NAV
check("an unattributed remainder over 1% of NAV IS flagged",
      "unattributed $+1,200 ⚠ over 1% of NAV" in body, re.findall(r"NAV \$[^<]*", body))
body = scenario(50_100.0)                  # unattributed -700 = 1.4% -> a NEGATIVE gap flags too
check("a negative gap over 1% is flagged as well", "unattributed $-700 ⚠" in body,
      re.findall(r"NAV \$[^<]*", body))
check("the bases line labels them overlapping, not allocations, with their sum",
      "Return bases (overlapping — shared collateral, not allocations; sum $175,000)" in body
      and "magic $50,000 · trend $75,000 · options $50,000" in body,
      re.findall(r"Return bases[^<]*", body))
body = scenario(None)
check("without an account read: bases still shown, no NAV claim",
      "Return bases" in body and "NAV $" not in body, "")

print("\nOVERRIDES ARE VISIBLE")
body = scenario(51_000.0, env={"TREND_BASE": "100000"})
check("an env override of a base is printed as a warning naming the documented value",
      "return base OVERRIDDEN" in body and "trend 100,000 (documented 75,000)" in body,
      re.findall(r"OVERRIDDEN[^<]*", body))
body = scenario(51_000.0)
check("no override -> no warning", "OVERRIDDEN" not in body, "")

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} capital-base checks behaved as expected")
