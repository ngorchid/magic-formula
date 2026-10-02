"""Tests for the SPY benchmark line in the daily email.

WHY THIS EXISTS. The since-inception "vs SPY" number jumped day to day even when the strategy
tracked SPY almost exactly. Two causes, both fixed and pinned here:
  1. auto_adjust re-adjusted SPY history on every fetch (dividends) -> drifting anchor. (raw now.)
  2. the run fires intraday (16:00 CET ~ 10:00 ET), so a fresh fetch compared today's PARTIAL bar
     against yesterday's SETTLED close -> the since-inception move != the day's move. The email
     now reads SPY from the stored run-time snapshots (same instants as the NAV), so the
     since-inception rel changes by exactly the daily rel.

Run: python scripts/test_email_benchmark.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.email_report import _pct, build_email_body  # noqa: E402
from paper.state import PortfolioState  # noqa: E402

fails, ran = [], 0


def expect(label: str, cond: bool, detail: str = "") -> None:
    global ran
    ran += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(label)


# Real snapshots from the Oct-1/Oct-2 incident (stored at run time = 10:00 ET intraday).
INC = {"date": "2026-08-17", "nav": 50000.0, "spy": 775.065}
OCT1 = {"date": "2026-10-01", "nav": 49150.57, "spy": 762.09}
OCT2 = {"date": "2026-10-02", "nav": 49747.33, "spy": 770.225}


def _state(hist):
    return PortfolioState(inception_date="2026-08-17", inception_nav=50000.0,
                          cash=hist[-1]["nav"], positions=[], realized_pnl=0.0,
                          nav_history=[dict(h) for h in hist])


def _rel(body: str):
    m = re.search(r"<b>([+\-][\d.]+)%</b> rel", body)
    return float(m.group(1)) if m else None


# Build with DELIBERATELY WRONG fetched params to prove the stored snapshots override them.
body2 = build_email_body(_state([INC, OCT1, OCT2]), {}, {},
                         spy_day_ret=0.50, spy_incep_ret=0.50, today="2026-10-02")

exp_spy_day = 770.225 / 762.09 - 1          # today vs yesterday, both stored snapshots
exp_spy_inc = 770.225 / 775.065 - 1         # today vs inception snapshot
exp_outperf = (49747.33 / 50000 - 1) - exp_spy_inc

expect("stored snapshots override the fetched spy params (no +50.00% leaks through)",
       "50.00%" not in body2)
expect("spy day return taken from stored snapshots", f"vs SPY {_pct(exp_spy_day)}" in body2,
       _pct(exp_spy_day))
expect("spy since-inception taken from stored snapshots", f"vs SPY {_pct(exp_spy_inc)}" in body2,
       _pct(exp_spy_inc))

# THE consistency property the user cares about: rel moves by the daily rel.
rel_oct2 = _rel(body2)
rel_oct1 = _rel(build_email_body(_state([INC, OCT1]), {}, {}, None, None, today="2026-10-01"))
strat_day = 49747.33 / 49150.57 - 1
daily_rel = (strat_day - exp_spy_day) * 100
expect("rel is parseable", rel_oct1 is not None and rel_oct2 is not None)
expect("since-inception rel changes by the daily rel (was off by ~0.23%)",
       abs((rel_oct2 - rel_oct1) - daily_rel) < 0.01,
       f"rel {rel_oct1:+.3f}->{rel_oct2:+.3f} (chg {rel_oct2-rel_oct1:+.3f}) vs daily {daily_rel:+.3f}")

# First run (only one snapshot): no pair exists, must fall back to the passed value, not crash.
body0 = build_email_body(_state([INC]), {}, {}, spy_day_ret=0.001, spy_incep_ret=0.002,
                         today="2026-08-17")
expect("first run falls back to passed spy params without error", "vs SPY" in body0)

print("=" * 70)
print(f"{ran} ran, {'ALL PASS' if not fails else f'{len(fails)} FAIL: ' + ', '.join(fails)}")
if fails:
    raise SystemExit(1)
