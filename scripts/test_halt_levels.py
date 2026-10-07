"""Tests for the kill-switch levels and the documented sizing config, magic-formula (2026-10-07).

magic-formula has no SAFETY-flagged closes, so HALT_ALL and HALT_HARD both stop it before the
universe refresh and before any broker is made. Its budget comes from config/capital_bases.json and
is the CAP on its NAV-linked deployment base -- not a fixed sizing number.
scripts/mutate_halt_levels.py seeds the faults. No IB, no network.

Run: python scripts/test_halt_levels.py
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import risk_guard as rg  # noqa: E402
import run_paper as runner  # noqa: E402

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


print("LEVELS IN risk_guard (shared)")
t = Path(tempfile.mkdtemp())
os.environ.pop("TRADING_HALT", None)
(t / "HALT_ALL").write_text("x")
check("HALT_ALL file -> all", rg.halt_state(t)[0] == rg.HALT_ALL, "")
(t / "HALT_HARD").write_text("x")
check("HALT_HARD file beats HALT_ALL", rg.halt_state(t)[0] == rg.HALT_HARD, "")
os.environ["TRADING_HALT"] = "3"
check("TRADING_HALT=3 -> hard", rg.halt_state(Path(tempfile.mkdtemp()))[0] == rg.HALT_HARD, "")
os.environ.pop("TRADING_HALT")

print("\nmain() STOPS ENTIRELY UNDER HALT_ALL AND HALT_HARD (no SAFETY closes here)")


def boom(*a, **k):
    raise AssertionError("must not get this far under a hard/all halt")


runner._refresh_ranking = boom
runner.make_broker = boom
runner.push_if_alerts = lambda *a, **k: None
for lvl in (rg.HALT_ALL, rg.HALT_HARD):
    runner.halt_state = lambda root, lvl=lvl: (lvl, "test")
    r = safe(runner.main, dry_run=True, force=True)
    check(f"{lvl}: returns before the universe refresh and before any broker", r is None, str(r))
runner.halt_state = lambda root: (rg.HALT_NEW, "test")
r = safe(runner.main, dry_run=True, force=True)
check("HALT_NEW still proceeds (manages existing positions) — reaches the refresh",
      isinstance(r, str) and "must not get this far" in r, str(r))

print("\nDOCUMENTED SIZING — and how this NAV-linked sleeve reads it")
os.environ.pop("BUDGET", None)
WARN: list[str] = []
_ow = rg.logging.warning
rg.logging.warning = lambda f, *a: WARN.append(f % a if a else f)
cfg = runner.paper_config()
check("budget comes from config/capital_bases.json (50,000; the old code defaulted to 100,000)",
      cfg.budget == 50_000 and WARN == [], str((cfg.budget, WARN)))
os.environ["BUDGET"] = "70000"
cfg2 = runner.paper_config()
check("an env BUDGET that differs is used but WARNED about", cfg2.budget == 70_000
      and any("DIFFERS" in w and "magic-formula" in w for w in WARN), str(WARN))
os.environ.pop("BUDGET")
rg.logging.warning = _ow
from paper import orchestrator  # noqa: E402
check("the budget is the CAP on the deployment base: sizing stays gap-to-NAV vs min(NAV, budget)",
      "nav = min(state.nav(marks, fx), cfg.budget)" in inspect.getsource(orchestrator), "")
check("main() builds its config through paper_config (no second BUDGET read)",
      "cfg = paper_config()" in inspect.getsource(runner.main)
      and 'os.getenv("BUDGET"' not in inspect.getsource(runner.main), "")

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} halt-level checks behaved as expected")
