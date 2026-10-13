"""Tests for the owner's buy exclusions (PaperConfig.no_buy, scripts/run_paper.py NO_BUY).

2026-10-13: CTVA is kept out of NEW buys until 2027-03-01 -- Corteva separated on 2026-10-01
and the frozen October ranking scores the pre-separation company. No IB, no network.

Run: python scripts/test_no_buy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from paper.orchestrator import PaperConfig, run_daily  # noqa: E402
from paper.state import PortfolioState, Position  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


class FillBroker:
    """Fills every order; dry_run keeps run_daily off the margin / FX-sweep IB calls."""
    dry_run = True

    def __init__(self):
        self.orders = []

    def order(self, ticker, action, shares, wait=20.0):
        self.orders.append((ticker, action))
        return {"ok": True, "status": "Filled", "fill_price": 100.0, "exec_ids": [f"E.{ticker}"],
                "order_ref": f"magic-formula:RUN-{action}", "conid": 1, "commission": 1.0,
                "currency": "USD"}


TODAY = "2026-10-13"
NAMES = ["CTVA"] + [f"N{i}" for i in range(1, 12)]            # CTVA ranks FIRST
RANK = pd.Series(range(len(NAMES)), index=NAMES, dtype=float)


def panels_to(today):
    """A price panel ending on `today` (so no name reads stale), 10 slots = 10% per order."""
    idx = pd.bdate_range(end=pd.Timestamp(today), periods=400)
    rng = np.random.default_rng(11)
    px = pd.DataFrame({n: 100 * np.exp(np.cumsum(rng.standard_normal(len(idx)) * 0.01))
                       for n in NAMES}, index=idx)
    return {"adj": px, "currency": {n: "USD" for n in NAMES}}


def day(no_buy, positions=(), today=TODAY):
    st = PortfolioState(cash=100_000.0, positions=list(positions))
    b = FillBroker()
    try:
        run_daily(st, RANK, panels_to(today), {"USD": 1.0}, b,
                  PaperConfig(max_new_buys_per_day=2, hold_n=12, top_n=10, budget=100_000.0,
                              no_buy=no_buy), today)
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        return f"crashed: {type(e).__name__}: {e}", st
    return [t for t, a in b.orders if a == "BUY"], st


EXCL = {"CTVA": ("2026-11-01", "test")}
print("EXCLUDED FROM NEW BUYS")
buys, _ = day(EXCL)
check("the top-ranked but excluded name is skipped; the next two are bought",
      buys == ["N1", "N2"], str(buys))
buys, _ = day({})
check("control: without the exclusion it IS bought first", isinstance(buys, list) and buys[:1] == ["CTVA"],
      str(buys))
buys, _ = day(EXCL, today="2026-11-02")
check("expired (today >= until) -> bought again", isinstance(buys, list) and buys[:1] == ["CTVA"], str(buys))
buys, _ = day({"CTVA": ("2026-11-02", "test")}, today="2026-11-02")
check("'until' is exclusive: ON that day it may be bought", isinstance(buys, list) and buys[:1] == ["CTVA"],
      str(buys))
buys, _ = day(EXCL, today="2026-10-30")
check("days before 'until' it is still excluded", isinstance(buys, list) and "CTVA" not in buys, str(buys))

print("\nHOLDS AND SELLS UNTOUCHED")
held = Position("CTVA", 10, 100.0, "2026-10-12", entry_order_ref="magic-formula:R", entry_exec_ids=["X"])
buys, st = day(EXCL, positions=[held])
check("an excluded name that is already held is NOT sold because of the exclusion",
      any(p.ticker == "CTVA" for p in st.positions), str([p.ticker for p in st.positions]))

print("\nTHE LIVE CONFIG")
import run_paper  # noqa: E402
check("NO_BUY holds CTVA until 2027-03-01", run_paper.NO_BUY.get("CTVA", ("",))[0] == "2027-03-01",
      str(run_paper.NO_BUY))
try:
    _cfg = run_paper.paper_config()
    _nb = _cfg.no_buy
except Exception as e:  # noqa: BLE001
    _nb = f"crashed: {type(e).__name__}: {e}"
check("paper_config() passes NO_BUY to the orchestrator", _nb == run_paper.NO_BUY, str(_nb))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} no-buy checks behaved as expected")
