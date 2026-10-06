"""Tests for the order -> IB record links: orderRef tag and execution ids. No IB, no network.

Every order must carry orderRef "magic-formula:<run id>" and every fill's IB execution ids
(= Flex `ibExecID`) must reach the ledger, or the audit trail cannot tie this sleeve's records to
IB's. Driven through the REAL Broker and PortfolioState against a fake IB.

Run: python scripts/test_order_links.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from paper.broker import Broker  # noqa: E402
from paper.state import PortfolioState, Position  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


class FakeIB:
    """Fills every order at once; `fills_after` sleeps before the execDetails arrive."""

    def __init__(self, fills_after: int = 0):
        self.sent, self.n, self.fills_after = [], 0, fills_after

    def qualifyContracts(self, *cs):
        for c in cs:
            c.conId = 1
        return list(cs)

    def placeOrder(self, contract, order):
        self.n += 1
        self.sent.append(order)
        n, self._trade = self.n, None
        fills = [NS(execution=NS(execId=f"0001.{n}.01"))]
        t = NS(order=order, fills=[] if self.fills_after else fills,
               orderStatus=NS(status="Filled", avgFillPrice=10.0, filled=order.totalQuantity))
        self._pending = (t, fills, self.fills_after)
        return t

    def sleep(self, s):
        t, fills, after = getattr(self, "_pending", (None, None, 0))
        if t is not None and not t.fills:
            self._pending = (t, fills, after - 1)
            if after - 1 <= 0:
                t.fills = fills

    def cancelOrder(self, o):
        pass


def broker(ib, ref="magic-formula:20261006-160002") -> Broker:
    b = Broker(dry_run=False)
    b.ib, b.order_ref = ib, ref
    b.qualify = lambda ticker: NS(symbol=ticker)
    return b


print("ORDER TAGS")
ib = FakeIB()
b = broker(ib)
buy = b.order("AAPL", "BUY", 3, wait=2)
check("a stock order carries orderRef", ib.sent[-1].orderRef == "magic-formula:20261006-160002",
      repr(ib.sent[-1].orderRef))
b.convert_fx("EUR", 1000.0, 1.1, wait=2)
check("an FX sweep order carries orderRef", ib.sent[-1].orderRef == "magic-formula:20261006-160002",
      repr(ib.sent[-1].orderRef))
nb = Broker.__new__(Broker)                     # built without __init__: no order_ref attribute
nb.ib, nb.dry_run, nb._contracts = FakeIB(), False, {}
nb.qualify = lambda ticker: NS(symbol=ticker)
r = nb.order("AAPL", "BUY", 1, wait=2)
check("a broker without a tag still places the order (a tag never blocks a trade)",
      r["ok"] is True and r["status"] == "Filled", str(r))

print("\nEXECUTION IDS")
check("a filled order returns its IB execution ids", buy.get("exec_ids") == ["0001.1.01"], str(buy))
check("...and its orderRef", buy.get("order_ref") == "magic-formula:20261006-160002", str(buy))
late = broker(FakeIB(fills_after=3)).order("MSFT", "BUY", 1, wait=2)
check("execution details arriving just after 'Filled' are still captured",
      late.get("exec_ids") == ["0001.1.01"], str(late))
fx = broker(FakeIB()).convert_fx("GBP", 500.0, 1.3, wait=2)
check("an FX sweep returns its execution ids", fx.get("exec_ids") == ["0001.1.01"], str(fx))

print("\nLEDGER")
st = PortfolioState()
st.open_position(Position("AAPL", 3, buy["fill_price"], "2026-10-06",
                          entry_order_ref=buy["order_ref"], entry_exec_ids=buy["exec_ids"]))
sell = b.order("AAPL", "SELL", 3, wait=2)
rec = st.close_position("AAPL", sell["fill_price"], 1.0, "2026-11-05", "test",
                        order_ref=sell["order_ref"], exec_ids=sell["exec_ids"])
check("a closed trade keeps the ENTRY tag and execution ids",
      rec["entry_order_ref"] == "magic-formula:20261006-160002"
      and rec["entry_exec_ids"] == ["0001.1.01"], str(rec))
check("...and the EXIT tag and execution ids",
      rec["exit_order_ref"] == "magic-formula:20261006-160002"
      and rec["exit_exec_ids"] == [f"0001.{ib.n}.01"], str(rec))
old = Position(**{"ticker": "INCY", "shares": 15, "entry_price": 122.0, "entry_date": "2026-09-15"})
check("a position saved before these fields existed still loads",
      old.entry_order_ref == "" and old.entry_exec_ids == [], str(old))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} order-link checks behaved as expected")
