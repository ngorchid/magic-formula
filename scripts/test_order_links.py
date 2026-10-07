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
    """Fills every order at once; `fills_after` sleeps before the execDetails arrive;
    reports=False: IB never sends the commission reports."""

    def __init__(self, fills_after: int = 0, reports: bool = True):
        self.sent, self.n, self.fills_after, self.reports = [], 0, fills_after, reports

    def qualifyContracts(self, *cs):
        for c in cs:
            c.conId = 1
        return list(cs)

    def placeOrder(self, contract, order):
        self.n += 1
        self.sent.append(order)
        n, self._trade = self.n, None
        fills = [NS(execution=NS(execId=f"0001.{n}.01"),
                    commissionReport=NS(execId=f"0001.{n}.01" if self.reports else "", commission=1.25))]
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
    b.qualify = lambda ticker: NS(symbol=ticker, conId=265598, currency="USD")
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
                          entry_order_ref=buy["order_ref"], entry_exec_ids=buy["exec_ids"],
                          entry_conid=buy["conid"], entry_commission=buy["commission"]))
sell = b.order("AAPL", "SELL", 3, wait=2)
rec = st.close_position("AAPL", sell["fill_price"], 1.0, "2026-11-05", "test",
                        order_ref=sell["order_ref"], exec_ids=sell["exec_ids"],
                        commission=sell["commission"])
check("a closed trade keeps the ENTRY tag and execution ids",
      rec["entry_order_ref"] == "magic-formula:20261006-160002"
      and rec["entry_exec_ids"] == ["0001.1.01"], str(rec))
check("...and the EXIT tag and execution ids",
      rec["exit_order_ref"] == "magic-formula:20261006-160002"
      and rec["exit_exec_ids"] == [f"0001.{ib.n}.01"], str(rec))
old = Position(**{"ticker": "INCY", "shares": 15, "entry_price": 122.0, "entry_date": "2026-09-15"})
check("a position saved before these fields existed still loads",
      old.entry_order_ref == "" and old.entry_exec_ids == [], str(old))


print("\nCANCEL RACE")


class RaceIB(FakeIB):
    """Order sits 'Submitted' through the poll and fills just as the cancel lands."""

    def placeOrder(self, contract, order):
        self.n += 1
        self.sent.append(order)
        self._race = NS(order=order, fills=[],
                        orderStatus=NS(status="Submitted", avgFillPrice=0.0, filled=0))
        return self._race

    def sleep(self, s):
        pass

    def cancelOrder(self, o):
        self._race.orderStatus = NS(status="Filled", avgFillPrice=10.0, filled=o.totalQuantity)
        self._race.fills = [NS(execution=NS(execId=f"0009.{self.n}.01"))]


race = broker(RaceIB()).order("NVDA", "BUY", 2, wait=2)
check("a fill that lands during the cancel race keeps its execution ids",
      race.get("status") == "Filled" and race.get("exec_ids") == ["0009.1.01"], str(race))
check("...and its orderRef", race.get("order_ref") == "magic-formula:20261006-160002", str(race))

# ---------------------------------------------------------------------------------------------
# RUNNER WIRING. Everything above hands the broker its tag by hand, so it cannot see the runner
# forgetting to: deleting that one line in run_paper.py sent every order out untagged while all
# of the above stayed green. These pin the runner's own wiring.
print("\nCONID / COMMISSION / CURRENCY (additive, 2026-10-07)")
check("a filled order returns its conid, commission and currency",
      buy.get("conid") == 265598 and buy.get("commission") == 1.25 and buy.get("currency") == "USD",
      str(buy))
nr = broker(FakeIB(reports=False)).order("AAPL", "BUY", 1, wait=2)
check("commission reports that never arrive -> commission None, and the order still succeeds",
      nr.get("ok") is True and nr.get("status") == "Filled" and nr.get("commission") is None
      and nr.get("exec_ids"), str(nr))
check("a closed trade records conid, entry and exit commission",
      rec.get("conid") == 265598 and rec.get("entry_commission") == 1.25
      and rec.get("exit_commission") == 1.25, str(rec))
check("a position saved before these fields existed loads with conid 0 / commission None",
      old.entry_conid == 0 and old.entry_commission is None, str(old))

print("\nRUNNER WIRING")
import inspect  # noqa: E402
import re  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_paper  # noqa: E402

check("ORDER_REF is 'magic-formula:<YYYYmmdd-HHMMSS>'",
      re.fullmatch(r"magic-formula:\d{8}-\d{6}", run_paper.ORDER_REF) is not None,
      run_paper.ORDER_REF)
mb = run_paper.make_broker(dry_run=True)
check("make_broker tags the broker with this run's ORDER_REF",
      mb.order_ref == run_paper.ORDER_REF, repr(mb.order_ref))
_src = inspect.getsource(run_paper.main)
check("main() builds its broker through make_broker, never Broker() directly",
      "make_broker(" in _src and "Broker(" not in _src, "")

# ---------------------------------------------------------------------------------------------
# ORCHESTRATOR -> LEDGER. The ledger checks above call open_position / close_position directly
# with the values the broker returned, so they cannot see run_daily dropping them on the way.
# This drives the REAL run_daily: one clock-expired name is sold, new names are bought.
# top_n must be large enough that one order clears the 15% single-order cap (top_n=3 makes every
# order 33% of budget, the guard rejects them all and the BUY cases could not fail).
print("\nORCHESTRATOR -> LEDGER")
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paper.orchestrator import PaperConfig, run_daily  # noqa: E402


class LinkBroker:
    """Collaborator stub: fills every order and returns ids the way the real Broker does.
    dry_run=True only keeps run_daily off the margin/FX-sweep IB calls; the fills are 'Filled'."""

    dry_run = True

    def __init__(self):
        self.n = 0

    def order(self, ticker, action, shares, wait=20.0):
        self.n += 1
        return {"ok": True, "status": "Filled", "fill_price": 100.0,
                "exec_ids": [f"E{self.n}.{ticker}"], "order_ref": f"magic-formula:RUN-{action}",
                "conid": 1000 + self.n, "commission": 1.0 + self.n / 100, "currency": "USD"}


_today = "2026-08-17"
_idx = pd.bdate_range(end=pd.Timestamp(_today), periods=400)
_rng = np.random.default_rng(7)
_names = [f"N{i}" for i in range(14)] + ["OLD"]       # OLD sits outside the hold band
_px = pd.DataFrame({n: 100 * np.exp(np.cumsum(_rng.standard_normal(len(_idx)) * 0.01))
                    for n in _names}, index=_idx)
_rank = pd.Series(range(len(_names)), index=_names, dtype=float)       # OLD ranks last
_st = PortfolioState(cash=100_000.0, positions=[
    Position("OLD", 10, 100.0, "2026-07-01", entry_order_ref="magic-formula:OLD-RUN",
             entry_exec_ids=["X1"])])
run_daily(_st, _rank, {"adj": _px, "currency": {c: "USD" for c in _names}}, {"USD": 1.0},
          LinkBroker(), PaperConfig(max_new_buys_per_day=2, hold_n=12, top_n=10, budget=100_000.0),
          _today)
_new = [p for p in _st.positions if p.ticker != "OLD"]
_sold = [r for r in _st.trade_log if r.get("ticker") == "OLD"]
check("run_daily bought at least one name (the fixture can fail)", len(_new) > 0,
      str([p.ticker for p in _st.positions]))
check("a BUY placed by run_daily stores its orderRef on the position",
      bool(_new) and all(p.entry_order_ref == "magic-formula:RUN-BUY" for p in _new),
      str([(p.ticker, p.entry_order_ref) for p in _new]))
check("...and its execution ids",
      bool(_new) and all(len(p.entry_exec_ids) == 1 and p.entry_exec_ids[0].endswith(f".{p.ticker}")
                         for p in _new),
      str([(p.ticker, p.entry_exec_ids) for p in _new]))
check("a SELL placed by run_daily stores the EXIT tag and execution ids",
      len(_sold) == 1 and _sold[0]["exit_order_ref"] == "magic-formula:RUN-SELL"
      and len(_sold[0]["exit_exec_ids"]) == 1 and _sold[0]["exit_exec_ids"][0].endswith(".OLD"),
      str(_sold))
check("run_daily stores each BUY's conid and commission on the position",
      bool(_new) and all(p.entry_conid > 1000 and p.entry_commission and p.entry_commission > 1.0
                         for p in _new), str([(p.ticker, p.entry_conid, p.entry_commission) for p in _new]))
check("run_daily stores the SELL's commission on the closed trade",
      len(_sold) == 1 and (_sold[0].get("exit_commission") or 0) > 1.0, str(_sold))
check("...and keeps the ENTRY tag and execution ids of the position it closed",
      len(_sold) == 1 and _sold[0]["entry_order_ref"] == "magic-formula:OLD-RUN"
      and _sold[0]["entry_exec_ids"] == ["X1"], str(_sold))

with tempfile.TemporaryDirectory() as _tmp:
    _f = Path(_tmp) / "state.json"
    _st.save(_f)
    _back = PortfolioState.load(_f)
check("save -> load keeps each position's orderRef and execution ids",
      [(p.entry_order_ref, p.entry_exec_ids) for p in _back.positions]
      == [(p.entry_order_ref, p.entry_exec_ids) for p in _st.positions] and bool(_new), "")

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} order-link checks behaved as expected")
