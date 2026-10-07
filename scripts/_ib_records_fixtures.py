"""Shared fixtures for the IB-records test suites (no IB, no network).

Loads scripts/download_ib_records.py -- or the copy named by IB_RECORDS_MODULE, which is how the
mutation runners test a mutant without ever touching the real file -- and provides synthetic Flex
statements, ledgers and a check() that counts cases.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import quoteattr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
MOD = os.getenv("IB_RECORDS_MODULE", str(ROOT / "scripts" / "download_ib_records.py"))
_spec = importlib.util.spec_from_file_location("ib_records_under_test", MOD)
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

d.EXPECT_ACCOUNT = "U27760647"
ALERTS: list[tuple[str, str]] = []
d.alert = lambda subject, body: ALERTS.append((subject, body))
d.log = lambda msg: None

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def finish(what: str) -> None:
    print("\n" + "=" * 88)
    if _fails:
        print(f"{len(_fails)} FAILURE(S) of {_ran}:")
        for f in _fails:
            print("   " + f)
        sys.exit(1)
    print(f"all {_ran} {what} checks behaved as expected")


def fresh() -> Path:
    """A new, empty records dir (and ledger dir) for one scenario; ledgers default to absent."""
    p = Path(tempfile.mkdtemp())
    d.RECORDS = p
    d.LEDGERS = {k: p / "ledgers" / f"{k}.json" for k in ("magic-formula", "trend-overlay",
                                                          "options-vrp")}
    return p


def ledger(sleeve: str, state: dict) -> None:
    path = d.LEDGERS[sleeve]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def empty_ledgers() -> None:
    ledger("magic-formula", {"positions": [], "trade_log": []})
    ledger("trend-overlay", {"trade_log": []})
    ledger("options-vrp", {"trade_log": []})


def stmt(frm: str, to: str, inner: str, acct: str = "U27760647") -> bytes:
    return (f'<FlexQueryResponse queryName="t" type="AF"><FlexStatements count="1">'
            f'<FlexStatement accountId="{acct}" fromDate="{frm}" toDate="{to}" '
            f'whenGenerated="{to};080000"><AccountInformation accountId="{acct}" currency="USD"/>'
            f'{inner}</FlexStatement></FlexStatements></FlexQueryResponse>').encode()


class At:
    """Pin datetime.now() inside the module (archive file names are second-resolution)."""

    def __init__(self, ts: str):
        self.ts = datetime.fromisoformat(ts)

    def __enter__(self):
        fixed = self.ts

        class _DT(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed
        self._orig, d.datetime = d.datetime, _DT

    def __exit__(self, *a):
        d.datetime = self._orig


def put(frm: str, to: str, inner: str, when: str) -> None:
    with At(when):
        d.archive(stmt(frm, to, inner), "999")


def table(name: str) -> list[dict]:
    p = d.RECORDS / "tables" / f"{name}.csv"
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def trade(eid: str, sym: str, conid: str, ref: str, side: str, qty: float, price: float,
          cat: str = "STK", date: str = "20261007", underlying: str = "", expiry: str = "",
          strike: str = "", right: str = "", notes: str = "", comm: str = "-1",
          ccy: str = "USD") -> str:
    q = qty if side == "BUY" else -qty
    return (f'<Trade tradeID="t{eid}" transactionID="x{eid}" ibExecID="{eid}" symbol="{sym}" '
            f'conid="{conid}" assetCategory="{cat}" tradeDate="{date}" buySell="{side}" '
            f'currency="{ccy}" '
            f'quantity="{q}" tradePrice="{price}" ibCommission="{comm}" fxRateToBase="1" '
            f'levelOfDetail="EXECUTION" orderReference="{ref}" underlyingSymbol="{underlying}" '
            f'expiry="{expiry}" strike="{strike}" putCall="{right}" notes="{notes}"/>')


def cash(tid, typ, amount, desc, ccy="USD", conid="", cat="", date="20261007"):
    return (f'<CashTransaction transactionID="{tid}" type="{typ}" amount="{amount}" '
            f'currency="{ccy}" fxRateToBase="1" description={quoteattr(desc)} conid="{conid}" '
            f'assetCategory="{cat}" levelOfDetail="DETAIL" dateTime="{date}"/>')


def eae(tid, conid, underlying_conid, sym, kind, date="20261007", strike="", qty="2"):
    return (f'<OptionEAE transactionID="{tid}" conid="{conid}" underlyingConid="{underlying_conid}" '
            f'symbol="{sym}" assetCategory="OPT" transactionType="{kind}" date="{date}" '
            f'quantity="{qty}" multiplier="100" strike="{strike}"/>')
