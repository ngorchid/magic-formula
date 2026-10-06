"""Tests for scripts/download_ib_records.py — archive, tables, attribution, reconciliation.

No IB, no network: synthetic Flex statements in a temp records dir. Every rule here is pinned by
a case built so that breaking THAT rule fails it — scripts/mutate_ib_records.py seeds those faults
and demands the suite catches every one (a case that cannot fail is decoration).

IB_RECORDS_MODULE may point at another copy of the script (the mutation runner uses this, so the
real file is never modified).

Run: python scripts/test_ib_records.py
"""
from __future__ import annotations

import csv
import hashlib
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
ALERTS: list[str] = []
d.alert = lambda subject, body: ALERTS.append(subject)
d.log = lambda msg: None

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def fresh() -> Path:
    """A new, empty records dir for one scenario."""
    p = Path(tempfile.mkdtemp())
    d.RECORDS = p
    return p


def stmt(frm: str, to: str, inner: str, acct: str = "U27760647") -> bytes:
    return (f'<FlexQueryResponse queryName="t" type="AF"><FlexStatements count="1">'
            f'<FlexStatement accountId="{acct}" fromDate="{frm}" toDate="{to}" '
            f'whenGenerated="{to};080000"><AccountInformation accountId="{acct}" currency="USD"/>'
            f'{inner}</FlexStatement></FlexStatements></FlexQueryResponse>').encode()


class _At:
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


def table(name: str) -> list[dict]:
    p = d.RECORDS / "tables" / f"{name}.csv"
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def trade(tid, sym, conid, ref, price="1", cat="STK", date="20261006", comm="-1"):
    return (f'<Trade tradeID="{tid}" transactionID="{tid}0" ibExecID="e{tid}" symbol="{sym}" '
            f'conid="{conid}" assetCategory="{cat}" tradeDate="{date}" quantity="1" '
            f'tradePrice="{price}" ibCommission="{comm}" fxRateToBase="1" '
            f'levelOfDetail="EXECUTION" orderReference="{ref}"/>')


def cash(tid, typ, amount, desc, ccy="USD", conid="", cat=""):
    return (f'<CashTransaction transactionID="{tid}" type="{typ}" amount="{amount}" '
            f'currency="{ccy}" fxRateToBase="1" description={quoteattr(desc)} conid="{conid}" '
            f'assetCategory="{cat}" levelOfDetail="DETAIL" dateTime="20261006"/>')


def sof(tid, code, amount, desc, conid="", cat=""):
    return (f'<StatementOfFundsLine transactionID="{tid}" activityCode="{code}" amount="{amount}" '
            f'activityDescription={quoteattr(desc)} conid="{conid}" assetCategory="{cat}" currency="USD" '
            f'fxRateToBase="1" levelOfDetail="BaseCurrency" date="20261006"/>')


def by(rows: list[dict], field: str, value: str) -> dict:
    return next((r for r in rows if r.get(field) == value), {})


# =============================================================================================
print("ARCHIVE — raw files are the record")
fresh()
data = stmt("20261001", "20261007", trade(1, "AAPL", "265598", "magic-formula:20261006-160002"))
with _At("2026-10-07T08:30:00"):
    p1 = d.archive(data, "999")
    try:
        d.archive(data, "999")                     # same second -> same file name
        check("archiving to an existing file name REFUSES (never overwrites)", False,
              "second archive() overwrote silently")
    except FileExistsError:
        check("archiving to an existing file name REFUSES (never overwrites)", True)
check("raw file is byte-identical to what IB sent", p1.read_bytes() == data)
m = [json.loads(line) for line in open(d.RECORDS / "manifest.jsonl", encoding="utf-8")]
check("manifest sha256 matches the raw file",
      m and m[0]["sha256"] == hashlib.sha256(p1.read_bytes()).hexdigest(), str(m))
try:
    d.archive(stmt("20261001", "20261007", "", acct="U99999999"), "999")
    check("a statement for another account is REFUSED", False, "accepted")
except d.FlexError as e:
    check("a statement for another account is REFUSED", e.code == "account", str(e))

print("\nTABLES — corrections and tags")
fresh()
with _At("2026-10-07T08:30:00"):
    d.archive(stmt("20261001", "20261007", trade(2, "MCLZ6", "661016519",
                   "trend-overlay:20261006-203001", price="88.48", cat="FUT")), "999")
with _At("2026-10-08T08:30:00"):
    d.archive(stmt("20261002", "20261008", trade(2, "MCLZ6", "661016519",
                   "trend-overlay:20261006-203001", price="88.50", cat="FUT")), "999")
d.rebuild_tables()
t = table("trades")
check("the LATER download wins on the same IB trade id (IB correction applied)",
      len(t) == 1 and t[0]["tradePrice"] == "88.50", str([r.get("tradePrice") for r in t]))
check("orderReference is split into strategy and run_id",
      t and t[0]["strategy"] == "trend-overlay" and t[0]["run_id"] == "20261006-203001", str(t))

# =============================================================================================
print("\nATTRIBUTION — the owner's rules (2026-10-06)")
fresh()
inner = "".join([
    trade(10, "XLE", "777", "options-vrp:20261006-213001"),      # tagged: vrp owns conid 777
    trade(11, "DUP", "888", "magic-formula:20261006-160002"),    # conid 888 traded by two sleeves
    trade(12, "DUP", "888", "trend-overlay:20261006-203001"),
    cash(1, "Dividends", "12.5", "AAPL CASH DIVIDEND", conid="265598", cat="STK"),
    cash(2, "Withholding Tax", "-1.9", "AAPL US TAX", conid="265598", cat="STK"),
    cash(3, "Dividends", "4.0", "XLE CASH DIVIDEND", conid="777", cat="STK"),
    cash(4, "Dividends", "3.0", "DUP CASH DIVIDEND", conid="888", cat="STK"),
    cash(5, "Broker Interest Received", "11.41", "USD CREDIT INT FOR SEP-2026"),
    cash(6, "Broker Interest Paid", "-0.58", "EUR DEBIT INT FOR SEP-2026", ccy="EUR"),
    cash(7, "Broker Interest Received", "0.29", "USD IBKR MANAGED SECURITIES (SYEP) INTEREST"),
    cash(8, "Other Fees", "-1.50", "P01:OPRA FEE FOR OCT 2026"),
    cash(9, "Other Fees", "-10.00", "US SECURITIES SNAPSHOT FOR OCT 2026"),
    cash(20, "Other Fees", "-4.50", "NETWORK C FOR OCT 2026"),
    cash(21, "Other Fees", "-1.00", "ACTIVITY FEE"),
    cash(22, "Deposits/Withdrawals", "1000", "CASH RECEIPTS"),
    cash(23, "Price Adjustments", "0.10", "SOMETHING NEW"),
    sof(30, "ADJ", "-322", "MCL NOV26 Position MTM", conid="661016596", cat="FUT"),
    sof(31, "ADJ", "-0.004", "FX Translations P&L"),
    sof(32, "ADJ", "5", "STOCK ADJUSTMENT", conid="265598", cat="STK"),
    sof(33, "DINT", "-0.58", "EUR Debit Interest for Sep-2026"),   # base row: currency says USD
])
with _At("2026-10-07T08:30:00"):
    d.archive(stmt("20261001", "20261007", inner), "999")
d.rebuild_tables()
ct, sf = table("cash_transactions"), table("statement_of_funds")


def sl(tid):
    return by(ct, "transactionID", str(tid)).get("sleeve")


check("stock dividend -> magic-formula (asset class STK)", sl(1) == "magic-formula", sl(1))
check("withholding tax -> magic-formula", sl(2) == "magic-formula", sl(2))
check("dividend on a conid a TAGGED trade gave options-vrp -> options-vrp", sl(3) == "options-vrp", sl(3))
check("a conid traded by two sleeves -> 'conflict' (never silently picked)", sl(4) == "conflict", sl(4))
check("USD interest -> book", sl(5) == "book", sl(5))
check("foreign-currency interest -> magic-formula", sl(6) == "magic-formula", sl(6))
syep = by(ct, "transactionID", "7")
check("SYEP lending income -> magic-formula, category securities_lending",
      syep.get("sleeve") == "magic-formula" and syep.get("category") == "securities_lending", str(syep))
check("OPRA fee -> options-vrp", sl(8) == "options-vrp", sl(8))
check("snapshot-bundle fee -> options-vrp", sl(9) == "options-vrp", sl(9))
check("Network C fee -> options-vrp", sl(20) == "options-vrp", sl(20))
check("other account fee -> book", sl(21) == "book", sl(21))
check("deposit -> capital", sl(22) == "capital", sl(22))
check("unknown row type -> unassigned", sl(23) == "unassigned", sl(23))
fut = by(sf, "transactionID", "30")
check("futures variation margin -> trend-overlay, category futures_mtm",
      fut.get("sleeve") == "trend-overlay" and fut.get("category") == "futures_mtm", str(fut))
fxt = by(sf, "transactionID", "31")
check("FX translation -> magic-formula, category fx_translation",
      fxt.get("sleeve") == "magic-formula" and fxt.get("category") == "fx_translation", str(fxt))
adj = by(sf, "transactionID", "32")
check("ADJ on a stock is an 'adjustment', not futures variation margin",
      adj.get("category") == "adjustment" and adj.get("sleeve") == "magic-formula", str(adj))
dint = by(sf, "transactionID", "33")
check("base-currency ledger row: interest currency read from the description (EUR -> magic)",
      dint.get("sleeve") == "magic-formula", str(dint))
summ = table("attribution_summary")
check("attribution_summary totals the ledger by sleeve x category",
      any(r["sleeve"] == "trend-overlay" and r["category"] == "futures_mtm"
          and float(r["amount_base"]) == -322 for r in summ), str(summ))
warns = d.audit_checks({"trades": 1, "cash_transactions": 1, "open_positions": 1})
check("audit flags unassigned rows", any("not attributable" in w and "cash_transactions" in w
                                         for w in warns), str(warns))
check("audit counts the conflict row too (unassigned + conflict = 2 cash rows)",
      any("2 cash_transactions row(s) not attributable" in w for w in warns), str(warns))

# =============================================================================================
print("\nRECONCILIATION — IB's NAV bridge vs the rows it summarises")


def bridge(nav: str, rows: str) -> list[str]:
    fresh()
    with _At("2026-10-07T08:30:00"):
        d.archive(stmt("20261001", "20261007", f"<ChangeInNAV {nav}/>" + rows), "999")
    return d.reconcile_latest()


# Every bridge field has rows here, so a reconciliation that IGNORES a row type turns this clean
# statement into a mismatch (the first version had no fee rows, so ignoring fees went unnoticed).
rows = (trade(40, "AAPL", "265598", "magic-formula:x", comm="-2.78")
        + cash(41, "Dividends", "12.50", "AAPL CASH DIVIDEND", conid="265598", cat="STK")
        + cash(42, "Broker Interest Received", "8.20", "USD CREDIT INT")
        + cash(44, "Other Fees", "-1.00", "ACTIVITY FEE")
        + cash(45, "Withholding Tax", "-1.88", "AAPL US TAX", conid="265598", cat="STK")
        + cash(46, "Deposits/Withdrawals", "500", "CASH RECEIPTS")
        + '<TransactionTax transactionID="43" taxAmount="-1.25" fxRateToBase="1" '
          'symbol="AAF" date="20261006" taxDescription="UK STAMP DUTY"/>')
ok_nav = ('dividends="12.50" interest="8.20" commissions="-2.78" transactionTax="-1.25" '
          'otherFees="-1.00" withholdingTax="-1.88" depositsWithdrawals="500"')
check("a statement whose rows add up gives NO warning", bridge(ok_nav, rows) == [],
      str(bridge(ok_nav, rows)))
w = bridge(ok_nav.replace('commissions="-2.78"', 'commissions="-3.78"'), rows)
check("a $1 commissions gap is caught", any("commissions" in x for x in w), str(w))
w = bridge(ok_nav.replace('transactionTax="-1.25"', 'transactionTax="-2.25"'), rows)
check("a $1 transaction-tax gap is caught", any("transactionTax" in x for x in w), str(w))
w = bridge(ok_nav.replace('otherFees="-1.00"', 'otherFees="-5.50"'), rows)
check("a fee missing from the rows is caught", any("otherFees" in x for x in w), str(w))
w = bridge(ok_nav.replace('dividends="12.50"', 'dividends="12.53"'), rows)
check("a 3-cent difference is within tolerance (no false alarm)", w == [], str(w))

# =============================================================================================
print("\nREVIEW ALERTS — once per event, not once per day")
fresh()
ca = ('<CorporateAction transactionID="50" conid="265598" symbol="AAPL" assetCategory="STK" '
      'type="FS" description="AAPL 4 FOR 1 SPLIT" reportDate="20261006"/>')
with _At("2026-10-07T08:30:00"):
    d.archive(stmt("20261001", "20261007", ca), "999")
d.rebuild_tables()
first = [w for w in d.audit_checks({"trades": 1, "cash_transactions": 1, "open_positions": 1})
         if w.startswith("REVIEW")]
with _At("2026-10-08T08:30:00"):
    d.archive(stmt("20261002", "20261008", ca), "999")       # same event, next day's window
d.rebuild_tables()
second = [w for w in d.audit_checks({"trades": 1, "cash_transactions": 1, "open_positions": 1})
          if w.startswith("REVIEW")]
check("a corporate action is flagged for review", len(first) == 1 and "AAPL" in first[0], str(first))
check("...and NOT again the next day", second == [], str(second))

print("\nUNTAGGED TRADES")
fresh()
with _At("2026-10-07T08:30:00"):
    d.archive(stmt("20261001", "20261007", trade(60, "AAPL", "265598", "", date="20261007")), "999")
d.rebuild_tables()
w = d.audit_checks({"trades": 1, "cash_transactions": 1, "open_positions": 1})
check("a trade since 2026-10-06 without a strategy tag is flagged",
      any("no strategy tag" in x for x in w), str(w))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} ib-records checks behaved as expected")
