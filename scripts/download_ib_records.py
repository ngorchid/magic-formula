"""Daily archive of IB's own records (Flex Web Service) for the LIVE account — the audit trail.

WHY. The sleeves' state.json / run.log say what each strategy BELIEVES it did. IB's statements
say what actually happened — every execution with its commission and fees, cash movements
(dividends, withholding tax, interest, FX), corporate actions, transfers, positions and NAV —
and they are the books of record (tax, disputes, reconciliation). Nothing archived them.

HOW. One Activity Flex Query (configured once in Client Portal — see the setup notes printed by
`--help` and in docs/ib_records.md) is downloaded daily via the Flex Web Service:
  1. SendRequest(token, queryId) -> ReferenceCode + statement URL
  2. GetStatement(ReferenceCode) polled until generated (1019 = still generating)
The query covers a ROLLING window (e.g. last 7 calendar days), not one day: every download
overlaps the previous ones, so a missed run, a weekend or a late IB correction is still
captured. Duplicates are resolved downstream by IB's own ids.

STORAGE (outside any checkout, default C:\\Users\\Nicolas\\IB-records; IB_RECORDS_DIR overrides):
  raw/YYYY/flex_<query>_<from>_<to>_<downloaded>.xml   immutable, exactly as IB sent it
  manifest.jsonl                                      append-only: file, sha256, size, period
  tables/*.csv                                        DERIVED, rebuilt from ALL raw files each
                                                      run (trades, cash, positions, ...); safe
                                                      to delete — raw is the record.
Raw files are never modified or overwritten; the manifest's sha256 lets anyone verify that.

ATTRIBUTION. Every order is tagged orderRef "<strategy>:<run id>" (since 2026-10-06), which IB
reports as `orderReference` on executions — ONLY if the query has "Include Audit Trail Fields:
Yes". tables/trades.csv adds `strategy` and `run_id` columns parsed from it; untagged rows (older
trades, manual trades, FX sweeps placed before tagging) are left blank and counted in the log.

Env (live .env): FLEX_TOKEN, FLEX_QUERY_ID, optional IB_RECORDS_DIR, EXPECT_ACCOUNT (U27760647),
EMAIL_USER/EMAIL_PASS/TO_EMAIL for the alerts and the daily one-line success email; for the
two-way execution check MAGIC_STATE / TREND_STATE / OPTIONS_STATE (the sleeve ledgers, read-only);
optional HEARTBEAT_URL (dead-man's switch, pinged after each completed run).

Run:  python scripts/download_ib_records.py            # download + rebuild tables
      python scripts/download_ib_records.py --rebuild  # rebuild tables from raw only (offline)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import smtplib
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RECORDS = Path(os.getenv("IB_RECORDS_DIR", r"C:\Users\Nicolas\IB-records"))
EXPECT_ACCOUNT = os.getenv("EXPECT_ACCOUNT", "U27760647")
SEND_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
# IB rejects some default library user agents; identify plainly.
UA = {"User-Agent": "ib-records-archiver/1.0 (python-urllib)"}

# Codes worth waiting on (statement still being produced, or IB busy) vs. hard failures.
IN_PROGRESS = {"1019"}
RETRY_LATER = {"1001", "1004", "1005", "1006", "1007", "1008", "1009", "1018", "1021"}

# Derived tables: XML element -> (csv name, unique-key attributes for de-duplication).
# The key is IB's own id where one exists; otherwise the natural key of a daily row. PERIOD
# summaries (cash report, performance, NAV bridge) describe the statement's whole window, which
# differs on every rolling download -- their key includes "_period" so each window is kept
# rather than one overwriting another. Element names verified on the first real statement
# (2026-10-06).
TABLES = {
    "AccountInformation":   ("account_information", ("accountId", "_period")),
    "Trade":                ("trades",            ("tradeID", "transactionID", "ibExecID")),
    "Order":                ("orders",            ("ibOrderID", "orderReference", "dateTime")),
    "Lot":                  ("closed_lots",       ("transactionID", "conid", "dateTime", "openDateTime",
                                                   "quantity")),
    "CashReportCurrency":   ("cash_report",       ("currency", "levelOfDetail", "_period")),
    "FIFOPerformanceSummaryUnderlying": ("performance_summary", ("conid", "symbol", "description",
                                                                  "_period")),
    "UnbundledCommissionDetail": ("commission_details", ("tradeID", "brokerExecutionCharge",
                                                          "dateTime", "exchange")),
    # Transaction TAXES (UK stamp duty, Italian/French FTT, ...) — not in the commission columns.
    "TransactionTax":       ("transaction_taxes", ("tradeId", "transactionID", "date", "taxDescription",
                                                    "taxAmount")),
    "CashTransaction":      ("cash_transactions", ("transactionID",)),
    "StatementOfFundsLine": ("statement_of_funds", ("transactionID", "date", "activityCode",
                                                    "currency", "amount", "balance")),
    "OpenPosition":         ("open_positions",    ("reportDate", "conid", "currency")),
    # Daily price + MTM per held instrument (NB: IB gives no quantity here; open_positions has it).
    "PriorPeriodPosition":  ("daily_positions",   ("date", "conid")),
    "MTMPerformanceSummaryUnderlying": ("mtm_performance", ("conid", "symbol", "description",
                                                             "_period")),
    # Foreign-cash balances and their realised FX P&L. FxTransaction has no IB id, so the
    # natural key is used; raw keeps every row regardless.
    "FxPosition":           ("fx_positions",      ("reportDate", "fxCurrency", "levelOfDetail",
                                                   "lotOpenDateTime", "lotDescription")),
    "FxTransaction":        ("fx_transactions",   ("reportDate", "fxCurrency", "dateTime",
                                                   "activityDescription", "quantity", "proceeds",
                                                   "levelOfDetail")),
    "TierInterestDetail":   ("interest_tiers",    ("reportDate", "valueDate", "currency",
                                                   "interestType", "tierBreak")),
    "EquitySummaryByReportDateInBase": ("nav_daily", ("reportDate",)),
    "ChangeInNAV":          ("change_in_nav",     ("fromDate", "toDate", "_period")),
    "CorporateAction":      ("corporate_actions", ("transactionID",)),
    "Transfer":             ("transfers",         ("transactionID",)),
    "OptionEAE":            ("option_exercises",  ("transactionID", "date", "conid")),
    "InterestAccrualsCurrency": ("interest_accruals", ("fromDate", "toDate", "currency")),
    "ConversionRate":       ("conversion_rates",  ("reportDate", "fromCurrency", "toCurrency")),
    "SecurityInfo":         ("securities",        ("conid",)),
}


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def _get(url: str, params: dict) -> bytes:
    with urlopen(Request(f"{url}?{urlencode(params)}", headers=UA), timeout=60) as r:
        return r.read()


class FlexError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def _error(root: ET.Element) -> tuple[str, str] | None:
    """(code, message) if this response is an error/status document, else None."""
    if root.tag == "FlexStatementResponse":
        if (root.findtext("Status") or "") == "Success":
            return None
        return root.findtext("ErrorCode") or "?", root.findtext("ErrorMessage") or ""
    if root.tag == "FlexQueryResponse":
        return None
    return root.findtext("ErrorCode") or root.tag, root.findtext("ErrorMessage") or ET.tostring(
        root, encoding="unicode")[:200]


def download(token: str, query_id: str, max_wait: int = 300) -> bytes:
    """Return the statement XML bytes exactly as IB sent them."""
    root = ET.fromstring(_get(SEND_URL, {"t": token, "q": query_id, "v": 3}))
    err = _error(root)
    if err:
        raise FlexError(*err)
    ref, url = root.findtext("ReferenceCode"), root.findtext("Url")
    log(f"statement requested (reference {ref}); polling")
    waited, delay = 0, 5
    while waited < max_wait:
        time.sleep(delay)                     # also keeps us under 1 req/s, 10 req/min
        waited += delay
        data = _get(url, {"t": token, "q": ref, "v": 3})
        root = ET.fromstring(data)
        err = _error(root)
        if not err:
            return data
        if err[0] not in IN_PROGRESS and "in progress" not in err[1].lower():
            raise FlexError(*err)
        delay = min(delay + 5, 20)
    raise FlexError("timeout", f"statement not ready after {max_wait}s")


def archive(data: bytes, query_id: str) -> Path:
    """Write raw XML immutably and append to the manifest. Returns the file path."""
    root = ET.fromstring(data)
    stmts = root.findall(".//FlexStatement")
    if not stmts:
        raise FlexError("empty", "no FlexStatement in response")
    accounts = {s.get("accountId") for s in stmts}
    if EXPECT_ACCOUNT and accounts != {EXPECT_ACCOUNT}:
        raise FlexError("account", f"statement is for {accounts}, expected {EXPECT_ACCOUNT}")
    frm = min(s.get("fromDate", "") for s in stmts)
    to = max(s.get("toDate", "") for s in stmts)
    now = datetime.now()
    out = RECORDS / "raw" / now.strftime("%Y") / f"flex_{query_id}_{frm}_{to}_{now:%Y%m%dT%H%M%S}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as f:                # "x": never overwrite an existing record
        f.write(data)
    entry = {"file": str(out.relative_to(RECORDS)).replace("\\", "/"),
             "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
             "query_id": query_id, "account": sorted(accounts), "from": frm, "to": to,
             "generated": stmts[0].get("whenGenerated"), "downloaded": now.isoformat(timespec="seconds")}
    with open(RECORDS / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log(f"archived {entry['file']} ({len(data):,} bytes, {frm}..{to}, sha256 {entry['sha256'][:12]}…)")
    return out


def _split_ref(ref: str) -> tuple[str, str]:
    """'trend-overlay:20261006-203001' -> ('trend-overlay', '20261006-203001')."""
    if ref and ":" in ref:
        s, r = ref.split(":", 1)
        return s, r
    return "", ""


# ---------------------------------------------------------------------------------------------
# ATTRIBUTION — which sleeve does each cash event belong to? (owner's rules, 2026-10-06)
#
# Every row of IB's cash ledger lands in exactly one bucket, so the buckets add up to the
# account's change in NAV:
#   instrument events (dividends, withholding, corporate actions, option exercise/expiry,
#   futures variation margin, transaction taxes) -> the sleeve that holds the instrument: by the
#       orderReference of the tagged trade that traded it, else by asset class (each sleeve
#       trades its own class: STK/CASH -> magic-formula, FUT -> trend-overlay, OPT -> vrp)
#   SYEP securities-lending income    -> magic-formula (the only sleeve holding stock)
#   interest in a FOREIGN currency    -> magic-formula (its stocks + FX sweeps create those
#                                        balances)
#   interest in the BASE currency     -> book    (the shared USD pool belongs to no sleeve)
#   market-data subscription fees     -> options-vrp (OPRA exists for it)
#   other account fees                -> book
#   deposits / withdrawals            -> capital (flows, not P&L)
#   anything else                     -> unassigned -> audit alert
# ---------------------------------------------------------------------------------------------
ASSET_SLEEVE = {"STK": "magic-formula", "CASH": "magic-formula", "FUT": "trend-overlay",
                "OPT": "options-vrp", "BAG": "options-vrp"}
MARKET_DATA_WORDS = ("OPRA", "SNAPSHOT", "MARKET DATA", "TOP OF BOOK", "BUNDLE", "NETWORK A",
                     "NETWORK B", "NETWORK C", "NP,L1", "NON-PROFESSIONAL")
SYEP_WORDS = ("SYEP", "MANAGED SECURITIES", "SECURITIES LENT", "LENDING")
CASH_TYPE_CATEGORY = {
    "Dividends": "dividend", "Payment In Lieu Of Dividends": "dividend",
    "Withholding Tax": "withholding_tax", "Broker Interest Paid": "interest",
    "Broker Interest Received": "interest", "Bond Interest Paid": "interest",
    "Bond Interest Received": "interest", "Other Fees": "fee",
    "Deposits/Withdrawals": "capital", "Commission Adjustments": "commission",
}
SOF_CODE_CATEGORY = {
    "BUY": "trade", "SELL": "trade", "ADJ": "futures_mtm", "DINT": "interest", "CINT": "interest",
    "INT": "interest", "DIV": "dividend", "PIL": "dividend", "FRTAX": "withholding_tax",
    "WHT": "withholding_tax", "OFEE": "fee", "FEE": "fee", "DEP": "capital", "WITH": "capital",
    "CA": "corporate_action", "TTAX": "transaction_tax", "SLINC": "securities_lending",
    # STAX = sales tax / VAT that IB charges ON a fee (e.g. on the market-data subscription,
    # first seen 2026-10-07). It follows the fee it is charged on -- see attribute().
    "STAX": "sales_tax",
}


def _interest_ccy(row: dict) -> str:
    """Currency an interest line is about. The ledger's base-currency rows say 'USD' for every
    line, so read the description ('EUR Debit Interest for Sep-2026') first."""
    desc = (row.get("description") or row.get("activityDescription") or "").strip()
    head = desc[:3].upper()
    if len(desc) > 4 and head.isalpha() and desc[3] == " " and "INT" in desc.upper():
        return head
    return row.get("currency", "")


def _conid_owners(trades: dict[tuple, dict]) -> dict[str, str]:
    """conid -> sleeve, from TAGGED trades. A conid traded by two sleeves maps to 'conflict'."""
    owners: dict[str, str] = {}
    for t in trades.values():
        s, c = t.get("strategy"), t.get("conid")
        if s and c:
            owners[c] = s if owners.get(c, s) == s else "conflict"
    return owners


# DELIVERED SHARES (owner's rule, 2026-10-07). An assigned short put delivers stock: an UNTAGGED
# STK row on the option's underlying. By asset class it would land on magic-formula, which never
# traded it. So: when options-vrp had an assignment/exercise on an underlying and a stock row on
# that underlying appears within DELIVERY_WINDOW_DAYS trading days, the stock -- and every later
# row on it (dividends, the eventual sale) -- belongs to options-vrp. If magic-formula has a
# TAGGED trade in the same conid the row is "conflict" (two owners), never silently picked.
DELIVERY_WINDOW_DAYS = 3     # IB books the delivery on the assignment date; 3 covers a weekend +
                             # a late statement row without catching an unrelated later trade


def _row_date(row: dict) -> str:
    """YYYYMMDD of a Flex row, whatever date field that row type carries."""
    for f in ("tradeDate", "date", "reportDate", "dateTime"):
        v = (row.get(f) or "").replace("-", "")[:8]
        if len(v) == 8 and v.isdigit():
            return v
    return ""


def _add_trading_days(yyyymmdd: str, n: int) -> str:
    d = datetime.strptime(yyyymmdd, "%Y%m%d")
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d.strftime("%Y%m%d")


def _vrp_deliveries(rows: dict[str, dict[tuple, dict]], owners: dict[str, str]) -> dict[str, str]:
    """{underlying conid -> date of the options-vrp assignment/exercise that delivered stock}."""
    out: dict[str, str] = {}
    stock = [t for t in rows.get("trades", {}).values()
             if t.get("assetCategory") == "STK" and not t.get("strategy")]
    for r in rows.get("option_exercises", {}).values():
        kind = r.get("transactionType", "")
        if not ("Assign" in kind or "Exercise" in kind):
            continue
        if _instrument_sleeve(r, owners) != "options-vrp":
            continue
        uc, ev = r.get("underlyingConid", ""), _row_date(r)
        if not (uc and ev):
            continue
        last = _add_trading_days(ev, DELIVERY_WINDOW_DAYS)
        window = [t for t in stock if t.get("conid") == uc and ev <= _row_date(t) <= last]
        if not window:
            continue
        # FINGERPRINT (owner's rule, 2026-10-07): delivered shares come in exactly
        # multiplier x contracts, at exactly the strike -- near-unique. The window alone is the
        # fallback; each delivery row records which test identified it.
        shares = abs(float(r.get("quantity") or 0)) * float(r.get("multiplier") or 100)
        strike = r.get("strike")
        for t in window:
            fp = (strike not in (None, "") and abs(abs(float(t.get("quantity") or 0)) - shares) < 1e-9
                  and abs(float(t.get("tradePrice") or 0) - float(strike)) < 0.01)
            t["delivery_match"] = "fingerprint" if fp else "window"
        out[uc] = min(out.get(uc, ev), ev)
    return out


def _delivered_owner(tagged_owner: str | None) -> str:
    return "options-vrp" if tagged_owner in (None, "", "options-vrp") else "conflict"


def _instrument_sleeve(row: dict, owners: dict[str, str],
                       delivered: dict[str, str] | None = None) -> str:
    ev = (delivered or {}).get(row.get("conid", ""))
    if ev and row.get("assetCategory", "STK") in ("STK", "") and _row_date(row) >= ev:
        return _delivered_owner(owners.get(row.get("conid", "")))
    return (owners.get(row.get("conid", ""))
            or ASSET_SLEEVE.get(row.get("assetCategory", ""), "unassigned"))


def attribute(row: dict, category: str, owners: dict[str, str], base: str,
              delivered: dict[str, str] | None = None) -> str:
    """The sleeve (or 'book' / 'capital' / 'unassigned') a cash event belongs to."""
    desc = (row.get("description") or row.get("activityDescription") or "").upper()
    if category == "capital":
        return "capital"
    if category == "securities_lending" or any(w in desc for w in SYEP_WORDS):
        return "magic-formula"
    if category == "interest":
        return "book" if _interest_ccy(row) == base else "magic-formula"
    if category == "fx_translation":
        return "magic-formula"            # revaluation of the foreign balances its trades create
    if category in ("fee", "sales_tax") and not row.get("conid"):
        # VAT on a fee goes where the fee goes: market data -> options-vrp, any other -> book.
        return "options-vrp" if any(w in desc for w in MARKET_DATA_WORDS) else "book"
    if row.get("conid") or row.get("assetCategory"):
        return _instrument_sleeve(row, owners, delivered)
    return "unassigned"


def _category(row: dict, base_category: str) -> str:
    """Refine a code/type-based category using the description (IB reuses codes: 'ADJ' is both
    a futures variation-margin line and the FX revaluation of foreign cash)."""
    desc = (row.get("description") or row.get("activityDescription") or "").upper()
    if any(w in desc for w in SYEP_WORDS):
        return "securities_lending"
    if "FX TRANSLATION" in desc:
        return "fx_translation"
    if base_category == "futures_mtm" and row.get("assetCategory") != "FUT":
        return "adjustment"
    return base_category


def _apply_attribution(rows: dict[str, dict[tuple, dict]]) -> None:
    """Add `sleeve` and `category` columns to every cash-type table (in place)."""
    owners = _conid_owners(rows.get("trades", {}))
    delivered = _vrp_deliveries(rows, owners)
    acct = next(iter(rows.get("account_information", {}).values()), {})
    base = acct.get("currency") or "USD"
    for t in rows.get("trades", {}).values():
        t["sleeve"] = t.get("strategy") or _instrument_sleeve(t, owners, delivered)
        t["category"] = "trade"
    for r in rows.get("cash_transactions", {}).values():
        r["category"] = _category(r, CASH_TYPE_CATEGORY.get(r.get("type", ""), "other"))
        r["sleeve"] = attribute(r, r["category"], owners, base, delivered)
    for r in rows.get("statement_of_funds", {}).values():
        code = r.get("activityCode", "")
        r["category"] = _category(r, SOF_CODE_CATEGORY.get(code, "other" if code else "balance"))
        r["sleeve"] = ("-" if r["category"] == "balance"
                       else attribute(r, r["category"], owners, base, delivered))
    for name, cat in (("corporate_actions", "corporate_action"), ("option_exercises", "option_event"),
                      ("transaction_taxes", "transaction_tax"), ("fx_transactions", "fx")):
        for r in rows.get(name, {}).values():
            r["category"] = cat
            r["sleeve"] = (_instrument_sleeve(r, owners, delivered) if cat != "fx"
                           else ("book" if r.get("fxCurrency") == base else "magic-formula"))


def _attribution_summary(rows: dict[str, dict[tuple, dict]]) -> list[dict]:
    """Base-currency cash ledger totals by date x sleeve x category. Uses ONE level of detail
    (BaseCurrency if present, else Currency) so nothing is counted twice."""
    sof = [r for r in rows.get("statement_of_funds", {}).values() if r.get("category") != "balance"]
    levels = {r.get("levelOfDetail") for r in sof}
    level = "BaseCurrency" if "BaseCurrency" in levels else (next(iter(levels)) if levels else "")
    agg: dict[tuple, float] = {}
    for r in sof:
        if r.get("levelOfDetail") != level:
            continue
        amt = float(r.get("amount") or 0) * (1.0 if level == "BaseCurrency"
                                              else float(r.get("fxRateToBase") or 1))
        k = (r.get("date") or r.get("reportDate", ""), r["sleeve"], r["category"])
        agg[k] = agg.get(k, 0.0) + amt
    return [{"date": d, "sleeve": s, "category": c, "amount_base": round(v, 4)}
            for (d, s, c), v in sorted(agg.items())]


def reconcile_latest() -> list[str]:
    """Check the LATEST statement against itself: IB's NAV bridge (ChangeInNAV) vs the sum of the
    individual rows it summarises. A mismatch means rows are missing from what we attribute
    (a section not ticked, a new IB row type) -- the trail is not complete."""
    files = sorted((RECORDS / "raw").rglob("*.xml"))
    if not files:
        return []
    warn = []
    for stmt in ET.parse(files[-1]).getroot().iter("FlexStatement"):
        nav = stmt.find(".//ChangeInNAV")
        if nav is None or not nav.attrib:
            return ["latest statement has no ChangeInNAV — cannot reconcile; is the section ticked?"]
        fx = lambda e, f: float(e.get(f) or 0) * float(e.get("fxRateToBase") or 1)  # noqa: E731
        got = {"dividends": 0.0, "withholdingTax": 0.0, "interest": 0.0, "otherFees": 0.0,
               "depositsWithdrawals": 0.0, "commissions": 0.0, "transactionTax": 0.0}
        bridge_key = {"dividend": "dividends", "withholding_tax": "withholdingTax",
                      "interest": "interest", "fee": "otherFees", "capital": "depositsWithdrawals",
                      "commission": "commissions"}
        for e in stmt.iter("CashTransaction"):
            if not e.attrib or e.get("levelOfDetail", "DETAIL") not in ("DETAIL", ""):
                continue
            k = bridge_key.get(CASH_TYPE_CATEGORY.get(e.get("type", ""), ""))
            if k:
                got[k] += fx(e, "amount")
        for e in stmt.iter("Trade"):
            if e.attrib and e.get("levelOfDetail", "EXECUTION") == "EXECUTION":
                got["commissions"] += fx(e, "ibCommission")
        for e in stmt.iter("TransactionTax"):
            if e.attrib:
                got["transactionTax"] += fx(e, "taxAmount")
        for k, v in got.items():
            want = float(nav.get(k) or 0)
            if abs(want - v) > max(0.05, 0.001 * abs(want)):
                warn.append(f"NAV bridge mismatch {stmt.get('fromDate')}-{stmt.get('toDate')}: "
                            f"{k} IB={want:,.2f} vs rows={v:,.2f} — rows missing from the trail?")
    return warn


def rebuild_tables() -> dict[str, int]:
    """Rebuild tables/*.csv from every raw file. Later downloads win on duplicate keys, so a
    corrected record from IB replaces the earlier version of the same id."""
    files = sorted((RECORDS / "raw").rglob("*.xml"))
    rows: dict[str, dict[tuple, dict]] = {name: {} for name, _ in TABLES.values()}
    unmapped: dict[str, int] = {}
    for p in files:
        for stmt in ET.parse(p).getroot().iter("FlexStatement"):
            period = f"{stmt.get('fromDate', '')}-{stmt.get('toDate', '')}"
            # Record types IB sent that no table maps: still safe in raw, but say so, so a newly
            # ticked section never silently stays out of the tables.
            for sec in stmt:
                for el in [sec, *sec]:
                    if el.attrib and el.tag not in TABLES:
                        unmapped[el.tag] = unmapped.get(el.tag, 0) + 1
            for tag, (name, key) in TABLES.items():
                for el in stmt.iter(tag):
                    if not el.attrib:     # a section CONTAINER sharing the row's tag (e.g. an
                        continue          # empty <OptionEAE/>) -- not a record
                    a = dict(el.attrib)
                    if tag == "Trade":
                        a["strategy"], a["run_id"] = _split_ref(a.get("orderReference", ""))
                    a["_period"], a["_source_file"] = period, p.name
                    rows[name][tuple(a.get(k, "") for k in key)] = a
    _apply_attribution(rows)
    summary = _attribution_summary(rows)
    if summary:
        rows["attribution_summary"] = {(r["date"], r["sleeve"], r["category"]): r for r in summary}
    out_dir = RECORDS / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, recs in rows.items():
        if not recs:
            continue
        cols: list[str] = []
        for r in recs.values():
            cols += [c for c in r if c not in cols]
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(recs.values())
        counts[name] = len(recs)
    if unmapped:
        log("in raw but not tabled (add to TABLES if needed): "
            + ", ".join(f"{k} {v}" for k, v in sorted(unmapped.items())))
    return counts


def audit_checks(counts: dict[str, int]) -> list[str]:
    """Warnings that mean the trail is NOT complete, even though the download worked."""
    warn = []
    trades = RECORDS / "tables" / "trades.csv"
    if trades.exists():
        with open(trades, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and "orderReference" not in rows[0]:
            warn.append("trades have NO orderReference column — set 'Include Audit Trail Fields: "
                        "Yes' on the Flex Query, or strategy attribution is impossible")
        tagged_from = "20261006"     # orders before this were never tagged
        untagged = [r for r in rows if not r.get("strategy")
                    and (r.get("tradeDate") or r.get("dateTime", "")[:8]) >= tagged_from]
        if untagged:
            warn.append(f"{len(untagged)} trade(s) since {tagged_from} have no strategy tag "
                        f"(manual trade? e.g. {untagged[0].get('symbol')} "
                        f"{untagged[0].get('tradeDate')})")
    for must in ("trades", "cash_transactions", "open_positions"):
        if must not in counts and (RECORDS / "raw").exists():
            warn.append(f"no '{must}' rows in any statement — is that section in the Flex Query?")
    # Attribution: every cash event must land in a bucket.
    for name in ("cash_transactions", "statement_of_funds", "corporate_actions",
                 "option_exercises", "transaction_taxes", "trades"):
        p = RECORDS / "tables" / f"{name}.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        bad = [r for r in rows if r.get("sleeve") in ("unassigned", "conflict")]
        if bad:
            r = bad[0]
            warn.append(f"{len(bad)} {name} row(s) not attributable to a sleeve (e.g. "
                        f"{r.get('type') or r.get('activityCode')} {r.get('symbol')} "
                        f"{r.get('description') or r.get('activityDescription', '')})")
        # Events that may need a manual correction in a sleeve's own ledger. Reported ONCE each:
        # an event stays in the rolling 7-day window for a week and must not alert 7 times.
        if name in ("corporate_actions", "option_exercises"):
            seen_file = RECORDS / "review_alerted.json"
            seen = set(json.loads(seen_file.read_text())) if seen_file.exists() else set()
            for r in rows:
                key = f"{name}|{r.get('transactionID')}|{r.get('conid')}|{r.get('date') or r.get('reportDate')}"
                needs = (name == "corporate_actions" or "Assign" in r.get("transactionType", "")
                         or "Exercise" in r.get("transactionType", ""))
                if needs and key not in seen:
                    warn.append(f"REVIEW {name}: {r.get('sleeve')} {r.get('symbol')} "
                                f"{r.get('transactionType') or r.get('type', '')} "
                                f"{r.get('description', '')} — check that sleeve's ledger")
                    seen.add(key)
            seen_file.write_text(json.dumps(sorted(seen), indent=1))
    warn += check_descriptions()
    warn += reconcile_latest()
    return warn


# ---------------------------------------------------------------------------------------------
# UNSEEN-DESCRIPTION REGISTRY (2026-10-07). A fee with no instrument whose description matches no
# market-data keyword goes to "book"; a reworded OPRA line or a brand-new IB charge would land
# there silently. So every fee/interest description is normalised (dates, amounts, symbols and
# ISINs stripped) and kept in description_registry.json; the first time an UNSEEN one appears it
# is reported once, with the sleeve it was given. A missing registry is SEEDED silently from the
# statements already archived, so deployment does not alert on every historical line.
REGISTRY_CATEGORIES = {"fee", "interest", "other", "adjustment", "sales_tax"}
_MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"


def normalise_description(desc: str, symbol: str = "") -> str:
    s = (desc or "").upper()
    if symbol:
        s = s.replace(symbol.upper(), " ")
    s = re.sub(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b", " ", s)                       # ISIN
    s = re.sub(rf"\b(?:{_MONTHS})[A-Z]*\.?(?:[-/ ]?\d{{2,4}})?\b", " ", s)  # SEP-2026, OCT 26
    s = re.sub(r"[^A-Z ]", " ", s)                                        # amounts, ids, punct.
    return " ".join(s.split())


def check_descriptions() -> list[str]:
    """Report each fee/interest description not seen before (once), then remember it."""
    seen_rows = []
    for name, field in (("cash_transactions", "description"),
                        ("statement_of_funds", "activityDescription")):
        p = RECORDS / "tables" / f"{name}.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("category") in REGISTRY_CATEGORIES and r.get(field):
                    n = normalise_description(r[field], r.get("symbol", ""))
                    if n:
                        seen_rows.append((n, r[field], r.get("sleeve", ""), _row_date(r)))
    reg_file = RECORDS / "description_registry.json"
    if not reg_file.exists():
        reg = {n: {"first_seen": d, "example": raw, "sleeve": sl} for n, raw, sl, d in seen_rows}
        reg_file.write_text(json.dumps(reg, indent=1, sort_keys=True))
        log(f"description registry seeded with {len(reg)} description(s)")
        # Silent ONLY the first time. A registry that goes missing LATER (deleted, a restore without
        # it) would otherwise reseed quietly and hide every description first seen since -- so if
        # the seeded-marker exists, that case alerts.
        marker = RECORDS / ".registry_seeded"
        if not marker.exists():
            marker.write_text(datetime.now().isoformat(timespec="seconds"))
        else:
            return [f"description registry was MISSING and has been re-seeded from the archive "
                    f"({len(reg)} descriptions) — any fee/interest description first seen since it "
                    f"was lost can no longer be told apart; review description_registry.json"]
        return []
    reg = json.loads(reg_file.read_text())
    warn = []
    for n, raw, sl, d in seen_rows:
        if n in reg:
            continue
        warn.append(f"NEW fee/interest description, assigned to {sl or '?'}: '{raw}' — "
                    f"check the attribution rule covers it")
        reg[n] = {"first_seen": d, "example": raw, "sleeve": sl}
    reg_file.write_text(json.dumps(reg, indent=1, sort_keys=True))
    return warn


# ---------------------------------------------------------------------------------------------
# TWO-WAY EXECUTION CHECK (2026-10-07): IB's Flex executions <-> the sleeves' own ledgers, keyed
# on the IB execution id (Flex `ibExecID`). Flex is the authority. Read-only on the ledgers.
#
#   matched              every id of a ledger fill is in Flex and the fields agree
#   field_mismatch       ids found, but contract / side / quantity / price disagree
#   ledger_not_in_flex   a ledger id Flex does not have, on a date Flex COVERS
#   flex_tagged_unbooked a tagged Flex execution no ledger references  -> a sleeve failed to book
#   flex_untagged        an untagged one -> manual trade or tagging failure (assignment/exercise
#                        deliveries are labelled as such)
#   linked_by_tag_only   a ledger fill with a tag but no ids (options-vrp self-heal books late
#                        fills that way) matched on tag + contract + side + quantity; for
#                        options-vrp the real ids are written to tables/exec_backfill.csv, which
#                        options-vrp applies to its OWN ledger (this job never writes a ledger)
#   fx_sweep_unledgered  magic-formula FX sweeps: tagged, deliberately not in its ledger
#
# NOT compared, because no ledger records them: commission and conid. The contract is compared by
# its description instead (symbol; future root + expiry; option underlying/expiry/strike/right).
# Combos are compared at LEG level. VERIFIED 2026-10-07 on the first three live spreads (IWM, XLE,
# NVDA, opened 2026-10-06): Flex reports ONLY the legs (Trade/OPT/EXECUTION, one per leg, each
# with the tag and its commission) and never the API's combo-level execution id. That id shares
# the legs' prefix and is sequence .01, the legs following as .02, .03: e.g. combo
# 0002be7d.6ac5508e.01.01, legs .02.01 / .03.01. So ONE missing id on a spread whose legs all
# matched is accepted as the combo-level id ONLY if it is exactly that sibling (_is_combo_level_id);
# any other missing id is reported. A combo that partially fills into several executions has not
# been seen yet -- it would surface as ledger_not_in_flex, to be calibrated then.
LINKED_FROM = "20261006"          # first day orders carried a tag and ledgers kept exec ids
AGEING_DAYS = 3                   # a ledger id still not covered by any statement after this
                                  # many business days alerts (IB lag is normally one day)
assignments: list[dict] = []      # options-vrp ASSIGNED events, read from its ledger
LEDGERS = {
    "magic-formula": Path(os.getenv("MAGIC_STATE", str(ROOT / "results" / "paper" / "state.json"))),
    "trend-overlay": Path(os.getenv("TREND_STATE",
        r"C:\Users\Nicolas\PycharmProjects\trend-overlay-live\results\paper\state.json")),
    "options-vrp": Path(os.getenv("OPTIONS_STATE",
        r"C:\Users\Nicolas\PycharmProjects\options-vrp-live\results\paper\state.json")),
}
FAIL_CLASSES = ("field_mismatch", "ledger_not_in_flex", "flex_tagged_unbooked", "flex_untagged",
                "not_covered_aged")


def _iso8(d: str) -> str:
    return (d or "").replace("-", "")[:8]


def _sym(s: str) -> str:
    """IB stock symbol, separators dropped ('VOLV.B', 'BRK B' -> 'VOLVB', 'BRKB')."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _stk_root(ticker: str) -> str:
    t = (ticker or "").upper()
    if "." in t and 1 <= len(t.rsplit(".", 1)[1]) <= 3 and t.rsplit(".", 1)[1].isalpha():
        t = t.rsplit(".", 1)[0]                       # yfinance exchange suffix (.DE, .ST, ...)
    return re.sub(r"[^A-Z0-9]", "", t)


def _conids(row: dict, *keys: str) -> list[str]:
    """Contract ids a ledger row recorded (additive field since 2026-10-07; [] on older rows)."""
    for k in (keys or ("conids", "conid")):
        v = row.get(k)
        if v in (None, "", 0, []):
            continue
        return [str(x) for x in (v if isinstance(v, list) else [v]) if x not in (None, "", 0)]
    return []


def _ledger_fills() -> tuple[list[dict], list[str]]:
    """Every fill the three ledgers record, normalised. Returns (fills, unavailable sleeves).
    Also fills the module-level `assignments` list (options-vrp ASSIGNED events)."""
    fills, missing = [], []
    assignments.clear()
    for sleeve, path in LEDGERS.items():
        if not Path(path).exists():
            missing.append(sleeve)
            continue
        try:
            st = json.loads(Path(path).read_text())
        except Exception:  # noqa: BLE001
            missing.append(sleeve)
            continue
        if sleeve == "magic-formula":
            legs = []
            for p in st.get("positions", []):
                legs.append(("BUY", p.get("ticker"), p.get("shares"), p.get("entry_price"),
                             p.get("entry_date"), p.get("entry_order_ref", ""),
                             p.get("entry_exec_ids") or [], _conids(p, "entry_conid"),
                             p.get("entry_commission"), p.get("currency")))
            for t in st.get("trade_log", []):
                legs.append(("BUY", t.get("ticker"), t.get("shares"), t.get("entry_price"),
                             t.get("entry_date"), t.get("entry_order_ref", ""),
                             t.get("entry_exec_ids") or [], _conids(t, "conid"),
                             t.get("entry_commission"), t.get("currency")))
                legs.append(("SELL", t.get("ticker"), t.get("shares"), t.get("exit_price"),
                             t.get("exit_date"), t.get("exit_order_ref", ""),
                             t.get("exit_exec_ids") or [], _conids(t, "conid"),
                             t.get("exit_commission"), t.get("currency")))
            for side, tk, qty, px, dt, ref, ids, cids, comm, ccy in legs:
                fills.append({"sleeve": sleeve, "date": _iso8(dt), "order_ref": ref,
                              "exec_ids": list(ids), "kind": "STK", "price": px,
                              "legs": [{"key": ("STK", _stk_root(tk)), "side": side,
                                        "qty": abs(float(qty or 0))}],
                              "conids": cids, "commission": comm, "currency": ccy,
                              "ref": f"{tk} {side} {dt}"})
        elif sleeve == "trend-overlay":
            for t in st.get("trade_log", []):
                if "signed_qty" not in t:             # malformed row
                    continue
                # RESYNC rows (ledger snapped to IB) carry neither a tag nor ids, so they are
                # never matched and never reported -- they are bookkeeping, not fills.
                q = float(t.get("signed_qty") or 0)
                fills.append({"sleeve": sleeve, "date": _iso8(t.get("date")),
                              "order_ref": t.get("order_ref", ""),
                              "exec_ids": list(t.get("exec_ids") or []), "kind": "FUT",
                              "price": t.get("price"), "conids": _conids(t),
                              "commission": t.get("commission"), "currency": t.get("currency"),
                              "legs": [{"key": ("FUT", (t.get("symbol") or "").upper(),
                                                _iso8(t.get("expiry"))),
                                        "side": "BUY" if q > 0 else "SELL", "qty": abs(q)}],
                              "ref": f"{t.get('market')} {t.get('symbol')} {t.get('expiry')} "
                                     f"{q:+g} {t.get('date')}"})
        else:
            for t in st.get("trade_log", []):
                act = t.get("action")
                if act in ("ASSIGNED", "ASSIGNED_STOCK_SOLD", "ASSIGNED_LONG_SOLD") and t.get("key"):
                    try:
                        tk, exp, ks, kl = t["key"].rsplit("_", 3)
                    except ValueError:
                        continue
                    if act == "ASSIGNED":
                        assignments.append({"ticker": tk.upper(), "expiry": _iso8(exp),
                                            "strike": float(ks), "contracts": float(t.get("contracts") or 0),
                                            "shares": float(t.get("shares") or 0),
                                            "date": _iso8(t.get("date"))})
                        continue
                    if act == "ASSIGNED_STOCK_SOLD":
                        legs = [{"key": ("STK", tk.upper()), "side": "SELL",
                                 "qty": abs(float(t.get("shares") or 0))}]
                    else:
                        legs = [{"key": ("OPT", tk.upper(), _iso8(exp), float(kl), "P"),
                                 "side": "SELL", "qty": abs(float(t.get("contracts") or 0))}]
                    fills.append({"sleeve": sleeve, "date": _iso8(t.get("date")),
                                  "order_ref": t.get("order_ref", ""),
                                  "exec_ids": list(t.get("exec_ids") or []), "kind": "LEG",
                                  "price": t.get("price"), "legs": legs,
                                  "conids": _conids(t), "commission": t.get("commission"),
                                  "currency": t.get("currency"),
                                  "ref": f"{t['key']} {act} {t.get('date')}"})
                    continue
                if act not in ("OPEN", "CLOSE") or not t.get("key"):
                    continue
                try:
                    tk, exp, ks, kl = t["key"].rsplit("_", 3)
                except ValueError:
                    continue
                n = abs(float(t.get("contracts") or 0))
                short_side, long_side = ("SELL", "BUY") if act == "OPEN" else ("BUY", "SELL")
                fills.append({"sleeve": sleeve, "date": _iso8(t.get("date")),
                              "order_ref": t.get("order_ref", ""),
                              "exec_ids": list(t.get("exec_ids") or []), "kind": "OPT",
                              "price": t.get("credit") if act == "OPEN" else t.get("close_value"),
                              "conids": _conids(t), "commission": t.get("commission"),
                              "currency": t.get("currency"),
                              "action": act, "key": t["key"],
                              "legs": [{"key": ("OPT", tk.upper(), _iso8(exp), float(ks), "P"),
                                        "side": short_side, "qty": n},
                                       {"key": ("OPT", tk.upper(), _iso8(exp), float(kl), "P"),
                                        "side": long_side, "qty": n}],
                              "ref": f"{t['key']} {act} {t.get('date')}"})
    return fills, missing


def _is_combo_level_id(missing: str, leg_ids: list[str]) -> bool:
    """True if `missing` is the combo-level execution id of these legs: same prefix, sequence 01,
    while every leg is a later sequence (verified 2026-10-07: combo 0002be7d.6ac5508e.01.01, legs
    .02.01 / .03.01). Anything else -- another prefix, a leg sequence, an odd shape -- is a real
    missing id."""
    m = missing.split(".")
    if len(m) < 3 or m[-2] != "01" or not leg_ids:
        return False
    for leg in leg_ids:
        p = leg.split(".")
        if len(p) != len(m) or p[:-2] != m[:-2] or p[-2] == "01":
            return False
    return True


def _flex_key(e: dict) -> tuple:
    cat = e.get("assetCategory", "")
    if cat == "OPT":
        return ("OPT", (e.get("underlyingSymbol") or "").upper(), _iso8(e.get("expiry")),
                float(e.get("strike") or 0), (e.get("putCall") or "").upper()[:1])
    if cat == "FUT":
        return ("FUT", (e.get("underlyingSymbol") or e.get("symbol") or "").upper(),
                _iso8(e.get("expiry")))
    return (cat, _sym(e.get("symbol", "")))


def _side(e: dict) -> str:
    bs = (e.get("buySell") or "").upper()
    if bs in ("BUY", "SELL"):
        return bs
    return "BUY" if float(e.get("quantity") or 0) > 0 else "SELL"


def _coverage() -> list[tuple[str, str]]:
    out = []
    for p in sorted((RECORDS / "raw").rglob("*.xml")):
        for stmt in ET.parse(p).getroot().iter("FlexStatement"):
            if stmt.get("fromDate") and stmt.get("toDate"):
                out.append((stmt.get("fromDate"), stmt.get("toDate")))
    return out


def _compare(fill: dict, execs: list[dict]) -> list[str]:
    """Field differences between one ledger fill and the Flex executions of its ids."""
    diffs = []
    legs = fill["legs"]
    keyf = _flex_key
    cids = fill.get("conids") or []
    if cids and len(cids) == len(legs):
        # Recorded contract ids replace the description match: no spelling workarounds, and a
        # leg is identified by the id IB itself assigned (ledger order = leg order).
        legs = [dict(leg, key=("CONID", c)) for leg, c in zip(legs, cids)]
        keyf = lambda e: ("CONID", str(e.get("conid")))  # noqa: E731
    extra = []                    # commission / currency: reported alongside, never instead of
    if fill.get("commission") not in (None, ""):
        flex_comm = sum(abs(float(e.get("ibCommission") or 0)) for e in execs)
        if abs(flex_comm - abs(float(fill["commission"]))) > 0.02:
            extra.append(f"commission: ledger {abs(float(fill['commission'])):.2f} vs Flex {flex_comm:.2f}")
    if fill.get("currency") and any(e.get("currency") and e.get("currency") != fill["currency"]
                                    for e in execs):
        extra.append(f"currency: ledger {fill['currency']} vs Flex "
                     f"{sorted({e.get('currency') for e in execs})}")
    want = {(leg["key"], leg["side"]): leg["qty"] for leg in legs}
    got: dict[tuple, float] = {}
    for e in execs:
        k = (keyf(e), _side(e))
        got[k] = got.get(k, 0.0) + abs(float(e.get("quantity") or 0))
    contracts_w = {k for k, _ in want}
    contracts_g = {k for k, _ in got}
    if contracts_w != contracts_g:
        diffs.append(f"contract ledger {sorted(map(str, contracts_w))} vs Flex "
                     f"{sorted(map(str, contracts_g))}")
    else:
        for k, q in want.items():
            if k not in got:
                diffs.append(f"side: Flex has no {k[1]} of {k[0]}")
            elif abs(got[k] - q) > 1e-9:
                diffs.append(f"quantity {k[0]} {k[1]}: ledger {q:g} vs Flex {got[k]:g}")
    if diffs or fill.get("price") in (None, ""):
        return diffs + extra
    lp = float(fill["price"])
    if fill["kind"] == "OPT":
        n = fill["legs"][0]["qty"] or 1.0
        net = 0.0
        for e in execs:
            sgn = -1.0 if _side(e) == "BUY" else 1.0      # OPEN credit = sells - buys
            net += sgn * float(e.get("tradePrice") or 0) * abs(float(e.get("quantity") or 0))
        fp = net / n if fill.get("action") == "OPEN" else -net / n
        tol = 0.01
    else:
        q = sum(abs(float(e.get("quantity") or 0)) for e in execs) or 1.0
        fp = sum(float(e.get("tradePrice") or 0) * abs(float(e.get("quantity") or 0))
                 for e in execs) / q
        tol = max(0.005, 1e-4 * abs(fp))
    if abs(fp - lp) > tol:
        diffs.append(f"price: ledger {lp:g} vs Flex {fp:g}")
    return diffs + extra


def _assignment_for(e: dict, date: str) -> str:
    """Detail string if this untagged Flex execution is IB's side of an assignment options-vrp
    recorded (its ledger's ASSIGNED row), else "". Two shapes: the delivered STOCK (BUY,
    100 x contracts, at the short strike) and the assigned short OPTION leg itself. IB books the
    assignment on day T; options-vrp records it at its next run, so the ledger date is on or up to
    DELIVERY_WINDOW_DAYS trading days after the Flex date."""
    for a in assignments:
        if not (a["date"] and date <= a["date"] <= _add_trading_days(date, DELIVERY_WINDOW_DAYS)):
            continue
        cat = e.get("assetCategory")
        if (cat == "STK" and _sym(e.get("symbol", "")) == _sym(a["ticker"]) and _side(e) == "BUY"
                and abs(abs(float(e.get("quantity") or 0)) - a["shares"]) < 1e-9
                and abs(float(e.get("tradePrice") or 0) - a["strike"]) < 0.01):
            return f"delivery of {a['shares']:g} {a['ticker']} at {a['strike']:g} (VRP ASSIGNED row)"
        if (cat == "OPT" and _flex_key(e) == ("OPT", a["ticker"], a["expiry"], a["strike"], "P")
                and abs(abs(float(e.get("quantity") or 0)) - a["contracts"]) < 1e-9):
            return f"assigned short {a['strike']:g}P x{a['contracts']:g} (VRP ASSIGNED row)"
    return ""


def reconcile_executions() -> tuple[list[str], dict[str, int]]:
    """Two-way check; writes tables/exec_reconciliation.csv and tables/exec_backfill.csv.
    Returns (warnings, counts per class)."""
    trades_p = RECORDS / "tables" / "trades.csv"
    flex = []
    if trades_p.exists():
        with open(trades_p, encoding="utf-8") as f:
            flex = [r for r in csv.DictReader(f)
                    if r.get("ibExecID") and r.get("levelOfDetail", "EXECUTION") in ("EXECUTION", "")
                    and r.get("assetCategory") != "BAG"]
    by_id = {r["ibExecID"]: r for r in flex}
    cover = _coverage()
    covered = lambda d: any(a <= d <= b for a, b in cover)  # noqa: E731
    fills, missing_ledgers = _ledger_fills()
    today = datetime.now().strftime("%Y%m%d")
    out, backfill, used = [], [], set()
    counts: dict[str, int] = {}

    def emit(cls, sleeve, date, ref, ids, detail=""):
        counts[cls] = counts.get(cls, 0) + 1
        out.append({"class": cls, "sleeve": sleeve, "date": date, "ref": ref,
                    "exec_ids": ";".join(ids), "detail": detail})

    for fl in fills:
        if fl["exec_ids"]:
            found = [by_id[i] for i in fl["exec_ids"] if i in by_id]
            absent = [i for i in fl["exec_ids"] if i not in by_id]
            used.update(i for i in fl["exec_ids"] if i in by_id)
            legs_found = {_flex_key(e) for e in found}
            if (absent and fl["kind"] == "OPT" and len(absent) == 1
                    and legs_found == {leg["key"] for leg in fl["legs"]}
                    and _is_combo_level_id(absent[0], [e["ibExecID"] for e in found])):
                absent = []                            # the combo-level id (verified rule)
            if absent and not found:
                if covered(fl["date"]):
                    emit("ledger_not_in_flex", fl["sleeve"], fl["date"], fl["ref"], absent,
                         "no execution with these ids in any Flex statement covering this date")
                elif fl["date"] and _add_trading_days(fl["date"], AGEING_DAYS) < today:
                    emit("not_covered_aged", fl["sleeve"], fl["date"], fl["ref"], fl["exec_ids"],
                         f"no statement covers this date after {AGEING_DAYS} business days — is "
                         f"the download stuck or the Flex window too short?")
                else:
                    emit("not_yet_covered", fl["sleeve"], fl["date"], fl["ref"], fl["exec_ids"])
                continue
            if absent and covered(fl["date"]):
                emit("ledger_not_in_flex", fl["sleeve"], fl["date"], fl["ref"], absent,
                     "part of this fill's ids are missing from Flex")
                continue
            diffs = _compare(fl, found)
            if diffs:
                emit("field_mismatch", fl["sleeve"], fl["date"], fl["ref"], fl["exec_ids"],
                     "; ".join(diffs))
            else:
                emit("matched", fl["sleeve"], fl["date"], fl["ref"], fl["exec_ids"])
        elif fl["order_ref"] and fl["date"] >= LINKED_FROM:
            cand = [e for e in flex if e.get("orderReference") == fl["order_ref"]
                    and e["ibExecID"] not in used
                    and any(_flex_key(e) == leg["key"] and _side(e) == leg["side"]
                            for leg in fl["legs"])]
            if cand and not _compare(fl, cand):
                ids = [e["ibExecID"] for e in cand]
                used.update(ids)
                emit("linked_by_tag_only", fl["sleeve"], fl["date"], fl["ref"], ids,
                     "ledger has the tag but no execution ids")
                if fl["sleeve"] == "options-vrp":
                    backfill.append({"sleeve": fl["sleeve"], "date": fl["date"],
                                     "action": fl.get("action", ""), "key": fl.get("key", ""),
                                     "order_ref": fl["order_ref"], "exec_ids": ";".join(ids)})
            elif covered(fl["date"]):
                emit("ledger_not_in_flex", fl["sleeve"], fl["date"], fl["ref"], [],
                     f"tag {fl['order_ref']} has no matching Flex executions"
                     + (f" ({'; '.join(_compare(fl, cand))})" if cand else ""))

    for e in flex:
        if e["ibExecID"] in used or _iso8(e.get("tradeDate") or e.get("dateTime", "")) < LINKED_FROM:
            continue
        strat = e.get("strategy") or _split_ref(e.get("orderReference", ""))[0]
        desc = f"{e.get('symbol')} {_side(e)} {e.get('quantity')} @ {e.get('tradePrice')}"
        date = _iso8(e.get("tradeDate") or e.get("dateTime", ""))
        if strat == "magic-formula" and e.get("assetCategory") == "CASH":
            emit("fx_sweep_unledgered", strat, date, desc, [e["ibExecID"]])
        elif strat:
            emit("flex_tagged_unbooked", strat, date, desc, [e["ibExecID"]],
                 "ledger unavailable" if strat in missing_ledgers
                 else f"tagged {e.get('orderReference')} but no {strat} ledger entry has this id")
        else:
            codes = {c.strip() for c in (e.get("notes") or "").split(";")}
            linked = codes & {"A", "Ex", "Ep"} and _assignment_for(e, date)
            if linked:
                emit("assignment_linked", "options-vrp", date, desc, [e["ibExecID"]], linked)
                continue
            how = ("assignment/exercise delivery" if codes & {"A", "Ex", "Ep"}
                   else "manual trade or tagging failure")
            emit("flex_untagged", e.get("sleeve") or "?", date, desc, [e["ibExecID"]], how)

    tdir = RECORDS / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for name, recs, cols in (("exec_reconciliation", out,
                              ["class", "sleeve", "date", "ref", "exec_ids", "detail"]),
                             ("exec_backfill", backfill,
                              ["sleeve", "date", "action", "key", "order_ref", "exec_ids"])):
        with open(tdir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(recs)
    warn = [f"two-way check incomplete: {s} ledger unavailable ({LEDGERS[s]})"
            for s in missing_ledgers]
    for cls in FAIL_CLASSES:
        bad = [r for r in out if r["class"] == cls]
        if bad:
            r = bad[0]
            warn.append(f"EXECUTIONS {cls}: {len(bad)} (e.g. {r['sleeve']} {r['ref']} "
                        f"[{r['exec_ids']}] {r['detail']})")
    return warn, counts


# ---------------------------------------------------------------------------------------------
# REALISED COMMISSION vs THE COST MODEL (2026-10-07). The options-vrp edge depends on its cost
# guard's per-contract commission assumption (COMMISSION_PER_CONTRACT, $0.65/side). IB's own records
# say what it really costs; tables/commission_by_sleeve.csv shows every sleeve's realised cost per
# unit (share / contract), and options-vrp alerts when realised exceeds the assumption by more than
# COST_WARN_RATIO over a meaningful sample.
ASSUMED_COMMISSION = {"options-vrp": float(os.getenv("VRP_ASSUMED_COMMISSION", "0.65"))}
COST_WARN_RATIO = 1.25
COST_MIN_UNITS = 10


def commission_check() -> list[str]:
    p = RECORDS / "tables" / "trades.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("strategy") and r.get("levelOfDetail", "EXECUTION") in ("EXECUTION", "")
                and _iso8(r.get("tradeDate") or r.get("dateTime", "")) >= LINKED_FROM
                and r.get("assetCategory") != "BAG"]
    agg: dict[tuple, list[float]] = {}
    for r in rows:
        k = (r["strategy"], r.get("assetCategory", ""))
        a = agg.setdefault(k, [0, 0.0, 0.0])
        a[0] += 1
        a[1] += abs(float(r.get("quantity") or 0))
        a[2] += abs(float(r.get("ibCommission") or 0))
    out, warn = [], []
    for (sleeve, cat), (n, units, comm) in sorted(agg.items()):
        per = comm / units if units else 0.0
        assumed = ASSUMED_COMMISSION.get(sleeve) if cat == "OPT" else None
        out.append({"sleeve": sleeve, "asset": cat, "executions": n, "units": units,
                    "commission": round(comm, 2), "per_unit": round(per, 4),
                    "assumed_per_unit": assumed if assumed is not None else "",
                    "ratio": round(per / assumed, 3) if assumed else ""})
        if assumed and units >= COST_MIN_UNITS and per > COST_WARN_RATIO * assumed:
            warn.append(f"COST MODEL: {sleeve} pays ${per:.3f}/contract/side realised vs the guard's "
                        f"${assumed:.2f} assumption ({per / assumed:.2f}x over {units:g} contracts) — "
                        f"the cost guard is under-pricing trades")
    tdir = RECORDS / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "commission_by_sleeve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sleeve", "asset", "executions", "units", "commission",
                                          "per_unit", "assumed_per_unit", "ratio"])
        w.writeheader()
        w.writerows(out)
    return warn


# ---------------------------------------------------------------------------------------------
# HEARTBEAT (2026-10-07). An email that only fires on failure cannot report a job that never
# ran (machine off, wrong path, missing .bat). Two signals on every completed run:
#   - a one-line success email (also on days with no activity), and
#   - an optional GET to a dead-man's-switch URL from HEARTBEAT_URL (never stored in the repo).
#     The ping carries NOTHING but the request itself -- no account, amounts or ids -- and fails
#     quietly: it must never block or fail the download.
def ping_heartbeat() -> bool:
    url = os.getenv("HEARTBEAT_URL", "").strip()
    if not url:
        return False
    try:
        with urlopen(Request(url, headers=UA), timeout=10) as r:
            r.read(64)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"heartbeat ping failed (ignored): {type(e).__name__}")
        return False


def alert(subject: str, body: str) -> None:
    u, p, to = os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"), os.getenv("TO_EMAIL")
    if not (u and p and to):
        log("(no email creds — alert not sent)")
        return
    m = MIMEText(body)
    m["Subject"], m["From"], m["To"] = subject, u, to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(u, p)
        s.sendmail(u, [to], m.as_string())
    log(f"alert emailed to {to}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="rebuild tables from raw only, no download")
    args = ap.parse_args()
    RECORDS.mkdir(parents=True, exist_ok=True)

    if not args.rebuild:
        token, qid = os.getenv("FLEX_TOKEN"), os.getenv("FLEX_QUERY_ID")
        if not (token and qid):
            log("FLEX_TOKEN / FLEX_QUERY_ID not set in .env — nothing downloaded")
            alert("[IB records] NOT CONFIGURED", "FLEX_TOKEN / FLEX_QUERY_ID missing from the live .env.")
            return 2
        try:
            archive(download(token, qid), qid)
        except FlexError as e:
            transient = e.code in RETRY_LATER or e.code == "timeout"
            log(f"FAILED ({'transient — retry later' if transient else 'needs attention'}): {e}")
            hint = ("" if transient else
                    "\n\n1012 = token expired (regenerate in Client Portal > Settings > Flex Web "
                    "Service, update FLEX_TOKEN). 1015/1020 = wrong token or query id.")
            alert(f"[IB records] download failed — {e.code}", f"{e}{hint}")
            return 1
        except Exception as e:  # noqa: BLE001 — network etc.; the archive must never half-write
            log(f"FAILED: {type(e).__name__}: {e}")
            alert("[IB records] download failed", f"{type(e).__name__}: {e}")
            return 1

    counts = rebuild_tables()
    log("tables: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"))
    warns = audit_checks(counts)
    xwarn, xcounts = reconcile_executions()
    warns += xwarn
    warns += commission_check()
    log("executions: " + (", ".join(f"{k} {v}" for k, v in sorted(xcounts.items())) or "none"))
    for w in warns:
        log(f"WARNING: {w}")
    if args.rebuild:
        return 0
    if warns:
        alert("[IB records] audit trail incomplete", "\n".join(warns))
    else:
        alert(f"[IB records] OK {datetime.now():%Y-%m-%d}: "
              f"{xcounts.get('matched', 0)} execution(s) matched, no issues", "")
    ping_heartbeat()
    return 0


if __name__ == "__main__":
    sys.exit(main())
