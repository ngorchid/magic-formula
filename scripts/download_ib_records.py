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
EMAIL_USER/EMAIL_PASS/TO_EMAIL for the failure alert.

Run:  python scripts/download_ib_records.py            # download + rebuild tables
      python scripts/download_ib_records.py --rebuild  # rebuild tables from raw only (offline)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import smtplib
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
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


def _instrument_sleeve(row: dict, owners: dict[str, str]) -> str:
    return (owners.get(row.get("conid", ""))
            or ASSET_SLEEVE.get(row.get("assetCategory", ""), "unassigned"))


def attribute(row: dict, category: str, owners: dict[str, str], base: str) -> str:
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
    if category == "fee" and not row.get("conid"):
        return "options-vrp" if any(w in desc for w in MARKET_DATA_WORDS) else "book"
    if row.get("conid") or row.get("assetCategory"):
        return _instrument_sleeve(row, owners)
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
    acct = next(iter(rows.get("account_information", {}).values()), {})
    base = acct.get("currency") or "USD"
    for t in rows.get("trades", {}).values():
        t["sleeve"] = t.get("strategy") or _instrument_sleeve(t, owners)
        t["category"] = "trade"
    for r in rows.get("cash_transactions", {}).values():
        r["category"] = _category(r, CASH_TYPE_CATEGORY.get(r.get("type", ""), "other"))
        r["sleeve"] = attribute(r, r["category"], owners, base)
    for r in rows.get("statement_of_funds", {}).values():
        code = r.get("activityCode", "")
        r["category"] = _category(r, SOF_CODE_CATEGORY.get(code, "other" if code else "balance"))
        r["sleeve"] = "-" if r["category"] == "balance" else attribute(r, r["category"], owners, base)
    for name, cat in (("corporate_actions", "corporate_action"), ("option_exercises", "option_event"),
                      ("transaction_taxes", "transaction_tax"), ("fx_transactions", "fx")):
        for r in rows.get(name, {}).values():
            r["category"] = cat
            r["sleeve"] = (_instrument_sleeve(r, owners) if cat != "fx"
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
    warn += reconcile_latest()
    return warn


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
    for w in warns:
        log(f"WARNING: {w}")
    if warns and not args.rebuild:
        alert("[IB records] audit trail incomplete", "\n".join(warns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
