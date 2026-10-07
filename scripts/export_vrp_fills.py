"""Export SANITISED options-vrp fills from the IB Flex archive, to calibrate the two-way check.

WHY (2026-10-07). The nightly check accepts one missing execution id on a spread whose legs all
matched as the API's combo-level id. That rule was PROVISIONAL until this export's first run
(2026-10-07, three live spreads) showed how Flex reports a combo: only the legs, never the combo
id, which is the legs' .01 sibling -- the rule is now verified and tightened to exactly that. Rerun
this after the first partial combo fill or assignment, and keep the export in
IB-records\\calibration\\. Flags from the nightly check calibrate the rules; never silence them.

Kept, per row: the record type, ibExecID, ibOrderID, orderReference, conid (+ underlying conid),
symbols, asset category, level of detail, side, quantity, price, commission (+ currency),
timestamps, and the option fields (put/call, strike, expiry, multiplier), plus notes/codes and the
exercise/assignment type. The ACCOUNT ID is masked everywhere -- in its own field and anywhere it
appears inside another value. Nothing else is exported.

Rows: every Trade / Order / OptionEAE that is tagged options-vrp, is an option or a combo (OPT,
BAG), or carries an assignment/exercise code. A summary printed at the end shows, per IB order,
which record types and levels of detail appear -- i.e. how the combo and its legs show up.

Run:  python scripts/export_vrp_fills.py [--out vrp_fills_sanitized.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RECORDS = Path(os.getenv("IB_RECORDS_DIR", r"C:\Users\Nicolas\IB-records"))
MASK = "U*******"
TYPES = ("Trade", "Order", "OptionEAE")
KEEP = ["record", "accountId", "transactionID", "tradeID", "ibExecID", "ibOrderID", "orderReference",
        "conid", "underlyingConid", "symbol", "underlyingSymbol", "assetCategory", "levelOfDetail",
        "buySell", "quantity", "tradePrice", "ibCommission", "ibCommissionCurrency", "currency",
        "dateTime", "tradeDate", "orderTime", "putCall", "strike", "expiry", "multiplier",
        "openCloseIndicator", "orderType", "notes", "transactionType"]


def _relevant(a: dict, record: str = "") -> bool:
    codes = {c.strip() for c in (a.get("notes") or "").split(";")}
    return (record == "OptionEAE"                       # exercise / assignment / expiry: always
            or a.get("orderReference", "").startswith("options-vrp")
            or a.get("assetCategory") in ("OPT", "BAG")
            or bool(codes & {"A", "Ex", "Ep"}))


def sanitise(a: dict, record: str, accounts: set[str]) -> dict:
    row = {"record": record}
    for k in KEEP[1:]:
        v = a.get(k, "")
        for acct in accounts:
            if acct and acct in v:
                v = v.replace(acct, MASK)
        row[k] = v
    if row.get("accountId"):
        row["accountId"] = MASK
    return row


def export(records: Path, out: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()
    for p in sorted((records / "raw").rglob("*.xml")):
        root = ET.parse(p).getroot()
        accounts = {s.get("accountId", "") for s in root.iter("FlexStatement")}
        for el in root.iter():
            if el.tag not in TYPES or not el.attrib or not _relevant(el.attrib, el.tag):
                continue
            r = sanitise(dict(el.attrib), el.tag, accounts)
            key = (r["record"], r["transactionID"], r["ibExecID"], r["ibOrderID"], r["levelOfDetail"],
                   r["conid"], r["dateTime"], r["tradeDate"])
            if key in seen:                       # rolling windows overlap: one row per record
                continue
            seen.add(key)
            rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEEP)
        w.writeheader()
        w.writerows(rows)
    shape: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["ibOrderID"]:
            k = f"{r['record']}/{r['assetCategory']}/{r['levelOfDetail'] or '-'}"
            shape.setdefault(r["ibOrderID"], {})
            shape[r["ibOrderID"]][k] = shape[r["ibOrderID"]].get(k, 0) + 1
    return rows, shape


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="vrp_fills_sanitized.csv")
    args = ap.parse_args()
    rows, shape = export(RECORDS, Path(args.out))
    print(f"{len(rows)} row(s) written to {args.out} (account id masked)")
    print("How each IB order appears (record/asset/level -> count):")
    for oid, kinds in sorted(shape.items()):
        print(f"  order {oid}: " + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
