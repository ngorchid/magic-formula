"""Tests for the unseen fee/interest description registry (2026-10-07).

A fee without an instrument whose description matches no market-data keyword goes to "book";
these pin that a NEW description is reported once (with where it went), that a recurring one is
recognised across months and amounts, and that a missing registry seeds silently.
scripts/mutate_description_registry.py seeds the faults. No IB, no network.

Run: python scripts/test_description_registry.py
"""
from __future__ import annotations

import json

from _ib_records_fixtures import cash, check, d, finish, fresh, put

COUNTS = {"trades": 1, "cash_transactions": 1, "open_positions": 1}


def audit():
    d.rebuild_tables()
    return [w for w in d.audit_checks(COUNTS)
            if w.startswith("NEW fee/interest") or w.startswith("description registry")]


print("NORMALISATION — dates, amounts, symbols and ISINs stripped")
n = d.normalise_description
check("month + year forms are stripped",
      n("P01:OPRA FEE FOR OCT 2026") == n("P01:OPRA FEE FOR NOV-2026") == n("P01:OPRA FEE FOR SEP26"),
      f"{n('P01:OPRA FEE FOR OCT 2026')!r} / {n('P01:OPRA FEE FOR NOV-2026')!r}")
check("numeric dates are stripped", n("FEE 2026-10-07") == n("FEE 20261107") == n("FEE 10/07/2026"),
      n("FEE 2026-10-07"))
check("amounts and rates are stripped",
      n("USD CREDIT INT 4.83% ON 12,345.67") == n("USD CREDIT INT 4.50% ON 9,000.00"),
      n("USD CREDIT INT 4.83% ON 12,345.67"))
check("the row's symbol and any ISIN are stripped",
      n("AAPL(US0378331005) ADR FEE", "AAPL") == n("SAP(DE0007164600) ADR FEE", "SAP"),
      n("AAPL(US0378331005) ADR FEE", "AAPL"))
check("different words stay different", n("OPRA FEE") != n("NETWORK C FEE"), "")

print("\nSEEDING — a missing registry is built silently from what is archived")
fresh()
put("20261001", "20261007", "".join([
    cash(1, "Other Fees", "-1.50", "P01:OPRA FEE FOR SEP 2026"),
    cash(2, "Broker Interest Received", "11.4", "USD CREDIT INT FOR SEP-2026"),
    cash(3, "Dividends", "4.0", "XLE CASH DIVIDEND USD 0.80", conid="777", cat="STK"),
]), "2026-10-07T08:30:00")
w = audit()
_rf = d.RECORDS / "description_registry.json"
reg = json.loads(_rf.read_text()) if _rf.exists() else {}
check("the registry file is written", _rf.exists(), str(_rf))
check("first run seeds without alerting", w == [], str(w))
check("...and remembers the fee and the interest line",
      n("P01:OPRA FEE FOR SEP 2026") in reg and n("USD CREDIT INT FOR SEP-2026") in reg, str(reg))
check("dividends are not fee/interest descriptions (not registered)",
      not any("DIVIDEND" in k for k in reg), str(reg))

print("\nRECURRING vs NEW")
put("20261002", "20261108", "".join([
    cash(11, "Other Fees", "-1.50", "P01:OPRA FEE FOR OCT 2026", date="20261102"),
    cash(12, "Broker Interest Received", "9.9", "USD CREDIT INT FOR OCT-2026", date="20261102"),
    cash(13, "Dividends", "5.0", "XLE SPECIAL DIVIDEND", conid="777", cat="STK", date="20261102"),
]), "2026-11-08T08:30:00")
check("next month's OPRA fee and interest line are recognised (no alert)", audit() == [], "")
put("20261103", "20261109", cash(14, "Other Fees", "-1.50",
                                 "OPTIONS PRICE REPORTING AUTHORITY L1 FOR NOV 2026",
                                 date="20261103"), "2026-11-09T08:30:00")
w = audit()
check("a REWORDED market-data fee is reported once, saying it went to book",
      len(w) == 1 and "assigned to book" in w[0] and "OPTIONS PRICE REPORTING" in w[0], str(w))
put("20261104", "20261110", cash(15, "Other Fees", "-1.50",
                                 "OPTIONS PRICE REPORTING AUTHORITY L1 FOR DEC 2026",
                                 date="20261104"), "2026-11-10T08:30:00")
check("...and NOT again (same wording, next month)", audit() == [], "")
put("20261105", "20261111", cash(16, "Price Adjustments", "0.10", "BRAND NEW IB CHARGE",
                                 date="20261105"), "2026-11-11T08:30:00")
w = audit()
check("an unknown row type's description is reported too (category other)",
      len(w) == 1 and "BRAND NEW IB CHARGE" in w[0], str(w))

print("\nA LOST REGISTRY IS NOT SILENT")
_rf = d.RECORDS / "description_registry.json"
if _rf.exists():
    _rf.unlink()
w = [x for x in (d.rebuild_tables() and d.audit_checks(COUNTS)) if "registry was MISSING" in x]
check("a registry missing AFTER it was first seeded alerts (and is re-seeded)",
      len(w) == 1 and (d.RECORDS / "description_registry.json").exists(), str(w))

finish("description-registry")
