"""Tests for the nightly two-way check: Flex executions <-> the sleeves' ledgers.

Every class of `reconcile_executions` is driven by a synthetic Flex statement and synthetic
ledgers built so that breaking THAT rule fails its case; scripts/mutate_exec_reconcile.py seeds
those faults. No IB, no network.

Run: python scripts/test_exec_reconcile.py
"""
from __future__ import annotations

from _ib_records_fixtures import (At, check, d, empty_ledgers, finish, fresh, ledger, put, table,
                                  trade)

M, T, V = "magic-formula:20261007-160002", "trend-overlay:20261007-203001", \
    "options-vrp:20261007-213001"


def run():
    d.rebuild_tables()
    try:
        return d.reconcile_executions()
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        return [f"crashed: {type(e).__name__}: {e}"], {}


def classes(name="exec_reconciliation"):
    return [r["class"] for r in table(name)]


def row(cls, needle=""):
    return next((r for r in table("exec_reconciliation")
                 if r["class"] == cls and needle in (r["ref"] + r["exec_ids"])), {})


def opt(eid, ref, side, qty, price, strike, date="20261007"):
    return trade(eid, f"IWM 261120P00{int(strike)}000", f"9{int(strike)}", ref, side, qty, price,
                 cat="OPT", date=date, underlying="IWM", expiry="20261120", strike=str(strike),
                 right="P")


# =============================================================================================
print("CLEAN BOOK — every ledger fill matches, nothing alerts")
fresh()
put("20261006", "20261008", "".join([
    trade("e1", "AAPL", "265598", M, "BUY", 3, 10.0),
    trade("e2a", "MSFT", "272093", M, "SELL", 1, 19.9),            # one order, two executions
    trade("e2b", "MSFT", "272093", M, "SELL", 1, 20.1),
    trade("fx1", "EUR.USD", "12087792", M, "BUY", 1000, 1.1, cat="CASH"),
    trade("e3", "MCLZ6", "661016519", T, "BUY", 1, 88.48, cat="FUT", underlying="MCL",
          expiry="20261119"),
    opt("e4s", V, "SELL", 2, 1.20, 263),
    opt("e4l", V, "BUY", 2, 0.46, 256),
    trade("old", "XOM", "13977", "", "BUY", 5, 100.0, date="20260930"),   # before tagging
]), "2026-10-08T08:30:00")
ledger("magic-formula", {
    "positions": [{"ticker": "AAPL", "shares": 3, "entry_price": 10.0, "entry_date": "2026-10-07",
                   "entry_order_ref": M, "entry_exec_ids": ["e1"]}],
    "trade_log": [{"ticker": "MSFT", "shares": 2, "entry_price": 15.0, "entry_date": "2026-09-01",
                   "entry_order_ref": "", "entry_exec_ids": [], "exit_price": 20.0,
                   "exit_date": "2026-10-07", "exit_order_ref": M,
                   "exit_exec_ids": ["e2a", "e2b"]}]})
ledger("trend-overlay", {"trade_log": [
    {"date": "2026-10-07", "market": "oil", "symbol": "MCL", "expiry": "20261119",
     "signed_qty": 1, "price": 88.48, "order_ref": T, "exec_ids": ["e3"]},
    {"date": "2026-10-07", "market": "rates_10y", "symbol": "", "expiry": "", "signed_qty": 1,
     "price": 112.0, "reason": "RESYNC ledger to broker (-1 -> 0)"}]})              # bookkeeping
ledger("options-vrp", {"trade_log": [
    {"date": "2026-10-07", "action": "OPEN", "key": "IWM_2026-11-20_263_256", "contracts": 2,
     "credit": 0.74, "order_ref": V, "exec_ids": ["bag4", "e4s", "e4l"]}]})
warn, counts = run()
check("a clean book raises NO warning", warn == [], str(warn))
check("magic BUY matched by execution id", row("matched", "e1") != {}, str(table("exec_reconciliation")))
check("one fill of TWO executions matches on the volume-weighted price",
      row("matched", "e2a") != {}, str(row("field_mismatch", "e2a")))
check("trend future matched (root symbol + expiry vs Flex underlying + expiry)",
      row("matched", "e3") != {}, str(row("field_mismatch", "e3")))
check("VRP spread matched at LEG level, net credit = sells - buys",
      row("matched", "e4s") != {}, str(table("exec_reconciliation")))
check("an FX sweep is reported as unledgered BY DESIGN, not as a failure",
      row("fx_sweep_unledgered", "fx1") != {} and counts.get("flex_tagged_unbooked", 0) == 0,
      str(counts))
check("an execution from before tagging is ignored", "old" not in str(table("exec_reconciliation")),
      str(table("exec_reconciliation")))
check("a trend RESYNC row (bookkeeping, no tag, no ids) is never reported",
      all("rates_10y" not in r["ref"] for r in table("exec_reconciliation")),
      str(table("exec_reconciliation")))
check("the pre-tagging half of a closed trade (no ids, no tag) is not reported",
      all("MSFT BUY" not in r["ref"] for r in table("exec_reconciliation")),
      str(table("exec_reconciliation")))

# =============================================================================================
print("\nFIELD MISMATCHES — matched id, different facts")
fresh()
put("20261006", "20261008", "".join([
    trade("q1", "MCLZ6", "661016519", T, "BUY", 1, 88.0, cat="FUT", underlying="MCL",
          expiry="20261119"),
    trade("p1", "NVDA", "4815747", M, "BUY", 2, 101.0),
    trade("c1", "CVX", "13981", M, "BUY", 4, 150.0),
    trade("s1", "AMD", "4391", M, "SELL", 4, 150.0),
    opt("o1s", V, "SELL", 2, 1.20, 263), opt("o1l", V, "BUY", 2, 0.46, 256),
]), "2026-10-08T08:30:00")
ledger("trend-overlay", {"trade_log": [
    {"date": "2026-10-07", "symbol": "MCL", "expiry": "20261119", "signed_qty": 2, "price": 88.0,
     "order_ref": T, "exec_ids": ["q1"]}]})
ledger("magic-formula", {"positions": [
    {"ticker": "NVDA", "shares": 2, "entry_price": 100.0, "entry_date": "2026-10-07",
     "entry_order_ref": M, "entry_exec_ids": ["p1"]},
    {"ticker": "XOM", "shares": 4, "entry_price": 150.0, "entry_date": "2026-10-07",
     "entry_order_ref": M, "entry_exec_ids": ["c1"]},
    {"ticker": "AMD", "shares": 4, "entry_price": 150.0, "entry_date": "2026-10-07",
     "entry_order_ref": M, "entry_exec_ids": ["s1"]}], "trade_log": []})
ledger("options-vrp", {"trade_log": [
    {"date": "2026-10-07", "action": "OPEN", "key": "IWM_2026-11-20_263_256", "contracts": 2,
     "credit": 0.80, "order_ref": V, "exec_ids": ["o1s", "o1l"]}]})
warn, counts = run()
check("quantity mismatch caught (ledger 2, Flex 1)", "quantity" in row("field_mismatch", "q1").get("detail", ""),
      str(row("field_mismatch", "q1")))
check("price mismatch caught (ledger 100, Flex 101)", "price" in row("field_mismatch", "p1").get("detail", ""),
      str(row("field_mismatch", "p1")))
check("contract mismatch caught (ledger XOM, Flex CVX)", "contract" in row("field_mismatch", "c1").get("detail", ""),
      str(row("field_mismatch", "c1")))
check("side mismatch caught (ledger BUY, Flex SELL)", row("field_mismatch", "s1") != {},
      str(table("exec_reconciliation")))
check("spread net-credit mismatch caught (ledger 0.80, Flex 0.74)",
      "price" in row("field_mismatch", "o1s").get("detail", ""), str(row("field_mismatch", "o1s")))
check("mismatches ALERT", any("field_mismatch: 5" in w for w in warn), str(warn))

# =============================================================================================
print("\nONE-SIDED — in Flex only, or in a ledger only")
fresh()
put("20261006", "20261008", "".join([
    trade("u1", "MCLZ6", "661016519", T, "BUY", 1, 88.0, cat="FUT", underlying="MCL",
          expiry="20261119"),
    trade("m1", "TSLA", "76792991", "", "BUY", 1, 250.0),
    trade("a1", "IWM", "9", "", "BUY", 200, 263.0, notes="A"),
]), "2026-10-08T08:30:00")
empty_ledgers()
ledger("magic-formula", {"positions": [
    {"ticker": "KO", "shares": 1, "entry_price": 60.0, "entry_date": "2026-10-07",
     "entry_order_ref": M, "entry_exec_ids": ["gone"]},
    {"ticker": "PEP", "shares": 1, "entry_price": 60.0, "entry_date": "2026-10-20",
     "entry_order_ref": M, "entry_exec_ids": ["later"]}], "trade_log": []})
warn, counts = run()
check("tagged Flex execution no ledger booked -> flex_tagged_unbooked (a sleeve failed to book)",
      row("flex_tagged_unbooked", "u1").get("sleeve") == "trend-overlay", str(table("exec_reconciliation")))
check("untagged Flex execution -> flex_untagged, labelled manual/tagging failure",
      "manual" in row("flex_untagged", "m1").get("detail", ""), str(row("flex_untagged", "m1")))
check("an assignment delivery is labelled as such", "assignment" in row("flex_untagged", "a1").get("detail", ""),
      str(row("flex_untagged", "a1")))
check("ledger id missing from Flex on a COVERED date -> ledger_not_in_flex",
      row("ledger_not_in_flex", "gone") != {}, str(table("exec_reconciliation")))
check("...but on a date Flex does not cover yet -> not_yet_covered, no alert",
      row("not_yet_covered", "later") != {} and not any("later" in w for w in warn),
      str(table("exec_reconciliation")))
check("each failure class alerts", all(any(c in w for w in warn) for c in
                                       ("flex_tagged_unbooked", "flex_untagged", "ledger_not_in_flex")),
      str(warn))

# =============================================================================================
print("\nTAG-ONLY ROWS — VRP self-heal: linked, then backfilled by options-vrp itself")
fresh()
# One run tags ALL its orders with the same tag: the XLE spread below was opened by the same run,
# so linking on the tag alone (without contract + side) would grab its legs too.
xle = (trade("x1", "XLE P90", "990", V, "SELL", 3, 1.10, cat="OPT", underlying="XLE",
             expiry="20261120", strike="90", right="P")
       + trade("x2", "XLE P85", "985", V, "BUY", 3, 0.40, cat="OPT", underlying="XLE",
               expiry="20261120", strike="85", right="P"))
put("20261006", "20261008", "".join([
    opt("h1", V, "BUY", 2, 0.30, 263), opt("h2", V, "SELL", 2, 0.10, 256), xle,
]), "2026-10-08T08:30:00")
empty_ledgers()
ledger("options-vrp", {"trade_log": [
    # the tag-only row comes FIRST, so its link is decided before the XLE ids are claimed
    {"date": "2026-10-07", "action": "CLOSE", "key": "IWM_2026-11-20_263_256", "contracts": 2,
     "close_value": 0.20, "order_ref": V, "exec_ids": []},
    {"date": "2026-10-07", "action": "OPEN", "key": "XLE_2026-11-20_90_85", "contracts": 3,
     "credit": 0.70, "order_ref": V, "exec_ids": ["xbag", "x1", "x2"]},
    {"date": "2026-10-07", "action": "CLOSE", "key": "IWM_2026-11-20_250_245", "contracts": 1,
     "close_value": 0.20, "order_ref": "options-vrp:NOPE", "exec_ids": []}]})
warn, counts = run()
lk = row("linked_by_tag_only", "IWM_2026-11-20_263_256")
check("a tag-only CLOSE links to its two Flex legs on tag + contract + side + quantity",
      lk != {} and set(lk["exec_ids"].split(";")) == {"h1", "h2"}, str(table("exec_reconciliation")))
bf = table("exec_backfill")
check("...and its real execution ids are written for options-vrp to backfill",
      len(bf) == 1 and set(bf[0]["exec_ids"].split(";")) == {"h1", "h2"}
      and bf[0]["key"] == "IWM_2026-11-20_263_256" and bf[0]["action"] == "CLOSE"
      and bf[0]["order_ref"] == V, str(bf))
check("linked executions are NOT also reported as unbooked", counts.get("flex_tagged_unbooked", 0) == 0,
      str(counts))
check("the same run tag on ANOTHER spread does not leak into the link (contract + side matter)",
      row("matched", "x1") != {} and "x1" not in lk.get("exec_ids", ""), str(table("exec_reconciliation")))
check("a tag-only row with no matching Flex executions -> ledger_not_in_flex",
      row("ledger_not_in_flex", "250_245") != {}, str(table("exec_reconciliation")))

# =============================================================================================
print("\nCOMBO RULE IS NARROW — one missing id is tolerated only when BOTH legs matched")
fresh()
put("20261006", "20261008", opt("k1", V, "SELL", 2, 1.20, 263), "2026-10-08T08:30:00")
empty_ledgers()
ledger("options-vrp", {"trade_log": [
    {"date": "2026-10-07", "action": "OPEN", "key": "IWM_2026-11-20_263_256", "contracts": 2,
     "credit": 0.74, "order_ref": V, "exec_ids": ["k1", "kgone"]}]})
warn, counts = run()
check("a spread with only ONE leg in Flex is a missing id, not 'the combo-level id'",
      row("ledger_not_in_flex", "kgone") != {}, str(table("exec_reconciliation")))

print("\nSPELLINGS AND OLDER ROW SHAPES")
fresh()
nobs = trade("n1", "MCLZ6", "661016519", T, "BUY", 1, 88.0, cat="FUT", underlying="MCL",
             expiry="20261119").replace(' buySell="BUY"', "")
put("20261006", "20261008", "".join([
    trade("eu1", "VOLV.B", "123", M, "BUY", 10, 250.0),       # IB spells the share class with a dot
    nobs,                                                      # a Flex row with no buySell field
]), "2026-10-08T08:30:00")
ledger("magic-formula", {"positions": [
    {"ticker": "VOLV-B.ST", "shares": 10, "entry_price": 250.0, "entry_date": "2026-10-07",
     "entry_order_ref": M, "entry_exec_ids": ["eu1"]}], "trade_log": []})
ledger("trend-overlay", {"trade_log": [
    {"date": "2026-10-07", "symbol": "MCL", "expiry": "20261119", "signed_qty": 1, "price": 88.0,
     "order_ref": T, "exec_ids": ["n1"]}]})
ledger("options-vrp", {"trade_log": [
    {"date": "2026-10-07", "action": "SNAPSHOT_NOTE", "key": "IWM_2026-11-20_263_256"},
    {"date": "2026-10-07", "action": "OPEN", "key": "malformed-key", "contracts": 1}]})
warn, counts = run()
check("a yfinance ticker with exchange suffix + hyphen class (VOLV-B.ST) matches IB's VOLV.B",
      row("matched", "eu1") != {}, str(table("exec_reconciliation")))
check("a Flex row without buySell takes its side from the quantity sign",
      row("matched", "n1") != {}, str(table("exec_reconciliation")))
check("ledger rows that are not fills (other actions, malformed keys) are skipped, not reported",
      warn == [] and not any("malformed" in r["ref"] or "SNAPSHOT" in r["ref"]
                             for r in table("exec_reconciliation")), str(warn))
fresh()
put("20261006", "20261008", "", "2026-10-08T08:30:00")
empty_ledgers()
d.LEDGERS["options-vrp"].write_text("{not json")
warn, counts = run()
check("an unreadable ledger is reported as unavailable (not as empty)",
      any("options-vrp ledger unavailable" in w for w in warn), str(warn))

print("\nAGEING — an uncovered ledger id alerts after 3 business days")
fresh()
put("20261006", "20261008", "", "2026-10-08T08:30:00")
empty_ledgers()
ledger("magic-formula", {"positions": [
    {"ticker": "KO", "shares": 1, "entry_price": 60.0, "entry_date": "2026-10-09",
     "entry_order_ref": M, "entry_exec_ids": ["k9"]}], "trade_log": []})
with At("2026-10-13T08:30:00"):                       # Fri 9th + 2 business days
    warn, counts = run()
check("2 business days uncovered -> not_yet_covered, no alert",
      counts.get("not_yet_covered") == 1 and not warn, str((counts, warn)))
with At("2026-10-15T08:30:00"):                       # Fri 9th + 4 business days
    warn, counts = run()
check("4 business days uncovered -> not_covered_aged, ALERTS",
      counts.get("not_covered_aged") == 1 and any("not_covered_aged" in w for w in warn),
      str((counts, warn)))

print("\nASSIGNMENT — IB's side of a VRP assignment links to the sleeve's ASSIGNED rows")
fresh()
K = "IWM_2026-11-20_263_256"
put("20261006", "20261009", "".join([
    trade("dlv", "IWM", "9", "", "BUY", 200, 263.0, date="20261007", notes="A"),   # delivery
    opt("asn", "", "BUY", 2, 0.0, 263, date="20261007").replace('notes=""', 'notes="A"'),
    trade("ss1", "IWM", "9", V, "SELL", 200, 255.0, date="20261008"),             # unwind: shares
    opt("ls1", V, "SELL", 2, 1.5, 256, date="20261008"),                          # unwind: long put
    trade("odd", "XLE", "8", "", "BUY", 100, 90.0, date="20261007", notes="A"),    # no VRP row
    trade("dq", "IWM", "9", "", "BUY", 300, 263.0, date="20261007", notes="A"),    # wrong quantity
    trade("dp", "IWM", "9", "", "BUY", 200, 250.0, date="20261007", notes="A"),    # wrong price
]), "2026-10-09T08:30:00")
put("20261007", "20261009", trade("dold", "IWM", "9", "", "BUY", 200, 263.0, date="20261009",
                                  notes="A"), "2026-10-09T09:30:00")   # AFTER the VRP row: no link
empty_ledgers()
ledger("options-vrp", {"trade_log": [
    {"date": "2026-10-08", "action": "ASSIGNED", "key": K, "contracts": 2, "strike": 263.0,
     "shares": 200},
    {"date": "2026-10-08", "action": "ASSIGNED_STOCK_SOLD", "key": K, "contracts": 2, "shares": 200,
     "price": 255.0, "order_ref": V, "exec_ids": ["ss1"]},
    {"date": "2026-10-08", "action": "ASSIGNED_LONG_SOLD", "key": K, "contracts": 2, "strike": 256.0,
     "price": 1.5, "order_ref": V, "exec_ids": ["ls1"]}]})
warn, counts = run()
check("the delivered shares (200 @ the 263 strike) link to the ASSIGNED row, no alert",
      row("assignment_linked", "dlv") != {}, str(table("exec_reconciliation")))
check("the assigned short option leg links too", row("assignment_linked", "asn") != {},
      str(table("exec_reconciliation")))
check("the unwind's share sale and long-put sale match their ledger rows",
      row("matched", "ss1") != {} and row("matched", "ls1") != {}, str(table("exec_reconciliation")))
check("an assignment-coded delivery with NO VRP ledger row still alerts",
      row("flex_untagged", "odd") != {} and any("flex_untagged: 4" in w for w in warn), str(warn))
check("...as does one with the wrong QUANTITY, the wrong PRICE, or outside the date window",
      all(row("flex_untagged", x) != {} for x in ("dq", "dp", "dold")), str(table("exec_reconciliation")))

print("\nNEW LEDGER FIELDS — conid, commission, currency (additive)")
fresh()
put("20261006", "20261008", "".join([
    trade("c1", "XYZ", "555", M, "BUY", 5, 40.0, comm="-1.00", ccy="EUR"),     # IB spells it XYZ
    trade("c2", "AAPL", "265598", M, "BUY", 5, 200.0, comm="-1.00"),
    trade("c3", "MSFT", "272093", M, "BUY", 5, 300.0, comm="-1.00"),
    trade("c4", "SAP", "1234", M, "BUY", 5, 100.0, comm="-1.00", ccy="EUR"),
    opt("v1", V, "SELL", 2, 1.20, 263).replace('conid="9263"', 'conid="7001"').replace('ibCommission="-1"', 'ibCommission="-1.30"'),
    opt("v2", V, "BUY", 2, 0.46, 256).replace('conid="9256"', 'conid="7002"').replace('ibCommission="-1"', 'ibCommission="-1.30"'),
]), "2026-10-08T08:30:00")
empty_ledgers()
ledger("magic-formula", {"positions": [
    {"ticker": "ABC.DE", "shares": 5, "entry_price": 40.0, "entry_date": "2026-10-07", "currency": "EUR",
     "entry_order_ref": M, "entry_exec_ids": ["c1"], "entry_conid": 555, "entry_commission": 1.00},
    {"ticker": "AAPL", "shares": 5, "entry_price": 200.0, "entry_date": "2026-10-07", "currency": "USD",
     "entry_order_ref": M, "entry_exec_ids": ["c2"], "entry_conid": 265598, "entry_commission": 3.50},
    {"ticker": "MSFT", "shares": 5, "entry_price": 300.0, "entry_date": "2026-10-07", "currency": "USD",
     "entry_order_ref": M, "entry_exec_ids": ["c3"], "entry_conid": 999, "entry_commission": 1.00},
    {"ticker": "SAP.DE", "shares": 5, "entry_price": 100.0, "entry_date": "2026-10-07", "currency": "USD",
     "entry_order_ref": M, "entry_exec_ids": ["c4"], "entry_conid": 1234, "entry_commission": 1.00}],
    "trade_log": []})
ledger("options-vrp", {"trade_log": [
    {"date": "2026-10-07", "action": "OPEN", "key": "IWM_2026-11-20_263_256", "contracts": 2,
     "credit": 0.74, "order_ref": V, "exec_ids": ["bagv", "v1", "v2"], "conids": [7001, 7002],
     "commission": 2.60, "currency": "USD"}]})
warn, counts = run()
check("a recorded conid replaces the spelling match (ledger ABC.DE, IB XYZ, same conid)",
      row("matched", "c1") != {}, str(table("exec_reconciliation")))
check("a commission difference is reported (ledger 3.50, Flex 1.00)",
      "commission" in row("field_mismatch", "c2").get("detail", ""), str(row("field_mismatch", "c2")))
check("a conid difference is reported", "contract" in row("field_mismatch", "c3").get("detail", "")
      or "CONID" in row("field_mismatch", "c3").get("detail", ""), str(row("field_mismatch", "c3")))
check("a currency difference is reported", "currency" in row("field_mismatch", "c4").get("detail", ""),
      str(row("field_mismatch", "c4")))
check("a VRP spread with leg conids + commission matches (2 x 1.30 = 2.60)",
      row("matched", "v1") != {}, str(table("exec_reconciliation")))

print("\nINCOMPLETE INPUT")
fresh()
put("20261006", "20261008", trade("z1", "MCLZ6", "661016519", T, "BUY", 1, 88.0, cat="FUT",
                                  underlying="MCL", expiry="20261119"), "2026-10-08T08:30:00")
ledger("magic-formula", {"positions": [], "trade_log": []})
warn, counts = run()
check("a missing ledger is reported, not silently treated as empty",
      any("trend-overlay ledger unavailable" in w for w in warn), str(warn))
check("...and that sleeve's executions say so", "ledger unavailable" in
      row("flex_tagged_unbooked", "z1").get("detail", ""), str(row("flex_tagged_unbooked", "z1")))

finish("two-way execution")
