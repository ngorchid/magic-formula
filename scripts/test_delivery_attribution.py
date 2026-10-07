"""Tests for the delivered-shares attribution rule (owner's rule, 2026-10-07).

Stock delivered by an options-vrp assignment/exercise belongs to options-vrp -- the delivery row
and every later row on that stock (dividends, the sale) -- when the stock row lands within
DELIVERY_WINDOW_DAYS trading days of the event and magic-formula has no tagged trade in it. If
magic-formula has, the row is "conflict". Rows from before the event keep the normal rule.
scripts/mutate_delivery_attribution.py seeds the faults. No IB, no network.

Run: python scripts/test_delivery_attribution.py
"""
from __future__ import annotations

from _ib_records_fixtures import cash, check, d, eae, finish, fresh, put, table, trade

V, M = "options-vrp:20261006-213001", "magic-formula:20261006-160002"


def opt(eid, conid, und, strike, side="SELL", date="20261002"):
    return trade(eid, f"{und} P{strike}", conid, V, side, 2, 1.0, cat="OPT", date=date,
                 underlying=und, expiry="20261120", strike=str(strike), right="P")


def sleeve_of(name, field, value):
    r = next((x for x in table(name) if x.get(field) == value), {})
    return r.get("sleeve")


print("ASSIGNED SHORT PUT -> the delivered stock belongs to options-vrp")
fresh()
put("20260928", "20261016", "".join([
    opt("o1", "901", "IWM", 263),                                    # VRP sold the put
    eae("ea1", "901", "9", "IWM P263", "Assignment", date="20261007"),
    trade("d1", "IWM", "9", "", "BUY", 200, 263.0, date="20261007", notes="A"),   # delivery
    cash("c1", "Dividends", "40", "IWM CASH DIVIDEND", conid="9", cat="STK", date="20261015"),
    cash("c0", "Dividends", "35", "IWM CASH DIVIDEND", conid="9", cat="STK", date="20261001"),
    # an event with NO delivery inside the window: a later unrelated stock trade stays normal
    opt("o2", "902", "XLE", 90),
    eae("ea2", "902", "8", "XLE P90", "Assignment", date="20261002"),
    trade("d2", "XLE", "8", "", "BUY", 100, 90.0, date="20261009"),   # 5 trading days later
    # an expiration is not a delivery
    opt("o3", "903", "GLD", 380),
    eae("ea3", "903", "7", "GLD P380", "Expiration", date="20261007"),
    trade("d3", "GLD", "7", "", "BUY", 10, 380.0, date="20261007"),
    # an option another sleeve owns (tagged trend-overlay) -> its assignment is NOT a VRP delivery
    trade("o6", "BAC P30", "906", "trend-overlay:20261002-203001", "SELL", 1, 1.0, cat="OPT",
          date="20261002", underlying="BAC", expiry="20261120", strike="30", right="P"),
    eae("ea6", "906", "4", "BAC P30", "Assignment", date="20261007"),
    trade("d6", "BAC", "4", "", "BUY", 100, 30.0, date="20261007", notes="A"),
    # magic-formula ALSO trades the stock -> conflict, never silently picked
    opt("o4", "904", "XOM", 110),
    trade("m4", "XOM", "6", M, "BUY", 5, 112.0, date="20261001"),
    eae("ea4", "904", "6", "XOM P110", "Assignment", date="20261007"),
    trade("d4", "XOM", "6", "", "BUY", 200, 110.0, date="20261008"),
]), "2026-10-16T08:30:00")
d.rebuild_tables()
check("the delivery trade -> options-vrp", sleeve_of("trades", "ibExecID", "d1") == "options-vrp",
      sleeve_of("trades", "ibExecID", "d1"))
check("a LATER dividend on the delivered stock -> options-vrp",
      sleeve_of("cash_transactions", "transactionID", "c1") == "options-vrp",
      sleeve_of("cash_transactions", "transactionID", "c1"))
check("a dividend from BEFORE the assignment keeps the normal rule (magic-formula)",
      sleeve_of("cash_transactions", "transactionID", "c0") == "magic-formula",
      sleeve_of("cash_transactions", "transactionID", "c0"))
check("a stock trade OUTSIDE the delivery window keeps the normal rule",
      sleeve_of("trades", "ibExecID", "d2") == "magic-formula", sleeve_of("trades", "ibExecID", "d2"))
check("an EXPIRATION delivers nothing (normal rule)",
      sleeve_of("trades", "ibExecID", "d3") == "magic-formula", sleeve_of("trades", "ibExecID", "d3"))
check("an assignment on an option ANOTHER sleeve owns is not a VRP delivery",
      sleeve_of("trades", "ibExecID", "d6") == "magic-formula", sleeve_of("trades", "ibExecID", "d6"))
check("magic-formula has a tagged trade in the stock -> 'conflict'",
      sleeve_of("trades", "ibExecID", "d4") == "conflict", sleeve_of("trades", "ibExecID", "d4"))
check("...and the audit flags it", any("trades row(s) not attributable" in w for w in
                                       d.audit_checks({"trades": 1, "cash_transactions": 1,
                                                       "open_positions": 1})), "")
check("the window is DELIVERY_WINDOW_DAYS = 3 trading days (over a weekend)",
      d.DELIVERY_WINDOW_DAYS == 3 and d._add_trading_days("20261009", 3) == "20261014",
      d._add_trading_days("20261009", 3))

print("\nFINGERPRINT — 100 x contracts at exactly the strike; the window is the fallback")
fresh()
put("20260928", "20261016", "".join([
    opt("o7", "907", "NKE", 70),
    eae("ea7", "907", "3", "NKE P70", "Assignment", date="20261007", strike="70", qty="2"),
    trade("f7", "NKE", "3", "", "BUY", 200, 70.0, date="20261007", notes="A"),     # 2 x 100 @ 70
    opt("o8", "908", "PFE", 25),
    eae("ea8", "908", "2", "PFE P25", "Assignment", date="20261007", strike="25", qty="3"),
    trade("f8", "PFE", "2", "", "BUY", 150, 25.0, date="20261008"),               # right price, wrong qty
    opt("o9", "909", "BAC", 40),
    eae("ea9", "909", "1", "BAC P40", "Assignment", date="20261007", strike="40", qty="2"),
    trade("f9", "BAC", "1", "", "BUY", 200, 39.5, date="20261007"),               # right qty, wrong price
]), "2026-10-16T08:30:00")
d.rebuild_tables()
f7 = next((x for x in table("trades") if x.get("ibExecID") == "f7"), {})
f8 = next((x for x in table("trades") if x.get("ibExecID") == "f8"), {})
check("a delivery of exactly 100 x contracts at the strike is labelled 'fingerprint'",
      f7.get("delivery_match") == "fingerprint" and f7.get("sleeve") == "options-vrp", str(f7))
f9 = next((x for x in table("trades") if x.get("ibExecID") == "f9"), {})
check("a stock row in the window that is NOT the print is kept by the fallback, labelled 'window'",
      f8.get("delivery_match") == "window" and f8.get("sleeve") == "options-vrp", str(f8))
check("...both when only the price matches (f8) and when only the quantity matches (f9)",
      f9.get("delivery_match") == "window" and f9.get("sleeve") == "options-vrp", str(f9))

print("\nEDGE OF THE WINDOW")
fresh()
put("20260928", "20261016", "".join([
    opt("o5", "905", "BAC", 40),
    eae("ea5", "905", "5", "BAC P40", "Assignment", date="20261009"),          # a Friday
    trade("d5", "BAC", "5", "", "BUY", 200, 40.0, date="20261014"),          # +3 trading days
]), "2026-10-16T08:30:00")
d.rebuild_tables()
check("a delivery on the LAST day of the window still counts",
      sleeve_of("trades", "ibExecID", "d5") == "options-vrp", sleeve_of("trades", "ibExecID", "d5"))

finish("delivered-shares attribution")
