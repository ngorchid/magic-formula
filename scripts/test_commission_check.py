"""Tests for the realised-commission check against the options-vrp cost model (2026-10-07).
scripts/mutate_commission_check.py seeds the faults. No IB, no network.

Run: python scripts/test_commission_check.py
"""
from __future__ import annotations

from _ib_records_fixtures import check, d, finish, fresh, put, table, trade

V, M = "options-vrp:20261007-213001", "magic-formula:20261007-160002"


def opt(eid, qty, comm, date="20261007", ref=V):
    return trade(eid, "IWM P263", "9263", ref, "SELL", qty, 1.2, cat="OPT", date=date,
                 underlying="IWM", expiry="20261120", strike="263", right="P", comm=str(-comm))


def run():
    d.rebuild_tables()
    return d.commission_check()


print("COST MODEL — realised VRP commission vs the guard's $0.65/contract/side")
fresh()
put("20261001", "20261008", "".join([
    opt("a1", 8, 5.20), opt("a2", 4, 2.60),                          # 12 contracts @ 0.65
    trade("s1", "AAPL", "265598", M, "BUY", 10, 200.0, comm="-1.00"),
    opt("old", 50, 500.0, date="20261001"),                          # before tagging: ignored
    trade("bag", "IWM BAG", "1", V, "SELL", 2, 0.74, cat="BAG", comm="-99"),   # combo row: ignored
]), "2026-10-08T08:30:00")
w = run()
t = table("commission_by_sleeve")
vrp = next((r for r in t if r["sleeve"] == "options-vrp" and r["asset"] == "OPT"), {})
check("realised VRP cost per contract = 0.65 at the assumption -> no alert",
      w == [] and float(vrp.get("per_unit", 0)) == 0.65 and float(vrp.get("ratio", 0)) == 1.0, str((w, vrp)))
check("units counted in contracts (12), executions (2)", float(vrp.get("units", 0)) == 12
      and int(vrp.get("executions", 0)) == 2, str(vrp))
check("combo (BAG) rows are not tabulated (their commission is on the legs)",
      not any(r["asset"] == "BAG" for r in t), str(t))
import inspect  # noqa: E402
check("main() runs the check on every download", "warns += commission_check()" in inspect.getsource(d.main), "")
check("every sleeve is tabulated (magic's per-share cost shown, no assumption)",
      any(r["sleeve"] == "magic-formula" and float(r["per_unit"]) == 0.1 and r["assumed_per_unit"] == ""
          for r in t), str(t))
fresh()
put("20261001", "20261008", "".join([opt("b1", 8, 8.00), opt("b2", 4, 4.00)]),   # 12 @ 1.00
    "2026-10-08T08:30:00")
w = run()
check("realised 1.00 vs 0.65 assumed (1.54x) over 12 contracts -> COST MODEL alert",
      len(w) == 1 and "COST MODEL" in w[0] and "1.54x" in w[0], str(w))
fresh()
put("20261001", "20261008", opt("c1", 4, 4.00), "2026-10-08T08:30:00")      # only 4 contracts
check("too small a sample (< 10 contracts) does not alert", run() == [], "")
fresh()
put("20261001", "20261008", "".join([opt("e1", 8, 6.24), opt("e2", 4, 3.12)]),   # 12 @ 0.78 = 1.20x
    "2026-10-08T08:30:00")
check("1.20x the assumption (inside the 1.25x tolerance) does not alert", run() == [], "")

finish("commission cost-model")
