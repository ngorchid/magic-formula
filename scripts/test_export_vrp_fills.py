"""Tests for scripts/export_vrp_fills.py: only VRP-relevant rows, only the agreed fields, the
account id masked everywhere, and a summary of how each combo order appears.
scripts/mutate_export_vrp_fills.py seeds the faults. No IB, no network.

Run: python scripts/test_export_vrp_fills.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_vrp_fills as ex  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


ACCT = "U27760647"
rec = Path(tempfile.mkdtemp())
(rec / "raw" / "2026").mkdir(parents=True)
xml = (f'<FlexQueryResponse><FlexStatements><FlexStatement accountId="{ACCT}" fromDate="20261001" '
       f'toDate="20261007"><Trades>'
       f'<Trade accountId="{ACCT}" transactionID="1" ibExecID="e1" ibOrderID="77" orderReference="options-vrp:R" '
       f'conid="9263" symbol="IWM 261120P00263000" underlyingSymbol="IWM" assetCategory="OPT" '
       f'levelOfDetail="EXECUTION" buySell="SELL" quantity="-2" tradePrice="1.2" ibCommission="-1.3" '
       f'putCall="P" strike="263" expiry="20261120" dateTime="20261007;153001" '
       f'description="client {ACCT} memo" fifoPnlRealized="12.5"/>'
       f'<Trade accountId="{ACCT}" transactionID="2" ibExecID="e2" ibOrderID="77" orderReference="options-vrp:R" '
       f'conid="9256" assetCategory="OPT" levelOfDetail="EXECUTION" buySell="BUY" quantity="2" '
       f'tradePrice="0.46" putCall="P" strike="256" expiry="20261120"/>'
       f'<Trade accountId="{ACCT}" transactionID="3" ibOrderID="77" orderReference="options-vrp:R" '
       f'conid="1" assetCategory="BAG" levelOfDetail="ORDER" buySell="BUY" quantity="2" tradePrice="-0.74"/>'
       f'<Trade accountId="{ACCT}" transactionID="4" ibExecID="m1" ibOrderID="88" '
       f'orderReference="magic-formula:R" conid="265598" assetCategory="STK" levelOfDetail="EXECUTION"/>'
       f'<Trade accountId="{ACCT}" transactionID="5" ibExecID="d1" conid="9" assetCategory="STK" '
       f'levelOfDetail="EXECUTION" buySell="BUY" quantity="200" tradePrice="263" notes="A"/>'
       f'<Trade accountId="U11111111" transactionID="7" ibOrderID="99" conid="2" assetCategory="BAG" '
       f'levelOfDetail="ORDER" notes="{ACCT}"/>'
       f'</Trades><OptionEAE><OptionEAE accountId="{ACCT}" transactionID="6" conid="9263" '
       f'transactionType="Assignment" strike="263"/></OptionEAE></FlexStatement></FlexStatements>'
       f'</FlexQueryResponse>')
(rec / "raw" / "2026" / "a.xml").write_text(xml)
(rec / "raw" / "2026" / "b.xml").write_text(xml)                 # overlapping rolling window
out = rec / "out.csv"
try:
    rows, shape = ex.export(rec, out)
except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
    rows, shape = [], {"crashed": str(e)}
    out.write_text("")
text = out.read_text()
got = list(csv.DictReader(open(out, encoding="utf-8")))

print("ROWS")
check("VRP legs, the combo row, an assignment delivery and the EAE row are exported",
      {r["transactionID"] for r in got} == {"1", "2", "3", "5", "6", "7"}, str([r["transactionID"] for r in got]))
check("an UNTAGGED combo (BAG) row is exported too (a manual combo is still a combo)",
      any(r["transactionID"] == "7" for r in got), "")
check("magic-formula's stock trade is NOT exported", all(r["transactionID"] != "4" for r in got), "")
check("overlapping statements do not duplicate rows", len(got) == 6, str(len(got)))

print("\nFIELDS AND MASKING")
check("the account id appears NOWHERE in the file", ACCT not in text, "")
check("...it is masked in its own field, even for a row whose account differs from the statement's",
      all(r["accountId"] == ex.MASK for r in got) and "U11111111" not in text, "")
check("...and inside any other kept value", all(ACCT not in (r.get("notes") or "") for r in got)
      and any(ex.MASK in (r.get("notes") or "") for r in got), "")
check("only the agreed fields are exported (no description, no P&L fields)",
      bool(got) and list(got[0].keys()) == ex.KEEP and "fifoPnlRealized" not in text and "memo" not in text,
      str(list(got[0]) if got else "no rows"))
leg = next((r for r in got if r["transactionID"] == "1"), {})
check("a leg keeps exec id, order id, tag, conid, side, qty, price, commission, time, option fields",
      tuple(leg.get(k) for k in ("ibExecID", "ibOrderID", "orderReference", "conid", "buySell",
                                  "quantity", "tradePrice", "ibCommission", "dateTime", "putCall",
                                  "strike", "expiry")) == ("e1", "77", "options-vrp:R", "9263", "SELL", "-2", "1.2",
                                         "-1.3", "20261007;153001", "P", "263", "20261120"), str(leg))

print("\nHOW THE COMBO APPEARS")
check("order 77 shows two OPT executions and one BAG order-level row",
      shape.get("77") == {"Trade/OPT/EXECUTION": 2, "Trade/BAG/ORDER": 1}, str(shape))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} export checks behaved as expected")
