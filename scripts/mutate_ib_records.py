"""Mutation-test `scripts/test_ib_records.py`: seed real faults, demand the suite catches them.

WHY. The audit trail's rules (attribution, reconciliation, write-once archive) fail SILENTLY when
broken: a fee lands in the wrong sleeve, a missing section stops being noticed, a raw record gets
overwritten. A suite that passes is only evidence if breaking each rule on purpose makes it fail.
The first ad-hoc tests caught 10 of 19 such faults (2026-10-06); every one must be CAUGHT.

Unlike mutate_trend_sizing.py this never edits the real script: each mutant is written to a temp
copy and the suite is pointed at it via IB_RECORDS_MODULE. An interrupted run therefore cannot
leave a broken download_ib_records.py for the IBRecordsDownload task (Tue–Sat 08:30).

Non-zero exit if any fault survives or a pattern no longer matches (the code moved: update this).

Run: python scripts/mutate_ib_records.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "download_ib_records.py"
SUITE = [sys.executable, str(ROOT / "scripts" / "test_ib_records.py")]

MUTATIONS = [
    # --- attribution: the owner's rules ---
    ('return "book" if _interest_ccy(row) == base else "magic-formula"', 'return "magic-formula"',
     "USD interest charged to magic-formula instead of book"),
    ('return "book" if _interest_ccy(row) == base else "magic-formula"', 'return "book"',
     "foreign-currency interest charged to book instead of magic-formula"),
    ('return "options-vrp" if any(w in desc for w in MARKET_DATA_WORDS) else "book"',
     'return "book"', "market-data fees no longer charged to options-vrp"),
    ('("OPRA", "SNAPSHOT",', '("SNAPSHOT",', "OPRA dropped from the market-data keywords"),
    ('"NETWORK A",\n                     "NETWORK B", "NETWORK C",', '"NETWORK A",\n                     "NETWORK B",',
     "Network C dropped from the market-data keywords"),
    ('if category == "capital":\n        return "capital"', 'if False:\n        pass',
     "deposits/withdrawals no longer treated as capital"),
    ('return "magic-formula"\n    if category == "interest"', 'return "book"\n    if category == "interest"',
     "SYEP lending income charged to book"),
    ('    if any(w in desc for w in SYEP_WORDS):\n        return "securities_lending"\n', '',
     "SYEP income no longer categorised as securities_lending"),
    ('return "magic-formula"            # revaluation', 'return "book"            # revaluation',
     "FX translation charged to book"),
    ('    if "FX TRANSLATION" in desc:\n        return "fx_translation"\n', '',
     "FX translation no longer recognised"),
    ('return (owners.get(row.get("conid", ""))\n            or ', 'return (None\n            or ',
     "ownership by tagged trade ignored (asset class only)"),
    ('"STK": "magic-formula"', '"STK": "trend-overlay"', "stocks attributed to trend-overlay"),
    ('    return "unassigned"\n\n\ndef _category', '    return "book"\n\n\ndef _category',
     "unknown rows silently booked to book instead of unassigned"),
    ('owners[c] = s if owners.get(c, s) == s else "conflict"', 'owners[c] = s',
     "a conid traded by two sleeves silently given to the last one"),
    ('if base_category == "futures_mtm" and row.get("assetCategory") != "FUT":\n        return "adjustment"', 'pass',
     "ADJ on a non-future counted as futures variation margin"),
    ('    if len(desc) > 4 and head.isalpha() and desc[3] == " " and "INT" in desc.upper():\n        return head\n', '',
     "interest currency no longer read from the description"),
    ('k = (r.get("date") or r.get("reportDate", ""), r["sleeve"], r["category"])',
     'k = (r.get("date") or r.get("reportDate", ""), "book", r["category"])',
     "attribution summary loses the sleeve"),
    ('bad = [r for r in rows if r.get("sleeve") in ("unassigned", "conflict")]',
     'bad = [r for r in rows if r.get("sleeve") == "unassigned"]',
     "audit stops counting conflict rows"),
    ('bad = [r for r in rows if r.get("sleeve") in ("unassigned", "conflict")]', 'bad = []',
     "audit stops flagging unattributable rows"),

    ('"STAX": "sales_tax",', '', "VAT on fees (STAX) unmapped -> unassigned alert"),
    ('if category in ("fee", "sales_tax") and not row.get("conid"):', 'if category == "fee" and not row.get("conid"):',
     "VAT on fees not attributed like the fee it is charged on"),

    # --- reconciliation ---
    ('max(0.05, 0.001 * abs(want))', 'max(1e9, 0.001 * abs(want))', "reconciliation tolerance made huge"),
    ('max(0.05, 0.001 * abs(want))', 'max(0.0, 0.0 * abs(want))', "reconciliation tolerance zero (false alarms)"),
    ('got["commissions"] += fx(e, "ibCommission")', 'pass', "reconciliation ignores commissions"),
    ('got["transactionTax"] += fx(e, "taxAmount")', 'pass', "reconciliation ignores transaction taxes"),
    ('"fee": "otherFees",', '', "reconciliation ignores fees"),

    # --- archive / tables ---
    ('with open(out, "xb")', 'with open(out, "wb")', "raw files made overwritable"),
    ('if EXPECT_ACCOUNT and accounts != {EXPECT_ACCOUNT}:', 'if False:', "wrong-account statements accepted"),
    ('rows[name][tuple(a.get(k, "") for k in key)] = a',
     'rows[name].setdefault(tuple(a.get(k, "") for k in key), a)',
     "a later IB correction no longer replaces the earlier record"),
    ('a["strategy"], a["run_id"] = _split_ref(a.get("orderReference", ""))',
     'a["strategy"], a["run_id"] = "", ""', "orderReference no longer parsed into strategy/run_id"),

    # --- audit alerts ---
    ('if needs and key not in seen:', 'if needs:', "review alert repeats every day"),
    ('if needs and key not in seen:', 'if False:', "corporate actions no longer flagged for review"),
    ('warn.append(f"{len(untagged)} trade(s) since', 'untagged and None and warn.append(f"{len(untagged)} trade(s) since',
     "untagged trades no longer flagged"),
]


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")
    base = subprocess.run(SUITE, capture_output=True, text=True, env=dict(os.environ))
    if base.returncode != 0:
        print("the suite does not pass on the ORIGINAL file — fix that first")
        print(base.stdout[-2000:])
        return 1
    print("=" * 100)
    print(f"  {len(MUTATIONS)} seeded faults; every one must be CAUGHT\n")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        mutant = Path(tmp) / "download_ib_records.py"
        # PYTHONDONTWRITEBYTECODE: a same-size mutant written within the same second could load a
        # stale .pyc of the previous one (the cache is validated only by size + mtime).
        env = dict(os.environ, IB_RECORDS_MODULE=str(mutant), PYTHONIOENCODING="utf-8",
                   PYTHONDONTWRITEBYTECODE="1")
        for find, repl, why in MUTATIONS:
            if find not in original:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} PATTERN MISSING")
                continue
            mutant.write_text(original.replace(find, repl, 1), encoding="utf-8")
            r = subprocess.run(SUITE, capture_output=True, text=True, env=env)
            caught = r.returncode != 0
            results.append((why, caught))
            print(f"  [{'ok  ' if caught else 'FAIL'}] {why:80} "
                  f"{'CAUGHT' if caught else '*** SURVIVED ***'}")
    survived = [w for w, c in results if c is False]
    missing = [w for w, c in results if c is None]
    print("\n" + "=" * 100)
    if missing:
        print(f"{len(missing)} mutation(s) could not be applied — the code moved, update this file:")
        for w in missing:
            print("   " + w)
    if survived:
        print(f"{len(survived)} MUTATION(S) SURVIVED — those cases cannot fail and are decoration:")
        for w in survived:
            print("   " + w)
    if survived or missing:
        return 1
    print(f"all {len(MUTATIONS)} seeded faults were caught; the real script was never modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
