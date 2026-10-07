"""Mutation-test `scripts/test_export_vrp_fills.py`: the sanitised VRP fill export.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_export_vrp_fills.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ('scripts/export_vrp_fills.py',
     '    if row.get("accountId"):\n        row["accountId"] = MASK\n',
     '',
     'the account id field is not masked'),
    ('scripts/export_vrp_fills.py',
     '            if acct and acct in v:\n                v = v.replace(acct, MASK)',
     '            if False:\n                v = v.replace(acct, MASK)',
     'the account id leaks inside other values'),
    ('scripts/export_vrp_fills.py',
     '            or a.get("orderReference", "").startswith("options-vrp")',
     '            or a.get("orderReference", "").startswith("")',
     "every sleeve's trades exported"),
    ('scripts/export_vrp_fills.py',
     '            or a.get("assetCategory") in ("OPT", "BAG")',
     '            or a.get("assetCategory") in ("OPT",)',
     'combo (BAG) rows dropped'),
    ('scripts/export_vrp_fills.py',
     '            or bool(codes & {"A", "Ex", "Ep"}))',
     '            or False)',
     'assignment deliveries dropped'),
    ('scripts/export_vrp_fills.py',
     '    return (record == "OptionEAE"                       # exercise / assignment / expiry: always\n            or',
     '    return (',
     'exercise/assignment rows dropped'),
    ('scripts/export_vrp_fills.py',
     '            if key in seen:                       # rolling windows overlap: one row per record\n                continue\n',
     '',
     'overlapping windows duplicate rows'),
    ('scripts/export_vrp_fills.py',
     '        "dateTime", "tradeDate", "orderTime", "putCall", "strike", "expiry", "multiplier",',
     '        "dateTime", "tradeDate", "orderTime", "putCall", "expiry", "multiplier",',
     'strike dropped from the export'),
    ('scripts/export_vrp_fills.py',
     '    for k in KEEP[1:]:\n        v = a.get(k, "")',
     '    for k in list(a):\n        v = a.get(k, "")',
     'all IB fields exported, not only the agreed ones'),
]

if __name__ == "__main__":
    sys.exit(run("test_export_vrp_fills.py", MUTATIONS))
