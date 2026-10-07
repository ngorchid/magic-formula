"""Mutation-test `scripts/test_delivery_attribution.py`: the delivered-shares attribution rule.

Seeds real faults into a TEMP COPY of download_ib_records.py (via IB_RECORDS_MODULE; the real
script is never edited) and demands the suite catches every one. Engine: _mutate_records_core.py.

Run: python scripts/mutate_delivery_attribution.py
"""
from __future__ import annotations

import sys

from _mutate_records_core import run

MUTATIONS = [
    ('    delivered = _vrp_deliveries(rows, owners)\n',
     '    delivered = {}\n',
     'rule switched off (delivered stock -> magic-formula)'),
    ('if not ("Assign" in kind or "Exercise" in kind):',
     'if False:',
     'expirations treated as deliveries'),
    ('if _instrument_sleeve(r, owners) != "options-vrp":',
     'if False:',
     "any sleeve's option event treated as a VRP delivery"),
    ('        if not window:\n            continue\n        # FINGERPRINT',
     '        if False:\n            continue\n        # FINGERPRINT',
     'no delivery required inside the window'),
    ('fp = (strike not in (None, "") and abs(abs(float(t.get("quantity") or 0)) - shares) < 1e-9',
     'fp = (strike not in (None, "") and True',
     'fingerprint ignores the quantity'),
    ('                  and abs(float(t.get("tradePrice") or 0) - float(strike)) < 0.01)',
     '                  and True)',
     'fingerprint ignores the price'),
    ('t["delivery_match"] = "fingerprint" if fp else "window"',
     't["delivery_match"] = "window"',
     'the fingerprint is never recorded'),
    ('last = _add_trading_days(ev, DELIVERY_WINDOW_DAYS)',
     'last = "99999999"',
     'window unbounded (a later unrelated trade is captured)'),
    ('DELIVERY_WINDOW_DAYS = 3 ',
     'DELIVERY_WINDOW_DAYS = 2 ',
     'window shortened to 2 trading days'),
    ('        if d.weekday() < 5:\n            n -= 1\n',
     '        n -= 1\n',
     'window counted in calendar, not trading, days'),
    ('and _row_date(row) >= ev:',
     'and True:',
     'rows from BEFORE the assignment re-attributed too'),
    ('return "options-vrp" if tagged_owner in (None, "", "options-vrp") else "conflict"',
     'return "options-vrp"',
     "magic-formula's tagged trade ignored (no conflict)"),
    ('        r["sleeve"] = attribute(r, r["category"], owners, base, delivered)\n',
     '        r["sleeve"] = attribute(r, r["category"], owners, base)\n',
     'later dividends do not follow the delivered owner'),
]

if __name__ == "__main__":
    sys.exit(run("test_delivery_attribution.py", MUTATIONS))
