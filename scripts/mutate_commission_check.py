"""Mutation-test `scripts/test_commission_check.py`: the realised-commission check vs the cost model.

Seeds faults into a TEMP COPY of download_ib_records.py (IB_RECORDS_MODULE; the real script is never
edited). Engine: _mutate_records_core.py.

Run: python scripts/mutate_commission_check.py
"""
from __future__ import annotations

import sys

from _mutate_records_core import run

MUTATIONS = [
    ('ASSUMED_COMMISSION = {"options-vrp": float(os.getenv("VRP_ASSUMED_COMMISSION", "0.65"))}',
     'ASSUMED_COMMISSION = {"options-vrp": float(os.getenv("VRP_ASSUMED_COMMISSION", "1.00"))}',
     "assumption no longer the guard's 0.65"),
    ('COST_WARN_RATIO = 1.25',
     'COST_WARN_RATIO = 2.0',
     'tolerance loosened to 2x'),
    ('COST_WARN_RATIO = 1.25',
     'COST_WARN_RATIO = 1.1',
     'tolerance tightened to 1.1x (false alarms)'),
    ('COST_MIN_UNITS = 10',
     'COST_MIN_UNITS = 1',
     'tiny samples alert'),
    ('        if assumed and units >= COST_MIN_UNITS and per > COST_WARN_RATIO * assumed:',
     '        if False:',
     'the cost alert never fires'),
    ('                and r.get("assetCategory") != "BAG"]\n    agg',
     '                ]\n    agg',
     'combo rows double-count commission'),
    ('                and _iso8(r.get("tradeDate") or r.get("dateTime", "")) >= LINKED_FROM',
     '                and True',
     'pre-tagging executions counted'),
    ('        a[1] += abs(float(r.get("quantity") or 0))',
     '        a[1] += 1',
     'units counted as executions, not contracts'),
    ('    warns += commission_check()\n',
     '',
     'main never runs the check'),
]

if __name__ == "__main__":
    sys.exit(run("test_commission_check.py", MUTATIONS))
