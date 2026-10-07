"""Mutation-test `scripts/test_alert_email.py`: alert emails on every path without a daily report (magic-formula).

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_alert_email.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ('risk_guard.py',
     '    if not recs:\n        return False\n    pick',
     '    if True:\n        return False\n    pick',
     'email_if_alerts never sends'),
    ('scripts/run_paper.py',
     '        email_if_alerts(ALERTS, f"Magic Formula {_halt.upper()}", today)\n',
     '',
     'a halted run sends no email'),
    ('scripts/run_paper.py',
     '            email_if_alerts(ALERTS, "Magic Formula", today)\n            return',
     '            return',
     'a failed connection sends no email'),
]

if __name__ == "__main__":
    sys.exit(run("test_alert_email.py", MUTATIONS))
