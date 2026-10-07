"""Mutation-test `scripts/test_capital_bases.py`: the documented capital bases and the NAV reconciliation line.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_capital_bases.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ('scripts/book_summary.py',
     'float(os.getenv("TREND_BASE", CAPITAL_BASES["trend-overlay"]["amount"]))',
     'float(os.getenv("TREND_BASE", "80000"))',
     'trend base hard-coded again (config ignored)'),
    ('scripts/book_summary.py',
     'float(os.getenv("OPTIONS_BASE", CAPITAL_BASES["options-vrp"]["amount"]))',
     'float(os.getenv("OPTIONS_BASE", "40000"))',
     'options base hard-coded again (config ignored)'),
    ('scripts/book_summary.py',
     '    unattributed = net_liq - inception - sum(s["l_total"] for s in sleeves)',
     '    unattributed = net_liq - inception + sum(s["l_total"] for s in sleeves)',
     'unattributed term computed with the wrong sign'),
    ('scripts/book_summary.py',
     'UNATTRIBUTED_WARN_FRAC = 0.01',
     'UNATTRIBUTED_WARN_FRAC = 0.05',
     'warning threshold loosened to 5%'),
    ('scripts/book_summary.py',
     '    big = abs(unattributed) > UNATTRIBUTED_WARN_FRAC * abs(net_liq)',
     '    big = unattributed > UNATTRIBUTED_WARN_FRAC * abs(net_liq)',
     'a negative gap is never flagged'),
    ('scripts/book_summary.py',
     '    big = abs(unattributed) > UNATTRIBUTED_WARN_FRAC * abs(net_liq)',
     '    big = False',
     'the gap is never flagged'),
    ('scripts/book_summary.py',
     'f"Return bases (overlapping — shared collateral, not allocations; sum "',
     'f"Return bases (sum "',
     'bases no longer labelled as overlapping / not allocations'),
    ('scripts/book_summary.py',
     '    recon_note += recon_html\n',
     '',
     'the reconciliation lines never reach the email'),
    ('scripts/book_summary.py',
     '    if _over:\n',
     '    if False:\n',
     'an env override of a base is silent'),
    ('scripts/book_summary.py',
     '    if net_liq is None:\n        return f"<p style=\'color:#64748b;font-size:11px;margin:4px 0\'>{line1}</p>", None\n',
     '    if net_liq is None:\n        net_liq = 0.0\n',
     'a NAV claim is made without an account read'),
    ('config/capital_bases.json',
     '"amount": 75000',
     '"amount": 70000',
     'config trend base changed without the doc'),
    ('docs/capital_bases.md',
     '## Change log',
     '## History',
     'the change log section removed'),
]

if __name__ == "__main__":
    sys.exit(run("test_capital_bases.py", MUTATIONS))
