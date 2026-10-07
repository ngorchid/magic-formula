"""Mutation-test `scripts/test_book_overlap.py`: the book summary's overlap, TWR, %-of-base and open-assignment lines.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_book_overlap.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ('scripts/book_summary.py',
     '        if t in held:\n',
     '        if t not in held:\n',
     'overlap computed backwards'),
    ('scripts/book_summary.py',
     '            tag = " ASSIGNED" if sp.get("assigned_contracts") else ""',
     '            tag = ""',
     'an ASSIGNED overlap is not marked'),
    ('scripts/book_summary.py',
     '    root = lambda t: re.sub(r"[^A-Z0-9]", "", t.upper().rsplit(".", 1)[0]  # noqa: E731',
     '    root = lambda t: re.sub(r"[^A-Z0-9]", "", t.upper()  # noqa: E731',
     'yfinance suffix kept (SAP.DE never matches SAP)'),
    ('scripts/book_summary.py',
     '        return "Names held by both sleeves: unknown (a sleeve state could not be read)"',
     '        raise',
     'an unreadable state crashes the summary'),
    ('scripts/book_summary.py',
     '        r = max(rows, key=lambda x: (x.get("toDate", ""), -int(x.get("fromDate", "0") or 0)))',
     '        r = rows[0]',
     "an old statement's TWR shown"),
    ('scripts/book_summary.py',
     '        r = max(rows, key=lambda x: (x.get("toDate", ""), -int(x.get("fromDate", "0") or 0)))',
     '        r = max(rows, key=lambda x: (x.get("toDate", ""), int(x.get("fromDate", "0") or 0)))',
     "the narrower window's TWR shown"),
    ('scripts/book_summary.py',
     '               f"<b>Account TWR:</b> {twr_str}</p>"',
     '               f"</p>"',
     'the TWR headline is missing'),
    ('scripts/book_summary.py',
     '        dd = f"{_money(s[\'mdd\'])}" + (f" ({s[\'mdd\'] / s[\'base\']:.1%} of ${s[\'base\']:,.0f})"',
     '        dd = f"{_money(s[\'mdd\'])}" + (f" ({s[\'mdd\'] / s[\'base\']:.1%})"',
     'a percentage without its disclosed base'),
    ('scripts/book_summary.py',
     '               f"{overlap_line(MAGIC_STATE, OPTIONS_STATE)}</p>")',
     '               f"</p>")',
     'the overlap line is missing from the email'),
    ('scripts/book_summary.py',
     '        sev = "URGENT" if days >= 3 else ("ESCALATION" if days == 2 else "OPEN")',
     '        sev = "OPEN"',
     'open assignments never escalate'),
    ('scripts/book_summary.py',
     '        how = "automatic unwind retrying" if sp.get("assigned_auto") else "MANUAL unwind needed"',
     '        how = "automatic unwind retrying"',
     'a manual assignment is reported as automatic'),
    ('scripts/book_summary.py',
     '        return ["options-vrp ledger unreadable — open assignments UNKNOWN"]',
     '        return []',
     "an unreadable ledger reads as 'nothing open'"),
    ('scripts/book_summary.py',
     '    subject = (("⚠ ASSIGNED " if _asg else "") + f"Live Book',
     '    subject = (f"Live Book',
     'no subject marker for an open assignment'),
    ('scripts/book_summary.py',
     '    pv_line = asg_block + (f"',
     '    pv_line = (f"',
     'the open-assignment block is not in the email'),
    ('scripts/book_summary.py',
     '        if not n:\n            continue\n        since',
     '        since',
     'intact spreads listed as assigned'),
]

if __name__ == "__main__":
    sys.exit(run("test_book_overlap.py", MUTATIONS))
