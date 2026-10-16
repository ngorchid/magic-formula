"""Mutation-test `scripts/test_email_trades.py`: the "Today's trades" section of the daily email.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_email_trades.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

E = "paper/email_report.py"
MUTATIONS = [
    (E, '            if r.get("exit_date") == today and r.get("ticker") in set(trades.get("sells") or [])]',
     '            if r.get("ticker") in set(trades.get("sells") or []) or r.get("exit_date") != today]',
     "old sells listed as today's"),
    (E, "    {_trades_section(state, trades, today)}\n", "",
     'the section is missing from the email'),
    (E, '        return "<p style=\'font-family:monospace;font-size:13px\'>Trades today: unavailable</p>"',
     '        return "<p style=\'font-family:monospace;font-size:13px\'>Trades today: none</p>"',
     "an unavailable summary reads as 'no trades'"),
    (E, '    if holds:\n        out +=', '    if False:\n        out +=',
     'held-through names not listed'),
    (E, '    if sweeps:\n        out +=', '    if False:\n        out +=',
     'the FX sweep not listed'),
    (E, '    if trades and (trades.get("buys") or trades.get("sells")):',
     '    if False:', 'no trade counts in the subject'),
    (E, '        price = f"{p.entry_price:.2f} {p.currency}" if p else "—"',
     '        price = "—"', "a buy's fill price not shown"),
    (E, '    body = build_email_body(state, marks, fx, spy_day_ret, spy_incep_ret, today, trades=trades)',
     '    body = build_email_body(state, marks, fx, spy_day_ret, spy_incep_ret, today)',
     'send_report drops the summary'),
    ("scripts/run_paper.py", "                       alerts=ALERTS, trades=summary)",
     "                       alerts=ALERTS)", 'the daily run does not pass the summary'),
]

if __name__ == "__main__":
    sys.exit(run("test_email_trades.py", MUTATIONS))
