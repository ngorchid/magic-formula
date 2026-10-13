"""Mutation-test `scripts/test_mark_refresh.py` (daily price splice, 2026-10-13) and
`scripts/test_no_buy.py` (owner buy exclusions).

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suites catch every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_mark_refresh.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

R = "scripts/run_paper.py"
O = "paper/orchestrator.py"
SPLICE = [
    (R, "        adj.loc[s.index, t] = s.values\n",
     "        adj.loc[day, t] = float(s.iloc[-1])\n        if len(s) >= 2:\n"
     "            prior = adj.index[(adj.index < day) & adj[t].notna()]\n"
     "            if len(prior):\n                adj.loc[prior[-1], t] = float(s.iloc[-2])\n",
     "the old splice (newest close on today, prior close on the cache date)"),
    (R, "        adj.loc[s.index, t] = s.values\n",
     "        adj.loc[s.index[-1:], t] = s.values[-1:]\n",
     "only the newest bar written (prior stays on the cached basis)"),
    (R, "    px = px[~px.index.duplicated(keep=\"last\")]\n", "",
     "duplicate dates kept (reindex on a duplicated index)"),
    (R, "        s = px[t].dropna()\n", "        s = px[t]\n",
     "missing bars written as NaN over cached values"),
]
NO_BUY = [
    (O, "            if _nb and today < _nb[0]:\n", "            if False:\n", "exclusion ignored"),
    (O, "            if _nb and today < _nb[0]:\n", "            if _nb:\n", "exclusion never expires"),
    (O, "            if _nb and today < _nb[0]:\n", "            if _nb and today <= _nb[0]:\n",
     "still excluded ON the 'until' day (off by one)"),
    (R, "    return PaperConfig(budget=budget, no_buy=NO_BUY)", "    return PaperConfig(budget=budget)",
     "the live config drops the exclusions"),
    (R, '    "CTVA": ("2027-03-01",', '    "CTVA": ("2026-11-01",', "CTVA's exclusion ends in November again"),
]

if __name__ == "__main__":
    rc = run("test_mark_refresh.py", SPLICE)
    sys.exit(rc or run("test_no_buy.py", NO_BUY))
