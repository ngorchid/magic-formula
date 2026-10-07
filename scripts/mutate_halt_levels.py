"""Mutation-test `scripts/test_halt_levels.py`: the kill-switch levels and documented sizing (magic-formula).

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_halt_levels.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ('risk_guard.py',
     '    for name, mode in ((r / "HALT_HARD", HALT_HARD), (r / "HALT_ALL", HALT_ALL), (r / "HALT", HALT_NEW)):',
     '    for name, mode in ((r / "HALT_ALL", HALT_ALL), (r / "HALT", HALT_NEW)):',
     'the HALT_HARD file is not recognised'),
    ('risk_guard.py',
     '    for name, mode in ((r / "HALT_HARD", HALT_HARD), (r / "HALT_ALL", HALT_ALL), (r / "HALT", HALT_NEW)):',
     '    for name, mode in ((r / "HALT_ALL", HALT_ALL), (r / "HALT_HARD", HALT_HARD), (r / "HALT", HALT_NEW)):',
     'HALT_ALL takes precedence over HALT_HARD'),
    ('risk_guard.py',
     '    if env in ("hard", "3"):\n        return HALT_HARD, "TRADING_HALT=hard"\n',
     '',
     'TRADING_HALT=hard not recognised'),
    ('scripts/run_paper.py',
     '    if _halt in (HALT_ALL, HALT_HARD):',
     '    if _halt == HALT_ALL:',
     'HALT_HARD lets magic-formula trade'),
    ('scripts/run_paper.py',
     '    if _halt in (HALT_ALL, HALT_HARD):',
     '    if _halt == HALT_HARD:',
     'HALT_ALL lets magic-formula trade'),
    ('scripts/run_paper.py',
     '    budget, src = documented_sizing(ROOT, "magic-formula")',
     '    budget, src = float(os.getenv("BUDGET", "100000")), "env"',
     'budget read from env only (old 100k default)'),
    ('scripts/run_paper.py',
     '    cfg = paper_config()',
     '    cfg = PaperConfig(budget=float(os.getenv("BUDGET", "100000")))',
     'main bypasses paper_config'),
]

if __name__ == "__main__":
    sys.exit(run("test_halt_levels.py", MUTATIONS))
