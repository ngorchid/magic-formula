"""Mutation-test `scripts/test_live_gateway_check.py`: the live-Gateway health check.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_live_gateway_check.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

F = "scripts/check_live_gateway.py"
MUTATIONS = [
    (F, '        time.sleep(RETRY_DELAY_S)\n        try:\n            accts = managed_accounts()\n            print(',
     '        try:\n            raise RuntimeError("no retry")\n            print(',
     'no retry: the 12:00 self-restart alerts and relaunches again'),
    (F, '        time.sleep(RETRY_DELAY_S)\n', '',
     'retry without waiting (lands in the same restart gap)'),
    (F, 'RETRY_DELAY_S = 180 ', 'RETRY_DELAY_S = 30 ', 'retry delay shorter than the restart gap'),
    (F, '            relaunched = relaunch_gateway()\n', '            relaunched = True\n',
     'really down, but no relaunch'),
    (F, '            return 1\n    live =', '            return 0\n    live =',
     'really down reported as success (exit 0)'),
    (F, '                  f"Must be up before the 16:00 CET run.")\n',
     '                  f"Must be up before the 16:00 CET run.") if False else None\n',
     'really down, but no alert'),
    (F, '    if not live:\n', '    if False:\n', 'a paper-only login passes as healthy'),
    (F, '    except Exception as first:  # noqa: BLE001\n        print(',
     '    except Exception as first:  # noqa: BLE001\n        raise first\n        print(',
     'a failed first attempt crashes the check'),
]

if __name__ == "__main__":
    sys.exit(run("test_live_gateway_check.py", MUTATIONS))
