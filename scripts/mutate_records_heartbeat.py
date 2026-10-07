"""Mutation-test `scripts/test_records_heartbeat.py`: the download job's success signals.

Seeds real faults into a TEMP COPY of download_ib_records.py (via IB_RECORDS_MODULE; the real
script is never edited) and demands the suite catches every one. Engine: _mutate_records_core.py.

Run: python scripts/mutate_records_heartbeat.py
"""
from __future__ import annotations

import sys

from _mutate_records_core import run

MUTATIONS = [
    ('    ping_heartbeat()\n    return 0',
     '    return 0',
     'no ping on success'),
    ('        alert(f"[IB records] OK {datetime.now():%Y-%m-%d}: "',
     '        (lambda *a: None)(f"[IB records] OK {datetime.now():%Y-%m-%d}: "',
     'no success email'),
    ('    if warns:\n        alert("[IB records] audit trail incomplete"',
     '    if False:\n        alert("[IB records] audit trail incomplete"',
     'warnings reported as OK'),
    ('url = os.getenv("HEARTBEAT_URL", "").strip()',
     'url = os.getenv("HEARTBEAT_URL", "https://example.invalid/default").strip()',
     'pings a default URL when none is configured'),
    ('        with urlopen(Request(url, headers=UA), timeout=10) as r:',
     '        with urlopen(Request(url + "?account=" + EXPECT_ACCOUNT, headers=UA), timeout=10) as r:',
     'the ping leaks the account id'),
    ('        with urlopen(Request(url, headers=UA), timeout=10) as r:',
     '        with urlopen(Request(url, data=b"x", headers=UA), timeout=10) as r:',
     'the ping sends a body'),
    ('    except Exception as e:  # noqa: BLE001\n        log(f"heartbeat ping failed (ignored)',
     '    except ValueError as e:  # noqa: BLE001\n        log(f"heartbeat ping failed (ignored)',
     'a network error in the ping fails the run'),
    ('    if args.rebuild:\n        return 0\n',
     '',
     '--rebuild sends email and pings'),
    ('            alert(f"[IB records] download failed — {e.code}", f"{e}{hint}")\n            return 1',
     '            alert(f"[IB records] download failed — {e.code}", f"{e}{hint}")\n            ping_heartbeat()\n            return 1',
     "a failed download still pings (dead-man's switch fooled)"),
]

if __name__ == "__main__":
    sys.exit(run("test_records_heartbeat.py", MUTATIONS))
