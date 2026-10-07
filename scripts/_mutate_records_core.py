"""Shared engine for the IB-records mutation runners (temp-copy style of mutate_ib_records.py).

Each mutant is written to a temp copy of download_ib_records.py and the suite is pointed at it via
IB_RECORDS_MODULE, so an interrupted run can never leave a broken script for the 08:30 download
task. A mutant that does not compile is rejected (it would be "caught" by the SyntaxError alone).
Non-zero exit if any fault survives, a pattern does not match exactly once, or a mutant is invalid.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "download_ib_records.py"


def run(suite_name: str, mutations: list[tuple[str, str, str]]) -> int:
    # -B / PYTHONDONTWRITEBYTECODE: a same-size mutant written within the same second as the last
    # one could otherwise load a stale .pyc (cache validated only by size + mtime).
    suite = [sys.executable, "-B", str(ROOT / "scripts" / suite_name)]
    original = TARGET.read_text(encoding="utf-8")
    base = subprocess.run(suite, capture_output=True, text=True, env=dict(os.environ))
    if base.returncode != 0:
        print(f"{suite_name} does not pass on the ORIGINAL file — fix that first")
        print(base.stdout[-2000:])
        return 1
    print("=" * 100)
    print(f"  {suite_name}: {len(mutations)} seeded faults; every one must be CAUGHT\n")
    results, crashes = [], []
    with tempfile.TemporaryDirectory() as tmp:
        mutant = Path(tmp) / "download_ib_records.py"
        env = dict(os.environ, IB_RECORDS_MODULE=str(mutant), PYTHONIOENCODING="utf-8",
                   PYTHONDONTWRITEBYTECODE="1")
        for find, repl, why in mutations:
            n = original.count(find)
            if n != 1:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} PATTERN {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}")
                continue
            src = original.replace(find, repl, 1)
            try:
                compile(src, "download_ib_records.py", "exec")
            except SyntaxError as e:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} INVALID MUTANT ({e.msg})")
                continue
            mutant.write_text(src, encoding="utf-8")
            r = subprocess.run(suite, capture_output=True, text=True, env=env)
            caught = r.returncode != 0
            # A crash also exits non-zero, but it may be the TEST's own code failing (e.g. a bad
            # attribute in a failure message) rather than a check detecting the fault. Label it so
            # each crash-catch is reviewed instead of counted silently.
            crashed = caught and "FAILURE(S)" not in r.stdout
            results.append((why, caught))
            if crashed:
                crashes.append(why)
            tail = (r.stderr.strip().splitlines() or ["?"])[-1][:60] if crashed else ""
            print(f"  [{'ok  ' if caught else 'FAIL'}] {why:80} "
                  f"{('CAUGHT (crash: ' + tail + ')') if crashed else 'CAUGHT' if caught else '*** SURVIVED ***'}")
    survived = [w for w, c in results if c is False]
    missing = [w for w, c in results if c is None]
    print("\n" + "=" * 100)
    if missing:
        print(f"{len(missing)} mutation(s) not applied (pattern moved or mutant invalid) — update:")
        for w in missing:
            print("   " + w)
    if survived:
        print(f"{len(survived)} MUTATION(S) SURVIVED — those cases cannot fail and are decoration:")
        for w in survived:
            print("   " + w)
    if crashes:
        print(f"{len(crashes)} fault(s) caught by a CRASH, not a check — review each:")
        for w in crashes:
            print("   " + w)
    if survived or missing:
        return 1
    print(f"all {len(mutations)} seeded faults were caught; the real script was never modified")
    return 0
