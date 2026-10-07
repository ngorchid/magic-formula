"""Mutation-test `scripts/test_backup.py`: the nightly OneDrive backup.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_backup.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

F = "scripts/backup_to_onedrive.py"
MUTATIONS = [
    # --- current/: additive copy ---
    (F, 'EXCLUDE_NAMES = {"panels.pkl"}', 'EXCLUDE_NAMES = set()', 'the price cache is backed up'),
    (F, '        if p.is_file() and p.name not in EXCLUDE_NAMES:', '        if p.is_file():',
     'exclusions ignored when walking a folder'),
    (F, '                    if ds.st_size == st.st_size and int(ds.st_mtime) == int(st.st_mtime):\n                        continue',
     '                    if ds.st_size == st.st_size:\n                        continue', 'a changed file of the same size is not updated'),
    (F, '                    if ds.st_size == st.st_size and int(ds.st_mtime) == int(st.st_mtime):\n                        continue',
     '                    pass', 'unchanged files are copied again every night'),
    (F, '    ("book_equity.json", PROJECTS / "book_equity.json"),\n', '',
     "the shared book_equity.json is not backed up"),
    (F, '            problems.append(f"source missing: {name} ({src})")', '            pass',
     'a missing source is silent'),
    # --- daily snapshot ---
    (F, '    while out.exists() or out.with_name(out.name + ".partial").exists():',
     '    while False:', 'a same-day rerun overwrites the first snapshot'),
    (F, '                arc = name if src.is_file() else f"{name}/{rel.as_posix()}"',
     '                arc = rel.as_posix()', 'snapshot loses the source names (files collide)'),
    (F, '    tmp.rename(out)\n    return out', '    return tmp', 'the .partial file is left as the snapshot'),
    (F, '    if bad:\n        problems.append(f"daily snapshot failed its CRC test at {bad}")',
     '    if False:\n        problems.append(f"daily snapshot failed its CRC test at {bad}")',
     'CRC test result ignored'),
    # --- encrypted .env archive ---
    (F, '"-mhe=on", ', '', 'file names left unencrypted'),
    (F, '    rel = [str(f.relative_to(PROJECTS)) for f in files]', '    rel = [f.name for f in files]',
     'archive loses which repo each .env belongs to'),
    (F, '        problems.append("env archive SKIPPED: no backup password set — run "',
     '        return None\n        problems.append("env archive SKIPPED: no backup password set — run "',
     'a missing password is silent'),
    (F, '            problems.append(f"env file missing: {f}")', '            pass',
     'a missing .env is silent'),
    # --- retention ---
    (F, 'KEEP_DAYS = 365', 'KEEP_DAYS = 30', 'retention shortened to a month'),
    (F, '            if d < cutoff:', '            if d <= today:', 'every archive pruned'),
    (F, '            except ValueError:\n                continue', '            except ValueError:\n                p.unlink(); continue',
     'undated files deleted by the pruning'),
    # --- main: alerts, heartbeat, step isolation ---
    (F, '    if problems:\n        alert(', '    if False:\n        alert(', 'problems never alert'),
    (F, '        return 1\n    ping_heartbeat()\n    return 0', '        ping_heartbeat()\n        return 1\n    ping_heartbeat()\n    return 0',
     'heartbeat pinged on a FAILED run (dead-man switch never fires)'),
    (F, '    ping_heartbeat()\n    return 0', '    return 0', 'heartbeat never pinged'),
    (F, '            return fn()\n        except Exception as ex:', '            return fn()\n        except KeyError as ex:',
     'one crashing step stops the rest'),
    (F, '    if not DEST.parent.exists():', '    if False:', 'a missing OneDrive folder is not detected'),
]

if __name__ == "__main__":
    sys.exit(run("test_backup.py", MUTATIONS))
