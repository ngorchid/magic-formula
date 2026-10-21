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
    (F, '        if p.is_file() and p.name not in EXCLUDE_NAMES and not is_env_file(p.name):',
     '        if p.is_file() and not is_env_file(p.name):', 'the cache exclusion ignored when walking a folder'),
    # --- .env copies never in plain text (restore drill finding, 2026-10-08) ---
    (F, '        if p.is_file() and p.name not in EXCLUDE_NAMES and not is_env_file(p.name):',
     '        if p.is_file() and p.name not in EXCLUDE_NAMES:', '.env copies in a folder are backed up in plain text'),
    (F, '        if not is_env_file(src.name):\n            yield src, Path(src.name)',
     '        if True:\n            yield src, Path(src.name)', 'a .env FILE source is backed up in plain text'),
    (F, '    return n.startswith(".env") or n.endswith(".env")', '    return n == ".env"',
     'only an exact .env is caught, not .env.bak-* copies'),
    (F, '    return n.startswith(".env") or n.endswith(".env")', '    return n.startswith(".env")',
     '*.env files are backed up in plain text'),
    (F, '    n = name.lower()', '    n = name', 'an upper-case .ENV slips through'),
    (F, '    return n.startswith(".env") or n.endswith(".env")', '    return "env" in n',
     'any file with "env" in its name is dropped from the backup'),
    (F, '                    if ds.st_size == st.st_size and int(ds.st_mtime) == int(st.st_mtime):\n                        continue',
     '                    if ds.st_size == st.st_size:\n                        continue', 'a changed file of the same size is not updated'),
    (F, '                    if ds.st_size == st.st_size and int(ds.st_mtime) == int(st.st_mtime):\n                        continue',
     '                    pass', 'unchanged files are copied again every night'),
    (F, '    ("book_equity.json", PROJECTS / "book_equity.json"),\n', '',
     "the shared book_equity.json is not backed up"),
    (F, '            problems.append(f"source missing: {name} ({src})")\n            continue\n        base = DEST / "current" / name\n        for p, rel in _files(src):\n            dst = base if',
     '            pass\n            continue\n        base = DEST / "current" / name\n        for p, rel in _files(src):\n            dst = base if',
     'a missing source is silent'),
    # --- zipped copy of the auction recordings (ARCHIVE_ONLY, 2026-10-21) ---
    (F, '    ("auction-recorder/results", PROJECTS / "auction-recorder" / "results"),\n', '',
     'the auction recordings are not backed up'),
    (F, '    a = step("zipped copy", lambda: copy_archive(problems), 0)\n', '    a = 0\n',
     'the nightly run never zips the recordings'),
    (F, '        for name, src in SOURCES:\n            if not src.exists():\n                continue',
     '        for name, src in SOURCES + ARCHIVE_ONLY:\n            if not src.exists():\n                continue',
     'the recordings go into every daily snapshot (~15 GB a year)'),
    (F, '                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:',
     '                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:', 'zipped but not compressed'),
    (F, '                    z.write(p, p.name)', '                    z.write(p, str(p))',
     'the zip entry carries the full PC path instead of the file name'),
    (F, '            dst = base / rel.parent / (rel.name + ".zip")', '            dst = base / (rel.name + ".zip")',
     'sub-folders flattened (same-named files collide)'),
    (F, '        if int(dst.stat().st_mtime) != int(st.st_mtime):\n            return False',
     '        pass', 'a changed file of the same size is not zipped again'),
    (F, '        return len(infos) == 1 and infos[0].file_size == st.st_size',
     '        return len(infos) == 1', 'a changed file with the same mtime is not zipped again'),
    (F, '                os.utime(dst, (st.st_atime, st.st_mtime))\n', '',
     'every recording is zipped again every night'),
    (F, '                if bad:\n                    problems.append(f"zipped copy of {p} failed its CRC test")',
     '                if False:\n                    problems.append(f"zipped copy of {p} failed its CRC test")',
     'a zip failing its CRC test is kept'),
    (F, '                os.replace(tmp, dst)\n', '                dst = tmp\n',
     'the .partial file is left as the zip'),
    (F, '    for name, src in ARCHIVE_ONLY:\n        if not src.exists():\n            problems.append(f"source missing: {name} ({src})")',
     '    for name, src in ARCHIVE_ONLY:\n        if not src.exists():\n            pass',
     'a missing recordings folder is silent'),
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
