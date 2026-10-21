"""Tests for scripts/backup_to_onedrive.py — no OneDrive, no email: temp dirs, real 7-Zip.

Every rule is pinned by a case built so that breaking THAT rule fails it; scripts/mutate_backup.py
seeds the faults. BACKUP_MODULE may point at another copy of the script (the mutation runner).

Run: python scripts/test_backup.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp())
PROJ, DEST, IBR = TMP / "projects", TMP / "OneDrive" / "TradingBackup", TMP / "IB-records"
PWFILE = TMP / "envzip.cred"
SEVEN = Path(r"C:\Program Files\7-Zip\7z.exe")
TEST_PW = "test-only-Pw-0123456789"


def w(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# A miniature PycharmProjects + IB-records, then point the module at it BEFORE importing it.
for repo in ("magic-formula-live", "trend-overlay-live", "options-vrp-live", "substitute-pairs"):
    w(PROJ / repo / "results" / "paper" / "state.json", f'{{"repo": "{repo}"}}')
    w(PROJ / repo / ".env", f"SECRET_{repo.split('-')[0].upper()}=s3cr3t-{repo}\n")
w(PROJ / "magic-formula-live" / "results" / "paper" / "panels.pkl", "x" * 1000)     # cache: excluded
w(PROJ / "magic-formula-live" / "logs" / "run.log", "log line\n")
# Copies of a .env lying in a results folder (the 2026-10-08 finding): credentials, must never be
# copied in plain text. A file that merely CONTAINS "env" in its name is ordinary state.
ENV_COPIES = ["magic-formula-live/results/.env.bak-statepaths-20261007",
              "options-vrp-live/results/paper/.ENV", "trend-overlay-live/results/live.env"]
for rel in ENV_COPIES:
    w(PROJ / rel, "FLEX_TOKEN=plain-text-secret\n")
w(PROJ / "options-vrp-live" / "results" / "paper" / "environment.json", '{"ok": 1}')
w(PROJ / "trend-overlay-live" / "results" / "env_notes.envelope", "ok")
w(PROJ / "substitute-pairs" / "book_equity.json", '{"pairs": 1}')
w(PROJ / "book_equity.json", '{"book": 1}')
w(IBR / "raw" / "2026" / "flex.xml", "<FlexQueryResponse/>")
# The auction recorder's output: a write-once, compressible daily CSV (ARCHIVE_ONLY: zipped, current/ only).
AUC = PROJ / "auction-recorder" / "results" / "auction"
AUC_ROW = "2026-10-20T15:59:57-04:00,AMZN,NASDAQ,6275429.0,262.27,-38717.0,0.0,262.40,262.38,262.43\n"
w(AUC / "2026-10-20.csv", "ts,symbol,exchange,auctionVolume,auctionPrice,auctionImbalance\n" + AUC_ROW * 2000)
w(AUC / "2026-10-20_meta.json", '{"group": 2}')
w(PROJ / "auction-recorder" / "results" / ".env.bak", "FLEX_TOKEN=plain-text-secret\n")
os.environ.update(PROJECTS_DIR=str(PROJ), BACKUP_DIR=str(DEST), IB_RECORDS_DIR=str(IBR),
                  BACKUP_PASSWORD_FILE=str(PWFILE), SEVEN_ZIP=str(SEVEN), BACKUP_HEARTBEAT_URL="")

MOD = os.getenv("BACKUP_MODULE", str(ROOT / "scripts" / "backup_to_onedrive.py"))
_spec = importlib.util.spec_from_file_location("backup_under_test", MOD)
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
ALERTS: list[str] = []
PINGS: list[int] = []
b.alert = lambda subject, body: ALERTS.append(subject + " | " + body)
b.ping_heartbeat = lambda: PINGS.append(1)
b.log = lambda msg: None

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def set_password(pw: str) -> None:
    ps = ("$s = ConvertTo-SecureString $env:PW -AsPlainText -Force; "
          "[System.Management.Automation.PSCredential]::new('t', $s) | Export-Clixml -LiteralPath $env:F")
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True,
                   env=dict(os.environ, PW=pw, F=str(PWFILE)))


CUR = DEST / "current"
TODAY = date(2026, 10, 7)

# =============================================================================================
print("CURRENT — additive copy")
probs: list[str] = []
n = b.copy_current(probs)
check("every source lands in current/ under its name",
      (CUR / "magic-formula-live/results/paper/state.json").exists()
      and (CUR / "IB-records/raw/2026/flex.xml").exists()
      and (CUR / "book_equity.json").exists() and (CUR / "book_equity.json").read_text() == '{"book": 1}'
      and (CUR / "substitute-pairs/book_equity.json").exists(), str(list(CUR.rglob("*"))))
check("the price cache (panels.pkl) is NOT copied",
      not (CUR / "magic-formula-live/results/paper/panels.pkl").exists(), "")
check("NO .env copy (.env.bak-*, .ENV, *.env) is copied in plain text",
      not any((CUR / rel).exists() for rel in ENV_COPIES)
      and not any("plain-text-secret" in p.read_text(errors="ignore") for p in CUR.rglob("*") if p.is_file()),
      str([rel for rel in ENV_COPIES if (CUR / rel).exists()]))
check("...but files that only contain 'env' in their name ARE copied",
      (CUR / "options-vrp-live/results/paper/environment.json").exists()
      and (CUR / "trend-overlay-live/results/env_notes.envelope").exists(), "")
check("no problems on a complete source tree", probs == [], str(probs))
check("a second run copies nothing unchanged", b.copy_current([]) == 0, "")
src_state = PROJ / "trend-overlay-live" / "results" / "paper" / "state.json"
w(src_state, '{"repo": "trend", "changed": true}')
os.utime(src_state, (src_state.stat().st_atime, src_state.stat().st_mtime + 5))
b.copy_current([])
check("a changed file is updated", "changed" in (CUR / "trend-overlay-live/results/paper/state.json").read_text(), "")
same = PROJ / "substitute-pairs" / "results" / "paper" / "state.json"
old_text = same.read_text()
w(same, old_text.replace("substitute", "SUBSTITUTE"))                  # same size, new content
os.utime(same, (same.stat().st_atime, same.stat().st_mtime + 5))
b.copy_current([])
check("a changed file of the SAME size is updated too (mtime counts, not only size)",
      "SUBSTITUTE" in (CUR / "substitute-pairs/results/paper/state.json").read_text(), "")
(PROJ / "options-vrp-live" / "results" / "paper" / "state.json").unlink()
b.copy_current([])
check("a file DELETED on the PC is NOT deleted from the backup",
      (CUR / "options-vrp-live/results/paper/state.json").exists(), "")
w(PROJ / "options-vrp-live" / "results" / "paper" / "state.json", '{"repo": "options-vrp-live"}')

print("\nZIPPED COPY — auction recordings (current/ only, one zip per file)")
ZAUC = CUR / "auction-recorder" / "results" / "auction"


def zread(p: Path, member: str) -> bytes | None:
    """One member of a zip, or None if the zip or the member is missing (a check, not a crash)."""
    try:
        with zipfile.ZipFile(p) as z:
            return z.read(member)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


csv_src, csv_zip = AUC / "2026-10-20.csv", ZAUC / "2026-10-20.csv.zip"
check("copy_current alone does NOT copy them in plain form",
      not (CUR / "auction-recorder").exists(), str(list(CUR.glob("auction-recorder/**/*"))))
probs = []
na = b.copy_archive(probs)
check("each file lands as <name>.zip under its source name, no problems",
      na == 2 and csv_zip.exists() and (ZAUC / "2026-10-20_meta.json.zip").exists() and probs == [],
      f"n={na} {probs} {list(ZAUC.glob('*')) if ZAUC.exists() else ''}")
zi = zipfile.ZipFile(csv_zip).namelist() if csv_zip.exists() else []
check("...holding exactly that file, byte-identical, under its own name",
      zi == ["2026-10-20.csv"] and zread(csv_zip, "2026-10-20.csv") == csv_src.read_bytes(), str(zi))
check("...COMPRESSED (the point: ~5x smaller than the CSV)",
      csv_zip.exists() and csv_zip.stat().st_size * 5 < csv_src.stat().st_size,
      f"{csv_zip.stat().st_size if csv_zip.exists() else '-'} vs {csv_src.stat().st_size}")
check("no plain copy and no half-written .partial next to the zips",
      not (ZAUC / "2026-10-20.csv").exists() and not list(ZAUC.glob("*.partial")), str(list(ZAUC.glob("*"))))
check("a .env copy in the recorder's folder is NOT zipped either",
      not list((CUR / "auction-recorder").rglob(".env*")), str(list((CUR / "auction-recorder").rglob("*"))))
check("a second run zips nothing unchanged", b.copy_archive([]) == 0, "")
w(csv_src, csv_src.read_text().replace("AMZN", "AMZX"))                   # same size, new content
os.utime(csv_src, (csv_src.stat().st_atime, csv_src.stat().st_mtime + 5))
b.copy_archive([])
check("a changed file is zipped again (mtime counts)",
      b"AMZX" in (zread(csv_zip, "2026-10-20.csv") or b""), "")
mt = csv_src.stat().st_mtime
w(csv_src, csv_src.read_text() + AUC_ROW)                                 # longer, SAME mtime
os.utime(csv_src, (mt, mt))
b.copy_archive([])
check("a changed file with the SAME mtime is zipped again too (size counts)",
      zread(csv_zip, "2026-10-20.csv") == csv_src.read_bytes(), "")
saved, saved_st = csv_src.read_bytes(), csv_src.stat()
csv_src.unlink()
b.copy_archive([])
check("a file DELETED on the PC keeps its zip in the backup", csv_zip.exists(), "")
csv_src.write_bytes(saved)
os.utime(csv_src, (saved_st.st_atime, saved_st.st_mtime))             # restored as it was
(PROJ / "auction-recorder" / "results").rename(PROJ / "auction-recorder" / "results.away")
probs = []
b.copy_archive(probs)
check("a missing source is reported", any("auction-recorder/results" in p for p in probs), str(probs))
(PROJ / "auction-recorder" / "results.away").rename(PROJ / "auction-recorder" / "results")
_real_testzip = zipfile.ZipFile.testzip
zipfile.ZipFile.testzip = lambda self: "2026-10-21.csv"                   # corrupt
w(AUC / "2026-10-21.csv", "ts\n")
probs = []
b.copy_archive(probs)
zipfile.ZipFile.testzip = _real_testzip
check("a zip failing its CRC test is REPORTED and not kept (retried next night)",
      any("CRC" in p for p in probs) and not (ZAUC / "2026-10-21.csv.zip").exists()
      and not list(ZAUC.glob("*.partial")), str(probs))
check("...and the next run zips it", b.copy_archive([]) == 1 and (ZAUC / "2026-10-21.csv.zip").exists(), "")

print("\nDAILY SNAPSHOT")
z = b.daily_zip(TODAY, probs)
names = zipfile.ZipFile(z).namelist() if z else []
check("named by date", z is not None and z.name == "2026-10-07.zip", str(z))
check("contains every source under its backup name",
      "magic-formula-live/results/paper/state.json" in names and "IB-records/raw/2026/flex.xml" in names
      and "book_equity.json" in names and "substitute-pairs/book_equity.json" in names, str(names))
check("the price cache is NOT in the snapshot", not any("panels.pkl" in x for x in names), str(names))
check("the auction recordings are NOT in the snapshot (current/ only)",
      z is not None and not any("auction" in x for x in names), str([x for x in names if "auction" in x]))
check("NO .env copy is in the snapshot, in plain text or otherwise",
      z is not None and not any(x in names for x in ENV_COPIES)
      and not any(b"plain-text-secret" in zipfile.ZipFile(z).read(x) for x in names), str(names))
check("...but environment.json IS in the snapshot",
      "options-vrp-live/results/paper/environment.json" in names, str(names))
check("a FILE source that is a .env copy is skipped too",
      list(b._files(PROJ / "magic-formula-live" / "results" / ".env.bak-statepaths-20261007")) == []
      and len(list(b._files(PROJ / "book_equity.json"))) == 1, "")
check("the snapshot passes its CRC test", z is not None and zipfile.ZipFile(z).testzip() is None, "")
check("no half-written .partial file is left", not list((DEST / "daily").glob("*.partial")), "")
z2 = b.daily_zip(TODAY, [])
z3 = b.daily_zip(TODAY, [])
check("reruns the same day (even within one second) never overwrite: _2, _3",
      z2 is not None and z3 is not None and z.exists() and z2.exists()
      and [z.name, z2.name, z3.name] == ["2026-10-07.zip", "2026-10-07_2.zip", "2026-10-07_3.zip"],
      f"{z} {z2} {z3}")

_real_testzip = zipfile.ZipFile.testzip
zipfile.ZipFile.testzip = lambda self: "magic-formula-live/results/paper/state.json"   # corrupt
probs = []
zc = b.daily_zip(TODAY + timedelta(days=3), probs)
zipfile.ZipFile.testzip = _real_testzip
check("a snapshot failing its CRC test is REPORTED and not kept",
      zc is None and any("CRC" in p for p in probs)
      and not list((DEST / "daily").glob(f"{TODAY + timedelta(days=3):%Y-%m-%d}*")), str(probs))

print("\nENCRYPTED .env ARCHIVE")
probs = []
check("without a password the env archive is SKIPPED and reported",
      b.env_archive(TODAY, probs) is None and any("no backup password" in p for p in probs), str(probs))
set_password(TEST_PW)
probs = []
e = b.env_archive(TODAY, probs)
check("with the password: an archive is written, no problems", e is not None and probs == [], str(probs))
listing = subprocess.run([str(SEVEN), "l", "-pWRONG", str(e)], capture_output=True, text=True) if e else None
check("file NAMES are encrypted too: a wrong password cannot even list the archive",
      listing is not None and listing.returncode != 0 and ".env" not in listing.stdout,
      listing.stdout[-300:] if listing else "")
out = TMP / "restore"
r = subprocess.run([str(SEVEN), "x", f"-p{TEST_PW}", "-y", f"-o{out}", str(e)],
                   capture_output=True, text=True) if e else None
check("RESTORE with the password gives back each repo's .env, byte-identical",
      r is not None and r.returncode == 0 and all(
          (out / repo / ".env").read_bytes() == (PROJ / repo / ".env").read_bytes()
          for repo in ("magic-formula-live", "trend-overlay-live", "options-vrp-live", "substitute-pairs")),
      str(list(out.rglob("*"))) if out.exists() else str(r))
(PROJ / "substitute-pairs" / ".env").rename(PROJ / "substitute-pairs" / ".env.moved")
probs = []
b.env_archive(TODAY + timedelta(days=1), probs)
check("a missing .env is reported", any("env file missing" in p for p in probs), str(probs))
(PROJ / "substitute-pairs" / ".env.moved").rename(PROJ / "substitute-pairs" / ".env")

print("\nRETENTION")
for d in (TODAY - timedelta(days=b.KEEP_DAYS + 1), TODAY - timedelta(days=b.KEEP_DAYS - 1)):
    w(DEST / "daily" / f"{d:%Y-%m-%d}.zip", "old")
    w(DEST / "env" / f"{d:%Y-%m-%d}.7z", "old")
w(DEST / "daily" / "notes.zip", "not dated")
gone = b.prune(TODAY)
old, keep = TODAY - timedelta(days=b.KEEP_DAYS + 1), TODAY - timedelta(days=b.KEEP_DAYS - 1)
check("archives older than KEEP_DAYS are deleted (daily and env)",
      gone == 2 and not (DEST / "daily" / f"{old:%Y-%m-%d}.zip").exists()
      and not (DEST / "env" / f"{old:%Y-%m-%d}.7z").exists(), str(gone))
check("...archives inside the window are kept", (DEST / "daily" / f"{keep:%Y-%m-%d}.zip").exists()
      and (DEST / "env" / f"{keep:%Y-%m-%d}.7z").exists(), "")
check("...and an undated file is never touched", (DEST / "daily" / "notes.zip").exists(), "")
check("the window is a year", b.KEEP_DAYS == 365, str(b.KEEP_DAYS))

print("\nMAIN — alerts and heartbeat")
ALERTS.clear(); PINGS.clear()
w(AUC / "2026-10-22.csv", "ts\n")
rc = b.main()
check("a complete run returns 0, sends NO alert and pings the heartbeat",
      rc == 0 and ALERTS == [] and PINGS == [1], f"rc={rc} alerts={ALERTS} pings={PINGS}")
check("...and the nightly run zips a new recording", (ZAUC / "2026-10-22.csv.zip").exists(), "")
(PROJ / "trend-overlay-live" / "results").rename(PROJ / "trend-overlay-live" / "results.away")
ALERTS.clear(); PINGS.clear()
rc = b.main()
check("a missing source: returns 1, ALERTS naming it, NO heartbeat",
      rc == 1 and any("trend-overlay-live/results" in a for a in ALERTS) and PINGS == [],
      f"rc={rc} alerts={ALERTS} pings={PINGS}")
(PROJ / "trend-overlay-live" / "results.away").rename(PROJ / "trend-overlay-live" / "results")
PWFILE.unlink()
ALERTS.clear(); PINGS.clear()
rc = b.main()
check("no password: still backs up the rest, but returns 1 and ALERTS",
      rc == 1 and any("no backup password" in a for a in ALERTS) and PINGS == [], f"rc={rc} {ALERTS}")
set_password(TEST_PW)
real_zip = b.daily_zip


def _boom(*a, **k):
    raise OSError("disk full (simulated)")


b.daily_zip = _boom
before = set((DEST / "env").glob("*.7z"))
ALERTS.clear(); PINGS.clear()
rc = b.main()
b.daily_zip = real_zip
check("a CRASHING step is reported and does NOT stop the steps after it (env archive still written)",
      rc == 1 and any("daily snapshot crashed" in a for a in ALERTS)
      and len(set((DEST / "env").glob("*.7z")) - before) == 1 and PINGS == [],
      f"rc={rc} {ALERTS} new env archives: {set((DEST / 'env').glob('*.7z')) - before}")
real_dest = b.DEST
b.DEST = TMP / "NoOneDrive" / "TradingBackup"
ALERTS.clear()
rc = b.main()
check("OneDrive folder missing: returns 1 and ALERTS", rc == 1 and any("OneDrive" in a for a in ALERTS),
      f"rc={rc} {ALERTS}")
b.DEST = real_dest

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} backup checks behaved as expected")
