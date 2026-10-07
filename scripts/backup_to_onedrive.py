"""Nightly backup of the live trading state to OneDrive (owner's decision, 2026-10-07).

WHY. The strategies' state.json files, run logs and the IB records archive exist only on this PC.
Losing them means rebuilding each book by hand (entry credits, inception values, P&L history) and
losing the as-reported audit trail. IB can regenerate its statements; nothing can regenerate these.

WHAT (active sleeves only, owner's choice):
  magic-formula-live/results (minus panels.pkl, a self-rebuilding 37 MB price cache) and /logs,
  trend-overlay-live/results, options-vrp-live/results, substitute-pairs/results +
  book_equity.json, the shared PycharmProjects/book_equity.json, and C:/Users/Nicolas/IB-records.

WHERE: %OneDrive%/TradingBackup (BACKUP_DIR overrides)
  current/<source>/...      latest copy. ADDITIVE: a file deleted or emptied on the PC is never
                            deleted from here (OneDrive's version history keeps older versions).
  daily/YYYY-MM-DD.zip      dated snapshot of everything above, kept KEEP_DAYS days.
  env/YYYY-MM-DD.7z         the four .env files (credentials), AES-256 with encrypted file names
                            (7-Zip). The password is stored DPAPI-encrypted for this Windows user
                            only (set it with scripts/set_backup_password.ps1); without it this
                            part is skipped and alerted, the rest still runs.

Every archive is verified after writing (zip CRC test; 7z test with the password). Failures are
logged and emailed (EMAIL_* in the live .env); BACKUP_HEARTBEAT_URL (optional) is pinged only after
a fully successful run, for a dead-man's switch.

Run: python scripts/backup_to_onedrive.py
"""
from __future__ import annotations

import os
import shutil
import smtplib
import subprocess
import sys
import zipfile
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

PROJECTS = Path(os.getenv("PROJECTS_DIR", r"C:\Users\Nicolas\PycharmProjects"))
DEST = Path(os.getenv("BACKUP_DIR", str(Path(os.getenv("OneDrive", r"C:\Users\Nicolas\OneDrive"))
                                         / "TradingBackup")))
SEVEN_ZIP = Path(os.getenv("SEVEN_ZIP", r"C:\Program Files\7-Zip\7z.exe"))
PASSWORD_FILE = Path(os.getenv("BACKUP_PASSWORD_FILE", str(Path(os.getenv("LOCALAPPDATA", ""))
                                                          / "TradingBackup" / "envzip.cred")))
KEEP_DAYS = 365
EXCLUDE_NAMES = {"panels.pkl"}                 # self-rebuilding cache, not state

# (name in the backup, path). A directory is copied recursively; a file is copied as is.
SOURCES = [
    ("magic-formula-live/results", PROJECTS / "magic-formula-live" / "results"),
    ("magic-formula-live/logs", PROJECTS / "magic-formula-live" / "logs"),
    ("trend-overlay-live/results", PROJECTS / "trend-overlay-live" / "results"),
    ("options-vrp-live/results", PROJECTS / "options-vrp-live" / "results"),
    ("substitute-pairs/results", PROJECTS / "substitute-pairs" / "results"),
    ("substitute-pairs/book_equity.json", PROJECTS / "substitute-pairs" / "book_equity.json"),
    ("book_equity.json", PROJECTS / "book_equity.json"),
    ("IB-records", Path(os.getenv("IB_RECORDS_DIR", r"C:\Users\Nicolas\IB-records"))),
]
ENV_FILES = [PROJECTS / r / ".env" for r in
             ("magic-formula-live", "trend-overlay-live", "options-vrp-live", "substitute-pairs")]


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def _files(src: Path):
    """(path, relative path) of every file to back up under `src` (a file or a directory)."""
    if src.is_file():
        yield src, Path(src.name)
        return
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.name not in EXCLUDE_NAMES:
            yield p, p.relative_to(src)


def copy_current(problems: list[str]) -> int:
    """Additive copy into current/: new or changed files are copied, nothing is ever deleted."""
    n = 0
    for name, src in SOURCES:
        if not src.exists():
            problems.append(f"source missing: {name} ({src})")
            continue
        base = DEST / "current" / name
        for p, rel in _files(src):
            dst = base if src.is_file() else base / rel
            try:
                st = p.stat()
                if dst.exists():
                    ds = dst.stat()
                    if ds.st_size == st.st_size and int(ds.st_mtime) == int(st.st_mtime):
                        continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                n += 1
            except OSError as e:          # a file locked mid-write: report, keep going
                problems.append(f"could not copy {p}: {e}")
    return n


def _new_name(folder: Path, today: date, ext: str) -> Path:
    """YYYY-MM-DD<ext>, or YYYY-MM-DD_2<ext>, _3 ... for a rerun the same day: never an existing
    file. (A clock-based suffix collided on two reruns within one second -- caught by the tests.)"""
    folder.mkdir(parents=True, exist_ok=True)
    out, k = folder / f"{today:%Y-%m-%d}{ext}", 1
    while out.exists() or out.with_name(out.name + ".partial").exists():
        k += 1
        out = folder / f"{today:%Y-%m-%d}_{k}{ext}"
    return out


def daily_zip(today: date, problems: list[str]) -> Path | None:
    """Dated snapshot of every source; CRC-verified after writing. Never overwrites an earlier one."""
    out = _new_name(DEST / "daily", today, ".zip")
    tmp = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, src in SOURCES:
            if not src.exists():
                continue
            for p, rel in _files(src):
                arc = name if src.is_file() else f"{name}/{rel.as_posix()}"
                try:
                    z.write(p, arc)
                except OSError as e:
                    problems.append(f"could not zip {p}: {e}")
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
    if bad:
        problems.append(f"daily snapshot failed its CRC test at {bad}")
        tmp.unlink(missing_ok=True)
        return None
    tmp.rename(out)
    return out


def backup_password() -> str | None:
    """The env-archive password, DPAPI-decrypted for this Windows user, or None if not set."""
    if not PASSWORD_FILE.exists():
        return None
    ps = ("$c = Import-Clixml -LiteralPath $env:BACKUP_PW_FILE; "
          "[Console]::Out.Write($c.GetNetworkCredential().Password)")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, BACKUP_PW_FILE=str(PASSWORD_FILE)))
    return r.stdout if r.returncode == 0 and r.stdout else None


def env_archive(today: date, problems: list[str]) -> Path | None:
    """The .env files as an AES-256 7z with encrypted names; verified with the password."""
    pw = backup_password()
    if not pw:
        problems.append("env archive SKIPPED: no backup password set — run "
                        "scripts/set_backup_password.ps1 once")
        return None
    if not SEVEN_ZIP.exists():
        problems.append(f"env archive SKIPPED: 7-Zip not found at {SEVEN_ZIP}")
        return None
    files = [f for f in ENV_FILES if f.exists()]
    for f in ENV_FILES:
        if not f.exists():
            problems.append(f"env file missing: {f}")
    if not files:
        return None
    out = _new_name(DEST / "env", today, ".7z")
    # Relative paths from PROJECTS, so each file is stored under its repo ("magic-formula-live\.env").
    rel = [str(f.relative_to(PROJECTS)) for f in files]
    r = subprocess.run([str(SEVEN_ZIP), "a", "-t7z", "-mhe=on", f"-p{pw}", "-y", "-bso0", "-bsp0",
                        str(out), *rel], cwd=PROJECTS, capture_output=True, text=True)
    if r.returncode != 0:
        problems.append(f"env archive failed (7z exit {r.returncode}): {r.stderr.strip()[:200]}")
        out.unlink(missing_ok=True)
        return None
    t = subprocess.run([str(SEVEN_ZIP), "t", f"-p{pw}", "-bso0", "-bsp0", str(out)],
                       capture_output=True, text=True)
    if t.returncode != 0:
        problems.append("env archive failed its 7z test with the stored password")
        return None
    return out


def prune(today: date) -> int:
    """Delete daily snapshots and env archives older than KEEP_DAYS (by the date in the name)."""
    cutoff, n = today - timedelta(days=KEEP_DAYS), 0
    for sub, ext in (("daily", ".zip"), ("env", ".7z")):
        for p in (DEST / sub).glob(f"*{ext}"):
            try:
                d = datetime.strptime(p.name[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            if d < cutoff:
                p.unlink()
                n += 1
    return n


def alert(subject: str, body: str) -> None:
    u, p, to = os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"), os.getenv("TO_EMAIL")
    if not (u and p and to):
        log("(no email creds — alert not sent)")
        return
    m = MIMEText(body)
    m["Subject"], m["From"], m["To"] = subject, u, to
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(u, p)
            s.sendmail(u, [to], m.as_string())
        log(f"alert emailed to {to}")
    except Exception as e:  # noqa: BLE001
        log(f"alert email failed: {type(e).__name__}")


def ping_heartbeat() -> None:
    url = os.getenv("BACKUP_HEARTBEAT_URL", "").strip()
    if not url:
        return
    try:
        with urlopen(Request(url, headers={"User-Agent": "trading-backup/1.0"}), timeout=10) as r:
            r.read(64)
    except Exception as e:  # noqa: BLE001
        log(f"heartbeat ping failed (ignored): {type(e).__name__}")


def main() -> int:
    today = date.today()
    problems: list[str] = []
    if not DEST.parent.exists():
        problems.append(f"OneDrive folder not found: {DEST.parent} — is OneDrive set up?")
        log("FAILED: " + problems[0])
        alert("[backup] FAILED — OneDrive folder missing", problems[0])
        return 1
    def step(name, fn, default):
        """Run one step on its own: a crash is reported and never stops the steps after it."""
        try:
            return fn()
        except Exception as ex:  # noqa: BLE001 — report, never die silently
            problems.append(f"{name} crashed: {type(ex).__name__}: {ex}")
            return default

    n = step("copy", lambda: copy_current(problems), 0)
    z = step("daily snapshot", lambda: daily_zip(today, problems), None)
    e = step("env archive", lambda: env_archive(today, problems), None)
    pruned = step("prune", lambda: prune(today), 0)
    snap = f"{z.name} ({z.stat().st_size / 1024:,.0f} KB)" if z else "NONE"
    log(f"current/: {n} file(s) updated | daily: {snap}")
    log(f"env: {e.name if e else 'NONE'} | pruned {pruned} old archive(s) | dest {DEST}")
    for pr in problems:
        log(f"PROBLEM: {pr}")
    if problems:
        alert(f"[backup] {len(problems)} problem(s) — {today}", "\n".join(problems)
              + f"\n\nDestination: {DEST}")
        return 1
    ping_heartbeat()
    return 0


if __name__ == "__main__":
    sys.exit(main())
