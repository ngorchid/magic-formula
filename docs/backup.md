# Backup to OneDrive — what, when, and how to restore

Set up 2026-10-07 (owner's decisions: OneDrive personal, active sleeves only, `.env` files in an
encrypted archive). Script: `scripts/backup_to_onedrive.py`; task `TradingBackupOneDrive`,
**daily 23:30** (after all runs; catches up if the PC was off); log `logs\backup.log`.

## What is backed up

| Source | Why |
|---|---|
| `magic-formula-live\results` (minus `panels.pkl`, a self-rebuilding price cache) and `\logs` | state, run log, book NetLiq history, health / records / backup logs |
| `trend-overlay-live\results`, `options-vrp-live\results` | each sleeve's state.json (options: entry credits — not reconstructible from IB) and run log |
| `substitute-pairs\results` + `book_equity.json` | the paper sleeve's book |
| `PycharmProjects\book_equity.json` | the shared book-equity file behind the book-level drawdown check |
| `C:\Users\Nicolas\IB-records` | the as-reported IB audit trail (raw XML + manifest) |
| the four live `.env` files | credentials — **encrypted** (below) |

Not backed up: the code (it is in Bitbucket), caches, the retired paper checkouts.

## Where — `%OneDrive%\TradingBackup`

```
current\<source>\...      latest copy. ADDITIVE: nothing is ever deleted here, so a file
                          deleted or wiped on the PC survives; OneDrive version history keeps
                          the earlier versions of each file.
daily\YYYY-MM-DD.zip      snapshot of everything, one per night (reruns: _2, _3), kept 365 days
env\YYYY-MM-DD.7z         the .env files, AES-256 with encrypted file names, kept 365 days
```

Every archive is verified after writing (zip CRC test; 7z test with the password). Any problem —
a missing source, a failed archive, no password set, OneDrive folder missing — is logged and
emailed; each step runs on its own, so one failure never stops the others. Optional
`BACKUP_HEARTBEAT_URL` (live `.env`): pinged only after a fully clean run, for a second
Healthchecks check (schedule `30 23 * * *`, Europe/Zurich).

## The `.env` password

Set once, by the owner, in a PowerShell window:

```
powershell -ExecutionPolicy Bypass -File C:\Users\Nicolas\PycharmProjects\magic-formula-live\scripts\set_backup_password.ps1
```

It is stored DPAPI-encrypted in `%LOCALAPPDATA%\TradingBackup\envzip.cred` — readable only by this
Windows user on this PC, which is what lets the task run unattended. It is never in the repo or in
OneDrive. **Keep it in a password manager**: on any other machine the archive cannot be opened
without it. (While the job runs, the password is passed to 7-Zip on its command line, visible only
to processes of this same Windows user.) Run the script again to change it; older archives keep
the old password.

## Restore

**One file or a sleeve's state, on this PC:** stop that sleeve's scheduled task, copy the file from
`TradingBackup\current\...` (or out of a `daily\` zip for a given night), start the task again. For
an earlier version than the latest, use OneDrive's version history (right-click the file > Version
history) or the matching `daily\` zip.

**Everything, on a new PC:**
1. Install Python, Git, IB Gateway + IBC, 7-Zip; sign in to OneDrive.
2. Clone the repos (Bitbucket `picard_capital/…`, branch `live`), create each `.venv`.
3. `.env` files: open `TradingBackup\env\<latest>.7z` with 7-Zip and the password; each file is
   stored under its repo folder (`magic-formula-live\.env`, …) — extract into `PycharmProjects`.
4. Copy each `TradingBackup\current\<repo>\results` back to `<repo>\results`,
   `current\IB-records` to `C:\Users\Nicolas\IB-records`, and the two `book_equity.json` files.
5. Recreate the scheduled tasks (see each repo's docs) and run each sleeve's tests before the first
   live run.
