"""Live-Gateway health check: connect to the live IB Gateway and confirm it's up and logged in.

Emails an alert ONLY on failure (so a healthy day is silent). Catches the main live risks
before the 16:00 trading run: an un-approved weekly 2FA (Gateway logged out), a hung Gateway,
or any connectivity loss. Reads connection + email settings from the live .env.

RETRY BEFORE ALERTING (2026-10-09). The Gateway restarts ITSELF every day at 12:00
(IBC AutoRestartTime=12:00 PM; no 2FA needed, the session survives) and port 4001 is closed for
~20-60 s while it does. The 12:00 check landed in that gap every weekday: a false "Gateway DOWN"
email, and a relaunch that on 2026-10-09 KILLED the self-restarting Gateway mid-way, forcing a
fresh 2FA login and a 3-hour outage (12:00-15:00). So a failed connect is retried once after
RETRY_DELAY_S; only if that fails too is the Gateway relaunched and the alert sent. A logged-out
Gateway (weekly 2FA) stays down, so both attempts fail and it alerts as before.

DISCONNECTED REMOTE DESKTOP (2026-10-12). While the Windows session is a DISCONNECTED Remote
Desktop session it has no display, and the Gateway (a Java desktop app) cannot open its login
window: IBC loops on "failed to display login dialog" and never logs in, so NO 2FA push is ever
sent. 2026-10-12: 74 such loops 08:00-09:22; the window opened 6 s after an RDP reconnect. So the
alert says so instead of "approve the push". Fix on the box: leave RDP via
ops/disconnect_to_console.bat (hands the session to the console, which keeps a display).
"""
import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from ib_insync import IB, util  # noqa: E402
util.logToConsole('ERROR')

PORT = int(os.getenv("IB_PORT", "4001"))
EXPECT_ACCT_PREFIX = "U"          # live accounts start with U; paper is DU...
CID = 10                          # distinct from the trading run's clientId (5)
RETRY_DELAY_S = 180               # longer than the Gateway's own daily restart gap

# Auto-recovery target. StartGatewayFresh.bat is idempotent — it skips if 4001 is already up and
# otherwise clears a stuck live Gateway and relaunches, which fires a fresh 2FA push. Overridable
# via env in case the IBC path ever moves.
FRESH_BAT = Path(os.getenv("LIVE_GATEWAY_FRESH_BAT",
                           r"C:\Users\Nicolas\IBC\IBCWin-live\StartGatewayFresh.bat"))


def relaunch_gateway() -> bool:
    """Fire StartGatewayFresh.bat DETACHED, so the recovery needs no login to this machine — the
    only manual step is approving the IBKR Mobile 2FA push. Detached (no wait, own process group)
    so a hung launch can never block or hang this check. Returns True if the launch was spawned."""
    if not FRESH_BAT.exists():
        print(f"auto-restart: {FRESH_BAT} not found — cannot relaunch")
        return False
    try:
        flags = (getattr(subprocess, "DETACHED_PROCESS", 0)
                 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        subprocess.Popen(["cmd", "/c", str(FRESH_BAT)], creationflags=flags, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        print("auto-restart: launched StartGatewayFresh.bat (detached)")
        return True
    except Exception as exc:  # noqa: BLE001 — recovery must never crash the health check
        print(f"auto-restart: FAILED to launch — {type(exc).__name__}: {exc}")
        return False


def alert(subject: str, body: str) -> None:
    print(f"ALERT: {subject}\n{body}")
    u, p, to = os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"), os.getenv("TO_EMAIL")
    if not (u and p and to):
        print("(no email creds — cannot send alert)"); return
    m = MIMEText(body); m["Subject"] = subject; m["From"] = u; m["To"] = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(u, p); s.sendmail(u, [to], m.as_string())
    print(f"alert emailed to {to}")


def session_disconnected() -> bool | None:
    """True if this process's Windows session is a DISCONNECTED (Remote Desktop) session, False if
    it has a display (active RDP or console, locked or not), None if it can't be told."""
    try:
        import ctypes
        from ctypes import wintypes
        wts, k32 = ctypes.windll.wtsapi32, ctypes.windll.kernel32
        sid = wintypes.DWORD()
        if not k32.ProcessIdToSessionId(k32.GetCurrentProcessId(), ctypes.byref(sid)):
            return None
        buf, n = ctypes.c_void_p(), wintypes.DWORD()
        if not wts.WTSQuerySessionInformationW(None, sid, 8, ctypes.byref(buf), ctypes.byref(n)):
            return None                                   # 8 = WTSConnectState
        try:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_int)).contents.value == 4  # WTSDisconnected
        finally:
            wts.WTSFreeMemory(buf)
    except Exception:  # noqa: BLE001 — never let this break the check
        return None


def managed_accounts() -> list[str]:
    """Connect, read the logged-in accounts, disconnect. Raises if the Gateway does not answer."""
    ib = IB()
    try:
        ib.connect("127.0.0.1", PORT, clientId=CID, timeout=20)
        return list(ib.managedAccounts())
    finally:
        ib.disconnect()


def main() -> int:
    try:
        accts = managed_accounts()
    except Exception as first:  # noqa: BLE001
        print(f"first attempt failed ({type(first).__name__}: {first}) — retrying in "
              f"{RETRY_DELAY_S} s (the Gateway's own daily restart closes the port briefly)")
        time.sleep(RETRY_DELAY_S)
        try:
            accts = managed_accounts()
            print("second attempt OK — a transient gap, no alert, no restart")
        except Exception as e:  # noqa: BLE001
            # Auto-recover BEFORE emailing, and report what happened in the same alert, so you get
            # one message that already tells you whether to just approve the push or intervene.
            relaunched = relaunch_gateway()
            disc = session_disconnected()
            if relaunched and disc:
                subject = ("[magic-formula LIVE] Gateway DOWN — CONNECT BY REMOTE DESKTOP "
                           "(no 2FA push will come until you do)")
                tail = ("An auto-restart was triggered, BUT this PC's Remote Desktop session is "
                        "DISCONNECTED: the Gateway cannot open its login window without a display, "
                        "so it cannot log in and IB will NOT send a 2FA push.\n\nConnect to the PC "
                        "by Remote Desktop: the login window opens within seconds and the push "
                        "follows — approve it. Then leave with the 'Disconnect to console' desktop "
                        "shortcut (ops/disconnect_to_console.bat), NOT by closing the RDP window.")
            elif relaunched:
                subject = "[magic-formula LIVE] Gateway DOWN — auto-restart triggered, APPROVE the 2FA push"
                tail = ("An auto-restart was triggered — APPROVE the IBKR Mobile 2FA push on your phone "
                        "to bring it back up. No need to log into the machine.\n\n(If it does not come "
                        "up, the weekly login may need attention, or the box is logged off so the task "
                        "could not run.)")
            else:
                subject = "[magic-formula LIVE] Gateway DOWN — AUTO-RESTART FAILED, manual fix"
                tail = "AUTO-RESTART COULD NOT BE LAUNCHED — a manual restart is needed."
            alert(subject,
                  f"Could not connect to the live IB Gateway on 127.0.0.1:{PORT} (twice, "
                  f"{RETRY_DELAY_S} s apart).\n\nError: {type(e).__name__}: {e}\n\n{tail}\n\n"
                  f"Must be up before the 16:00 CET run.")
            return 1
    live = [a for a in accts if a.startswith(EXPECT_ACCT_PREFIX)]
    if not live:
        alert("[magic-formula LIVE] Gateway up but NO live account",
              f"Connected on {PORT} but managedAccounts={accts} (expected a U... live account). "
              f"Check the Gateway login before the 16:00 run.")
        return 1
    print(f"OK {datetime.now():%Y-%m-%d %H:%M} — live Gateway healthy, account {live}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
