"""Tests for scripts/check_live_gateway.py — no Gateway, no email, no relaunch, no waiting.

Each case scripts what the two connection attempts return and records what the check then does
(sleep, relaunch, alert, exit code). scripts/mutate_live_gateway_check.py seeds the faults.

Run: python scripts/test_live_gateway_check.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gw_check", ROOT / "scripts" / "check_live_gateway.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)
_real_session_disconnected = g.session_disconnected

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def case(attempts: list, disconnected=False):
    """attempts: per connection attempt, a list of accounts or an Exception to raise.
    disconnected: what session_disconnected() reports (True / False / None = unknown)."""
    seq = list(attempts)
    log = {"connects": 0, "sleeps": [], "relaunch": 0, "alerts": []}

    def fake_accounts():
        log["connects"] += 1
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    g.managed_accounts = fake_accounts
    g.time.sleep = lambda s: log["sleeps"].append(s)
    g.relaunch_gateway = lambda: (log.__setitem__("relaunch", log["relaunch"] + 1), True)[1]
    g.alert = lambda subject, body: log["alerts"].append(subject + " | " + body)
    g.session_disconnected = lambda: disconnected
    try:
        log["rc"] = g.main()
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        log["rc"] = f"crashed: {type(e).__name__}: {e}"
    return log


print("HEALTHY")
r = case([["U27760647"]])
check("up on the first try: exit 0, ONE connect, no wait, no relaunch, no alert",
      r["rc"] == 0 and r["connects"] == 1 and r["sleeps"] == [] and r["relaunch"] == 0
      and r["alerts"] == [], str(r))

print("\nTHE DAILY 12:00 SELF-RESTART — down for a moment, back on the retry")
r = case([ConnectionRefusedError("refused"), ["U27760647"]])
check("first attempt fails, second succeeds: exit 0, NO alert, NO relaunch",
      r["rc"] == 0 and r["alerts"] == [] and r["relaunch"] == 0 and r["connects"] == 2, str(r))
check("...after waiting RETRY_DELAY_S between the attempts", r["sleeps"] == [g.RETRY_DELAY_S], str(r))
check("RETRY_DELAY_S outlasts the Gateway's restart gap (at least 2 minutes)",
      g.RETRY_DELAY_S >= 120, str(g.RETRY_DELAY_S))
r = case([TimeoutError(), ["DU1234567"]])
check("a retry that finds only a PAPER account still alerts (no live account)",
      r["rc"] == 1 and any("NO live account" in a for a in r["alerts"]) and r["relaunch"] == 0, str(r))

print("\nREALLY DOWN — both attempts fail")
r = case([ConnectionRefusedError("refused"), ConnectionRefusedError("still refused")])
check("both fail: relaunch ONCE, ONE alert, exit 1",
      r["rc"] == 1 and r["relaunch"] == 1 and len(r["alerts"]) == 1, str(r))
check("...the alert asks to approve the 2FA push and names the last error",
      any("APPROVE" in a and "still refused" in a for a in r["alerts"]), str(r["alerts"]))
check("...and the relaunch comes only AFTER the retry wait", r["sleeps"] == [g.RETRY_DELAY_S], str(r))

print("\nREMOTE DESKTOP DISCONNECTED — the Gateway cannot show its login window, no push comes")
r = case([ConnectionRefusedError("refused"), ConnectionRefusedError("still refused")], disconnected=True)
check("still relaunches ONCE (it logs in as soon as someone connects), ONE alert, exit 1",
      r["rc"] == 1 and r["relaunch"] == 1 and len(r["alerts"]) == 1, str(r))
check("...the alert says CONNECT BY REMOTE DESKTOP, not 'approve the push'",
      len(r["alerts"]) == 1 and "CONNECT BY REMOTE DESKTOP" in r["alerts"][0]
      and "APPROVE the 2FA push" not in r["alerts"][0] and "Disconnect to console" in r["alerts"][0]
      and "still refused" in r["alerts"][0], str(r["alerts"]))
r = case([ConnectionRefusedError("refused"), ConnectionRefusedError("still refused")], disconnected=None)
check("session state unknown -> the usual 'approve the push' alert",
      len(r["alerts"]) == 1 and "APPROVE the 2FA push" in r["alerts"][0], str(r["alerts"]))
r = case([ConnectionRefusedError("refused"), ["U27760647"]], disconnected=True)
check("disconnected but the Gateway answers on the retry -> no alert (only a DOWN Gateway matters)",
      r["rc"] == 0 and r["alerts"] == [] and r["relaunch"] == 0, str(r))
try:
    _probe = _real_session_disconnected()
except Exception as e:  # noqa: BLE001
    _probe = f"crashed: {type(e).__name__}"
check("the real session probe returns True/False/None and never raises (this box: a bool)",
      _probe in (True, False, None), str(_probe))

print("\nLOGGED IN, BUT NOT LIVE")
r = case([["DU1234567"]])
check("only a paper account: alert, exit 1, no relaunch",
      r["rc"] == 1 and r["relaunch"] == 0 and any("NO live account" in a for a in r["alerts"]), str(r))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} gateway-check checks behaved as expected")
