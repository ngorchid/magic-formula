"""Tests for the download job's success signals (2026-10-07).

A failure-only email cannot report a job that never ran. On every completed run main() sends a
one-line success email (also on a day with no activity) and, when HEARTBEAT_URL is set, pings it
-- quietly, carrying nothing but the request, never failing the download.
scripts/mutate_records_heartbeat.py seeds the faults. No IB, no network.

Run: python scripts/test_records_heartbeat.py
"""
from __future__ import annotations

import os
import sys

from _ib_records_fixtures import (ALERTS, cash, check, d, empty_ledgers, finish, fresh, put, stmt,
                                  trade)

PINGS: list = []
URL = "https://hc.example.invalid/ping/abc-123"


def fake_urlopen(req, timeout=None):
    PINGS.append(req)

    class R:
        def read(self, n=-1):
            return b"OK"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return R()


def run(statement=None, fail=None, argv=("download_ib_records.py",), url=URL):
    ALERTS.clear()
    PINGS.clear()
    os.environ["FLEX_TOKEN"], os.environ["FLEX_QUERY_ID"] = "tok", "999"
    if url is None:
        os.environ.pop("HEARTBEAT_URL", None)
    else:
        os.environ["HEARTBEAT_URL"] = url

    def dl(token, qid, max_wait=300):
        if fail:
            raise fail
        return statement
    d.download = dl
    sys.argv = list(argv)
    try:
        return d.main()
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        return f"crashed: {type(e).__name__}: {e}"


d.urlopen = fake_urlopen
M = "magic-formula:20261007-160002"
# What a REAL quiet day looks like: no trades and no cash lines, but IB still reports the open
# positions and a NAV bridge of zeros. Earlier activity sits in the archive (rolling window).
POS = ('<OpenPosition reportDate="{d}" conid="265598" symbol="AAPL" currency="USD" '
       'position="1" assetCategory="STK"/>')
NAV0 = ('<ChangeInNAV fromDate="{f}" toDate="{t}" dividends="0" interest="0" commissions="0" '
        'transactionTax="0" otherFees="0" withholdingTax="0" depositsWithdrawals="0"/>')
EARLIER = (trade("e0", "AAPL", "265598", "", "BUY", 1, 10.0, date="20261001")
           + cash("c0", "Broker Interest Received", "0.00", "USD CREDIT INT FOR SEP-2026")
           + POS.format(d="20261001"))


def quiet(frm, to):
    return stmt(frm, to, POS.format(d=to) + NAV0.format(f=frm, t=to))

print("SUCCESS SIGNALS")
fresh()
empty_ledgers()
put("20260930", "20261006", EARLIER, "2026-10-06T08:30:00")
rc = run(quiet("20261001", "20261007"))
subjects = [s for s, _ in ALERTS]
check("a quiet day still sends the one-line success email",
      rc == 0 and len(subjects) == 1 and subjects[0].startswith("[IB records] OK"), str(subjects))
check("...and pings the dead-man's switch once", len(PINGS) == 1, str(PINGS))
req = PINGS[0] if PINGS else None
check("the ping goes to exactly HEARTBEAT_URL and carries NO data (plain GET)",
      req is not None and req.full_url == URL and req.data is None
      and req.get_method() == "GET", str(req and (req.full_url, req.data)))

print("\nTHE PING NEVER BLOCKS OR FAILS THE DOWNLOAD")


def boom(req, timeout=None):
    raise OSError("network down")


d.urlopen = boom
fresh()
empty_ledgers()
put("20260930", "20261006", EARLIER, "2026-10-06T08:30:00")
rc = run(quiet("20261001", "20261007"))
check("a failing ping is swallowed: run still succeeds and the statement is archived",
      rc == 0 and len(list((d.RECORDS / "raw").rglob("*.xml"))) == 2, f"rc={rc}")
d.urlopen = fake_urlopen
rc = run(quiet("20261002", "20261008"), url=None)
check("no HEARTBEAT_URL -> no ping, success email still sent",
      rc == 0 and PINGS == [] and ALERTS and ALERTS[0][0].startswith("[IB records] OK"),
      str((PINGS, ALERTS)))

print("\nFAILURES DO NOT LOOK LIKE SUCCESS")
rc = run(fail=d.FlexError("1012", "token expired"))
check("a failed download sends the failure alert, no OK email, no ping",
      rc == 1 and PINGS == [] and ALERTS and "download failed" in ALERTS[0][0]
      and not any(s.startswith("[IB records] OK") for s, _ in ALERTS), str((rc, ALERTS)))
rc = run(stmt("20261003", "20261009", trade("u9", "TSLA", "76792991", "", "BUY", 1, 250.0,
                                             date="20261008")
              + POS.format(d="20261009") + NAV0.format(f="20261003", t="20261009")
              .replace('commissions="0"', 'commissions="-1"')))
check("audit warnings -> the failure email INSTEAD of the OK email",
      rc == 0 and ALERTS and ALERTS[0][0] == "[IB records] audit trail incomplete"
      and not any(s.startswith("[IB records] OK") for s, _ in ALERTS), str(ALERTS))
check("...and the job still pings (it ran; the email carries the problem)", len(PINGS) == 1,
      str(PINGS))
rc = run(argv=("download_ib_records.py", "--rebuild"))
check("--rebuild (offline) sends no email and no ping", rc == 0 and ALERTS == [] and PINGS == [],
      str((ALERTS, PINGS)))

finish("download-heartbeat")
