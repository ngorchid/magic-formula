"""Tests for the "Today's trades" section of the daily email (added 2026-10-16).

WHY THIS EXISTS. On 2026-10-15 the run sold MNST and bought ITH.L, and the email said nothing:
it showed only the resulting portfolio, so a rotation was visible only as a changed table row.
The section now lists every sell (fill, realized P/L, reason), every buy (fill, USD cost), the
clock restarts and the FX sweep, and the subject carries the counts. An unavailable summary
must read "unavailable", never "none".

Run: python scripts/test_email_trades.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper import email_report as er  # noqa: E402
from paper.state import PortfolioState, Position  # noqa: E402

fails, ran = [], 0


def expect(label: str, cond: bool, detail: str = "") -> None:
    global ran
    ran += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(label)


TODAY = "2026-10-15"


def _state():
    """The real 2026-10-15 rotation: MNST sold (clock, dropped from band), ITH.L bought."""
    st = PortfolioState(inception_date="2026-08-17", inception_nav=50000.0, cash=5364.74,
                        positions=[], realized_pnl=-20.96,
                        nav_history=[{"date": "2026-10-14", "nav": 48928.27},
                                     {"date": TODAY, "nav": 47541.5}])
    st.positions.append(Position("ITH.L", 406, 292.4, TODAY, 0.013262248, "GBp"))
    st.positions.append(Position("ZEAL.CO", 35, 300.0, TODAY, 0.155, "DKK"))
    st.trade_log.append({"ticker": "MNST", "shares": 16, "currency": "USD", "entry_price": 46.935,
                         "entry_date": "2026-09-16", "entry_fx": 1.0, "exit_price": 45.625,
                         "exit_fx": 1.0, "exit_date": TODAY, "pnl_usd": -20.96,
                         "reason": "clock: dropped from band"})
    # an OLD sell of the same kind must not appear today
    st.trade_log.append({"ticker": "OLD", "shares": 1, "currency": "USD", "entry_price": 1.0,
                         "entry_date": "2026-08-17", "entry_fx": 1.0, "exit_price": 2.0,
                         "exit_fx": 1.0, "exit_date": "2026-09-01", "pnl_usd": 1.0,
                         "reason": "clock: dropped from band"})
    return st


SUMMARY = {"date": TODAY, "buys": [("ITH.L", 406, 1575.0)], "sells": ["MNST"],
           "holds": ["ZEAL.CO"], "fx_sweep": [("GBP", 1196, "Filled")]}

print("\nTODAY'S TRADES")
st = _state()
body = er.build_email_body(st, {}, {}, None, None, TODAY, trades=SUMMARY)
sec = er._trades_section(st, SUMMARY, TODAY)
expect("the section is in the email", "Today's trades" in body)
expect("the sell is listed with side, shares, fill and currency",
       "<td style='padding:3px 10px;border-bottom:1px solid #e2e8f0'>SELL</td>" in sec
       and ">MNST<" in sec and ">16<" in sec and "45.62 USD" in sec, sec[:400])
expect("the sell shows its proceeds and realized P/L", "$730" in sec and "$-21</span> realized" in sec)
expect("the sell shows why", "clock: dropped from band" in sec)
expect("the buy is listed with shares, fill in its own currency and USD cost",
       ">BUY<" in sec and ">ITH.L<" in sec and ">406<" in sec and "292.40 GBp" in sec
       and "$1,575" in sec)
expect("an old sell is not listed as today's", ">OLD<" not in sec)
expect("held-through names are listed", "Held through" in sec and "ZEAL.CO" in sec)
expect("the FX sweep is listed", "FX sweep: GBP +1,196 (Filled)" in sec, sec[-200:])
expect("the section sits above the portfolio table",
       body.find("Today's trades") < body.find(">Portfolio<"))

print("\nQUIET AND UNKNOWN DAYS")
quiet = er._trades_section(st, {"buys": [], "sells": [], "holds": [], "fx_sweep": []}, TODAY)
expect("a day without trades says so explicitly", "Trades today: none" in quiet, quiet)
unk = er._trades_section(st, None, TODAY)
expect("no summary reads 'unavailable', never 'none'", "unavailable" in unk and "none" not in unk, unk)
hold_only = er._trades_section(st, {"buys": [], "sells": [], "holds": ["ZEAL.CO"], "fx_sweep": []}, TODAY)
expect("a day with only clock restarts says no buys or sells and lists them",
       "No buys or sells." in hold_only and "ZEAL.CO" in hold_only)

print("\nSUBJECT")
sent = {}


class _SMTP:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def login(self, *a): pass
    def send_message(self, msg):
        sent["subject"] = msg["Subject"]
        sent["body"] = msg.get_payload()[0].get_payload(decode=True).decode("utf-8")


er.smtplib.SMTP_SSL = _SMTP
import os  # noqa: E402
os.environ.update(EMAIL_USER="u", EMAIL_PASS="p", TO_EMAIL="t")
er.send_report(st, {}, {}, None, None, TODAY, trades=SUMMARY)
expect("the SENT email carries the section, not only the built body",
       "Today's trades" in sent.get("body", "") and "ITH.L" in sent.get("body", ""))
expect("the subject carries the trade counts", sent.get("subject", "").endswith("· 1 bought, 1 sold"),
       sent.get("subject"))
er.send_report(st, {}, {}, None, None, TODAY, trades={"buys": [], "sells": [], "holds": ["ZEAL.CO"]})
expect("no counts in the subject on a day without buys or sells", "bought" not in sent.get("subject", ""),
       sent.get("subject"))
er.send_report(st, {}, {}, None, None, TODAY)
expect("send_report without a summary still works (old callers)", "Magic Formula" in sent.get("subject", ""))

print("\nWIRING")
src = (ROOT / "scripts" / "run_paper.py").read_text(encoding="utf-8")
expect("the daily run hands run_daily's summary to the email",
       "alerts=ALERTS, trades=summary)" in src)

print(f"\n{ran - len(fails)}/{ran} passed")
if fails:
    print("FAILURE(S):", fails)
sys.exit(1 if fails else 0)
