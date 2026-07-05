"""Daily HTML email — same style as the contract-strategy report.

Shows: portfolio table (per-stock cost, mark, unrealized P/L %), realized + unrealized
P/L since inception, total NAV / return, and SPY performance vs the strategy (that day
and since inception).
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from paper.state import PortfolioState


def _pct(x) -> str:
    return "—" if x is None else f"{x*100:+.2f}%"


def _table(headers, rows) -> str:
    th = "".join(f"<th style='text-align:left;padding:4px 10px;border-bottom:2px solid #1a3c5e'>{h}</th>" for h in headers)
    trs = ""
    for r in rows:
        tds = "".join(f"<td style='padding:3px 10px;border-bottom:1px solid #e2e8f0'>{c}</td>" for c in r)
        trs += f"<tr>{tds}</tr>"
    return f"<table style='border-collapse:collapse;font-family:monospace;font-size:13px'>{th and '<tr>'+th+'</tr>'}{trs}</table>"


def build_email_body(state: PortfolioState, marks: dict, fx: dict,
                     spy_day_ret: float | None, spy_incep_ret: float | None,
                     today: str) -> str:
    unreal = state.unrealized_pnl(marks, fx)
    nav = state.nav(marks, fx)
    total_ret = nav / state.inception_nav - 1 if state.inception_nav else None
    # strategy day return from NAV history
    hist = state.nav_history
    day_ret = None
    if len(hist) >= 2 and hist[-1]["date"] == today:
        prev = hist[-2]["nav"]
        day_ret = (nav / prev - 1) if prev else None

    # portfolio rows
    rows = []
    for p in sorted(state.positions, key=lambda x: x.ticker):
        px = marks.get(p.ticker)
        f = fx.get(p.currency, p.entry_fx)
        if px is None:
            rows.append((p.ticker, f"{p.shares:g}", f"{p.entry_price:.2f} {p.currency}", "—", "—", "—")); continue
        pnl = p.pnl_usd(px, f)
        ret = px * f / (p.entry_price * p.entry_fx) - 1
        color = "#1a7f37" if pnl >= 0 else "#b91c1c"
        rows.append((p.ticker, f"{p.shares:g}", f"{p.entry_price:.2f} {p.currency}",
                     f"{px:.2f}", f"${p.value_usd(px,f):,.0f}",
                     f"<span style='color:{color}'>${pnl:+,.0f} ({ret*100:+.1f}%)</span>"))

    port_tbl = _table(["Ticker", "Shares", "Entry", "Mark", "Value USD", "Unrealized P/L"], rows)

    outperf = (total_ret - spy_incep_ret) if (total_ret is not None and spy_incep_ret is not None) else None
    summary = f"""
    <table style='font-family:monospace;font-size:13px;margin:8px 0'>
      <tr><td style='padding:2px 14px 2px 0'>NAV</td><td><b>${nav:,.0f}</b> &nbsp; (start ${state.inception_nav:,.0f}, since {state.inception_date})</td></tr>
      <tr><td>Total return</td><td><b>{_pct(total_ret)}</b></td></tr>
      <tr><td>Realized P/L</td><td>${state.realized_pnl:+,.0f}</td></tr>
      <tr><td>Unrealized P/L</td><td>${unreal:+,.0f}</td></tr>
      <tr><td>Cash</td><td>${state.cash:,.0f} &nbsp; ({len(state.positions)} positions)</td></tr>
      <tr><td style='padding-top:8px'>Strategy today</td><td style='padding-top:8px'>{_pct(day_ret)} &nbsp; vs SPY {_pct(spy_day_ret)}</td></tr>
      <tr><td>Strategy since inception</td><td>{_pct(total_ret)} &nbsp; vs SPY {_pct(spy_incep_ret)} &nbsp; (<b>{_pct(outperf)}</b> rel)</td></tr>
    </table>"""

    return f"""<html><body style='font-family:sans-serif;color:#1e293b'>
    <h2 style='color:#1a3c5e'>Magic Formula Paper — {today}</h2>
    {summary}
    <h3 style='color:#1a3c5e'>Portfolio</h3>
    {port_tbl}
    <p style='color:#64748b;font-size:11px;margin-top:14px'>US + Europe, ≥$500M, enhanced Magic Formula (Graham off, inverse-vol, 25% vol-target). Paper trading — fills optimistic.</p>
    </body></html>"""


def send_report(state: PortfolioState, marks: dict, fx: dict,
                spy_day_ret, spy_incep_ret, today: str, dry_run: bool = False) -> str:
    body = build_email_body(state, marks, fx, spy_day_ret, spy_incep_ret, today)
    nav = state.nav(marks, fx)
    total_ret = nav / state.inception_nav - 1 if state.inception_nav else 0
    subject = f"Magic Formula Paper — {today}: NAV ${nav:,.0f} ({total_ret*100:+.1f}%)"

    user, pw, to = os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"), os.getenv("TO_EMAIL")
    if dry_run or not all([user, pw, to]):
        if not dry_run:
            logging.warning("Email creds not set — skipping send.")
        return body
    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, pw)
            s.send_message(msg)
        logging.info("Report sent to %s", to)
    except Exception as e:  # noqa: BLE001
        logging.error("Email failed: %s", e)
    return body
