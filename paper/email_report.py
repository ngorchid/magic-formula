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



def _code_line() -> str:
    """The commit this run used (risk_guard.code_version), as the report's last line."""
    try:
        from risk_guard import code_version_note
        note = code_version_note()
    except Exception:  # noqa: BLE001 -- reporting must never break the run
        note = ""
    return (f"<p style='color:#64748b;font-size:11px'>Code: "
            f"{note or 'version not recorded'}</p>")

def _pct(x) -> str:
    return "—" if x is None else f"{x*100:+.2f}%"


def _usd(x) -> str:
    """A signed dollar figure, coloured green/red like the portfolio P/L cells."""
    if x is None:
        return "—"
    col = "#1a7f37" if x >= 0 else "#b91c1c"
    return f"<span style='color:{col}'>${x:+,.0f}</span>"


def _table(headers, rows) -> str:
    th = "".join(f"<th style='text-align:left;padding:4px 10px;border-bottom:2px solid #1a3c5e'>{h}</th>" for h in headers)
    trs = ""
    for r in rows:
        tds = "".join(f"<td style='padding:3px 10px;border-bottom:1px solid #e2e8f0'>{c}</td>" for c in r)
        trs += f"<tr>{tds}</tr>"
    return f"<table style='border-collapse:collapse;font-family:monospace;font-size:13px'>{th and '<tr>'+th+'</tr>'}{trs}</table>"


def _trades_section(state: PortfolioState, trades: dict | None, today: str) -> str:
    """Today's sells, buys, clock restarts and FX sweeps, from run_daily's summary.

    Before 2026-10-16 the email showed only the resulting portfolio, so a rotation (MNST sold,
    ITH.L bought on 2026-10-15) was visible only as a changed table row. Sells come from the
    trade log (fill price, realized P/L, reason), buys from the new positions (fill price, USD
    cost). `trades` None means the summary is unavailable -- said so, never shown as "no trades".
    """
    if trades is None:
        return "<p style='font-family:monospace;font-size:13px'>Trades today: unavailable</p>"
    sold = [r for r in state.trade_log
            if r.get("exit_date") == today and r.get("ticker") in set(trades.get("sells") or [])]
    rows = []
    for r in sold:
        col = "#1a7f37" if r["pnl_usd"] >= 0 else "#b91c1c"
        rows.append(("SELL", r["ticker"], f"{r['shares']:g}",
                     f"{r['exit_price']:.2f} {r['currency']}",
                     f"${r['shares'] * r['exit_price'] * r['exit_fx']:,.0f}",
                     f"<span style='color:{col}'>${r['pnl_usd']:+,.0f}</span> realized",
                     r.get("reason", "")))
    for b in trades.get("buys") or []:
        t, n, usd = b[0], b[1], b[2]
        p = state.get(t)
        price = f"{p.entry_price:.2f} {p.currency}" if p else "—"
        rows.append(("BUY", t, f"{n:g}", price, f"${usd:,.0f}", "", "new top-30 name"))
    holds = trades.get("holds") or []
    sweeps = trades.get("fx_sweep") or []
    if not rows and not holds and not sweeps:
        return "<p style='font-family:monospace;font-size:13px'>Trades today: none</p>"
    out = ("<h3 style='color:#1a3c5e'>Today's trades</h3>"
           + (_table(["Side", "Ticker", "Shares", "Fill", "USD", "P/L", "Why"], rows) if rows
              else "<p style='font-family:monospace;font-size:13px'>No buys or sells.</p>"))
    if holds:
        out += (f"<p style='font-family:monospace;font-size:13px;margin:6px 0'>Held through "
                f"(21-day clock up, still in the top 45, clock restarted): {', '.join(holds)}</p>")
    if sweeps:
        out += ("<p style='font-family:monospace;font-size:13px;margin:6px 0'>FX sweep: "
                + ", ".join(f"{c} {u:+,.0f} ({s})" for c, u, s in sweeps) + "</p>")
    return out


def build_email_body(state: PortfolioState, marks: dict, fx: dict,
                     spy_day_ret: float | None, spy_incep_ret: float | None,
                     today: str, trades: dict | None = None) -> str:
    unreal = state.unrealized_pnl(marks, fx)
    nav = state.nav(marks, fx)
    total_ret = nav / state.inception_nav - 1 if state.inception_nav else None
    # strategy day return AND absolute P&L from NAV history (prev = last recorded trading day)
    hist = state.nav_history
    day_ret = day_pnl = None
    if len(hist) >= 2 and hist[-1]["date"] == today:
        prev = hist[-2]["nav"]
        day_ret = (nav / prev - 1) if prev else None
        day_pnl = (nav - prev) if prev else None

    # SPY returns from the SAME stored run-time snapshots as the NAV — NOT the freshly-fetched
    # values passed in. The run fires at 16:00 CET ≈ 10:00 ET, 30 min into the US session, so every
    # price here (NAV marks AND SPY) is an intraday snapshot. The NAV path is snapshot-to-snapshot;
    # SPY must be too. Re-fetching compared today's partial bar against yesterday's SETTLED close
    # (which drifts intraday — up to 0.7% on SPY), so the since-inception line moved by more than
    # the day's move. Each stored `spy` is already a raw intraday close (we only ever stored
    # iloc[-1], the current bar, which is unadjusted), so the stored series is self-consistent.
    # The passed spy_*_ret remain as a fallback for the first run, before any snapshot pair exists.
    if hist and hist[-1]["date"] == today and hist[-1].get("spy"):
        spy_today = hist[-1]["spy"]
        if len(hist) >= 2 and hist[-2].get("spy"):
            spy_day_ret = spy_today / hist[-2]["spy"] - 1
        if hist[0].get("spy"):
            spy_incep_ret = spy_today / hist[0]["spy"] - 1

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
      <tr><td style='padding-top:8px'>Strategy today</td><td style='padding-top:8px'>{_usd(day_pnl)} ({_pct(day_ret)}) &nbsp; vs SPY {_pct(spy_day_ret)}</td></tr>
      <tr><td>Strategy since inception</td><td>{_pct(total_ret)} &nbsp; vs SPY {_pct(spy_incep_ret)} &nbsp; (<b>{_pct(outperf)}</b> rel)</td></tr>
    </table>"""

    label = os.getenv("BOOK_LABEL", "Paper")
    caveat = "Paper trading — fills optimistic." if label.lower() == "paper" else "Live trading — real fills."
    return f"""<html><body style='font-family:sans-serif;color:#1e293b'>
    <h2 style='color:#1a3c5e'>Magic Formula {label} — {today}</h2>
    {summary}
    {_trades_section(state, trades, today)}
    <h3 style='color:#1a3c5e'>Portfolio</h3>
    {port_tbl}
    <p style='color:#64748b;font-size:11px;margin-top:14px'>US + Europe, ≥$500M, enhanced Magic Formula (Graham off, inverse-vol, 25% vol-target). {caveat}</p>
    {_code_line()}
    </body></html>"""


def send_report(state: PortfolioState, marks: dict, fx: dict,
                spy_day_ret, spy_incep_ret, today: str, dry_run: bool = False,
                alerts=None, trades: dict | None = None) -> str:
    """`alerts` is an AlertCollector. Its records go at the TOP of the body and, crucially, a
    marker goes in the SUBJECT — the whole point is that a guard firing is visible without
    opening anything. Warnings previously went only to results/paper/run.log, which nobody
    reads daily, so an alert that fired was indistinguishable from one that never did."""
    body = build_email_body(state, marks, fx, spy_day_ret, spy_incep_ret, today, trades=trades)
    if alerts is not None and getattr(alerts, "records", None):
        body = alerts.html() + body
    nav = state.nav(marks, fx)
    total_ret = nav / state.inception_nav - 1 if state.inception_nav else 0
    mark = ""
    if alerts is not None and getattr(alerts, "worst", None):
        mark = f"[{alerts.worst} x{len(alerts.records)}] "
    traded = ""
    if trades and (trades.get("buys") or trades.get("sells")):
        traded = f" · {len(trades.get('buys') or [])} bought, {len(trades.get('sells') or [])} sold"
    subject = (f"{mark}Magic Formula {os.getenv('BOOK_LABEL', 'Paper')} — {today}: "
               f"NAV ${nav:,.0f} ({total_ret*100:+.1f}%){traded}")

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
