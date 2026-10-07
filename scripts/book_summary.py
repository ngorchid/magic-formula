"""Cross-strategy BOOK summary email — the whole live book at a glance.

Now that more than one strategy trades real capital on account U27760647 (magic-formula since
2026-08-17, trend-overlay since 2026-09-14), this reports each sleeve AND the combined book:
daily P&L, since-inception P&L, and max drawdown observed; plus live account risk — maintenance
margin used, excess-liquidity buffer, and utilisation — read from IB.

It reads each strategy's own state.json (the source of realised/marked P&L history it already
writes daily) and normalises the two shapes:
  magic-formula : nav_history [{date, nav}]        -> P&L = nav - inception_nav   (base = inception_nav)
  trend-overlay : nav_history [{date, total_pnl}]  -> P&L = total_pnl             (base = effective budget)

Read-only: it places no orders. It connects to IB purely to read account margin. Paper sleeves
(options-vrp) are deliberately excluded — this is the LIVE book on U27760647, not a P&L blend
across accounts.

Run (from the magic-formula-live venv):  python scripts/book_summary.py
Env: TREND_STATE (path to trend-overlay-live state.json), TREND_BASE (its % base, default 75000),
OPTIONS_STATE (path to options-vrp-live state.json), OPTIONS_BASE (its % base, default 50000).
"""
from __future__ import annotations

import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

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

# --- strategy sources: (label, state.json path, pnl-kind, %-base) --------------------------------
# Return bases come from config/capital_bases.json, documented (with effective dates and a change
# log) in docs/capital_bases.md. TREND_BASE / OPTIONS_BASE in the env still override for a one-off,
# and the email then SAYS so -- an override is a config change and must be visible.
CAPITAL_BASES = json.loads((ROOT / "config" / "capital_bases.json").read_text())
MAGIC_STATE = ROOT / "results" / "paper" / "state.json"
TREND_STATE = Path(os.getenv("TREND_STATE",
    r"C:\Users\Nicolas\PycharmProjects\trend-overlay-live\results\paper\state.json"))
TREND_BASE = float(os.getenv("TREND_BASE", CAPITAL_BASES["trend-overlay"]["amount"]))
OPTIONS_STATE = Path(os.getenv("OPTIONS_STATE",
    r"C:\Users\Nicolas\PycharmProjects\options-vrp-live\results\paper\state.json"))
OPTIONS_BASE = float(os.getenv("OPTIONS_BASE", CAPITAL_BASES["options-vrp"]["amount"]))

# The book's inception CAPITAL — the account's funded size when magic-formula went live
# (2026-08-17). Book P&L since inception is (live NetLiq - this), so NAV ties to
# INCEPTION_CAPITAL + book P&L by construction. Override if capital is ever added/withdrawn.
INCEPTION_CAPITAL = float(os.getenv("BOOK_INCEPTION_CAPITAL", "50000"))
# Persisted daily NetLiq, so the BOOK daily P&L and drawdown come from the ACCOUNT itself rather
# than the summed sleeve ledgers. Builds up from the first run of this version onward.
NETLIQ_HIST = ROOT / "results" / "paper" / "book_netliq.json"


def _load_netliq_hist() -> list[dict]:
    if NETLIQ_HIST.exists():
        try:
            return json.loads(NETLIQ_HIST.read_text())
        except Exception:  # noqa: BLE001
            return []
    return []


def _save_netliq(hist: list[dict], today: str, net_liq: float) -> list[dict]:
    hist = [h for h in hist if h.get("date") != today]      # replace today's row if re-run
    hist.append({"date": today, "net_liq": round(float(net_liq), 2)})
    hist.sort(key=lambda h: h["date"])
    try:
        NETLIQ_HIST.parent.mkdir(parents=True, exist_ok=True)
        NETLIQ_HIST.write_text(json.dumps(hist, indent=2))
    except Exception:  # noqa: BLE001
        pass
    return hist


def _pnl_series(state: dict, base_default: float) -> tuple[dict[str, float], float, str | None]:
    """{date -> since-inception P&L}, %-base, inception. Handles all three sleeve shapes.

    magic-formula: nav_history = [{date, nav, ...}] (equity; P&L = nav - inception_nav).
    trend-overlay: nav_history = [{date, total_pnl}].
    options-vrp:   nav_history = [[date, total_pnl]] (list/tuple pairs, not dicts).
    """
    nh = state.get("nav_history", [])
    inception = state.get("inception_date")
    if not nh:
        return {}, base_default, inception
    first = nh[0]
    if isinstance(first, dict) and "nav" in first:          # magic-formula: equity series
        base = float(state.get("inception_nav") or 0.0) or 1.0
        series = {h["date"]: float(h["nav"]) - base for h in nh if h.get("nav") is not None}
        return series, base, inception
    if isinstance(first, dict):                             # trend-overlay: {date, total_pnl}
        series = {h["date"]: float(h["total_pnl"]) for h in nh if h.get("total_pnl") is not None}
        return series, base_default, inception
    # options-vrp: [date, total_pnl] pairs
    series = {h[0]: float(h[1]) for h in nh if len(h) >= 2 and h[1] is not None}
    return series, base_default, inception


def _options_margin(path) -> float | None:
    """Defined-risk collateral held by the options-vrp sleeve: sum of each spread's max loss
    (= (strike width − entry credit) × 100 × contracts), which IS the Reg-T/maintenance margin
    for a vertical credit spread. Computed from the sleeve's own book — per-strategy margin is not
    separable from the shared account otherwise. None if the state is absent/unreadable."""
    if not Path(path).exists():
        return None
    try:
        st = json.loads(Path(path).read_text())
        return sum((abs(sp["short_strike"] - sp["long_strike"]) - sp["entry_credit"])
                   * 100 * sp["contracts"] for sp in st.get("open_spreads", []))
    except Exception:                                       # noqa: BLE001
        return None


def _daily(series: dict[str, float]) -> float | None:
    """Latest P&L minus the previous day's — the day's change. None if <2 observations."""
    if len(series) < 2:
        return None
    ds = sorted(series)
    return series[ds[-1]] - series[ds[-2]]


def _latest(series: dict[str, float]) -> float:
    return series[max(series)] if series else 0.0


def _max_drawdown(series: dict[str, float]) -> float:
    """Largest peak-to-trough fall of the P&L curve, as a NEGATIVE number ($). 0 if never fell."""
    peak, mdd = float("-inf"), 0.0
    for d in sorted(series):
        peak = max(peak, series[d])
        mdd = min(mdd, series[d] - peak)
    return mdd if mdd != 0.0 else 0.0


def _combined(series_list: list[dict[str, float]]) -> dict[str, float]:
    """Book P&L per date = sum of each sleeve's P&L on that date, carrying the last known value
    forward for sleeves that had not started / did not run that day (so the book curve is
    continuous rather than dropping when a young sleeve has no row)."""
    all_dates = sorted({d for s in series_list for d in s})
    out: dict[str, float] = {}
    for d in all_dates:
        tot = 0.0
        for s in series_list:
            prior = [dd for dd in s if dd <= d]
            tot += s[max(prior)] if prior else 0.0
        out[d] = tot
    return out


def _ib_margin() -> dict | None:
    try:
        from ib_insync import IB, util
        util.logToConsole("CRITICAL")
        ib = IB()
        ib.connect("127.0.0.1", int(os.getenv("IB_PORT", "4001")),
                   clientId=int(os.getenv("SUMMARY_CLIENT_ID", "22")), timeout=20)
        r = {x.tag: float(x.value) for x in ib.accountSummary()
             if x.tag in ("NetLiquidation", "ExcessLiquidity", "FullMaintMarginReq",
                          "GrossPositionValue")}
        ib.disconnect()
        return r or None
    except Exception as e:  # noqa: BLE001 — a margin-read failure must not kill the summary
        print(f"IB margin read failed: {type(e).__name__}: {e}")
        return None


# Above this share of NAV, the sleeve ledgers no longer explain the account: an unbooked fill, a
# capital flow not reflected in BOOK_INCEPTION_CAPITAL, or marking drift grown large. PROVISIONAL
# (2026-10-07): the normal size of FX financing + sweep commissions + yfinance-vs-IB marking is
# not yet measured; revisit once a few weeks of the line exist.
UNATTRIBUTED_WARN_FRAC = 0.01
RECORDS_DIR = Path(os.getenv("IB_RECORDS_DIR", r"C:\Users\Nicolas\IB-records"))


def ibkr_twr() -> tuple[float, str, str] | None:
    """(time-weighted return, from, to) as IBKR reports it in the Flex ChangeInNAV section of the
    most recent statement -- IBKR's own account-level TWR, the number for anything outside-facing.
    None when the archive or the field is unavailable. NB the daily query's window is rolling
    (7 days); an inception-to-date figure needs a second Flex query over that period."""
    p = RECORDS_DIR / "tables" / "change_in_nav.csv"
    if not p.exists():
        return None
    try:
        import csv
        with open(p, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("twr") not in (None, "")]
        if not rows:
            return None
        r = max(rows, key=lambda x: (x.get("toDate", ""), -int(x.get("fromDate", "0") or 0)))
        return float(r["twr"]), r.get("fromDate", ""), r.get("toDate", "")
    except Exception:  # noqa: BLE001
        return None


def overlap_line(magic_path, options_path) -> str:
    """Names held by BOTH magic-formula (stock) and options-vrp (a spread or delivered stock) --
    the owner's choice (2026-10-07) is to allow the overlap and watch it, not block it."""
    try:
        mg = json.loads(Path(magic_path).read_text()) if Path(magic_path).exists() else {}
        op = json.loads(Path(options_path).read_text()) if Path(options_path).exists() else {}
    except Exception:  # noqa: BLE001
        return "Names held by both sleeves: unknown (a sleeve state could not be read)"
    import re
    root = lambda t: re.sub(r"[^A-Z0-9]", "", t.upper().rsplit(".", 1)[0]  # noqa: E731
                            if "." in t and len(t.rsplit(".", 1)[1]) <= 3 else t.upper())
    held = {root(p["ticker"]): p for p in mg.get("positions", []) if p.get("ticker")}
    both = []
    for sp in op.get("open_spreads", []):
        t = root(sp.get("ticker", ""))
        if t in held:
            tag = " ASSIGNED" if sp.get("assigned_contracts") else ""
            both.append(f"{t} (magic {held[t].get('shares', 0):g} sh · VRP "
                        f"{sp.get('short_strike', 0):g}/{sp.get('long_strike', 0):g}P "
                        f"x{sp.get('contracts', 0)}{tag})")
    return "Names held by both sleeves: " + (", ".join(sorted(both)) if both else "none")


def reconciliation_lines(net_liq: float | None, sleeves: list[dict], inception: float,
                         bases: dict[str, float]) -> tuple[str, float | None]:
    """(html, unattributed). Two lines that make the sleeve figures tie VISIBLY to the account:

    1. the return bases, labelled for what they are -- overlapping denominators on shared
       collateral, NOT allocations (they sum to far more than the NAV, by design);
    2. NAV = inception capital + each sleeve's OWN ledger P&L + the unattributed remainder. The
       identity holds by definition, so the CHECK is the size of the remainder: above
       UNATTRIBUTED_WARN_FRAC of NAV the ledgers no longer explain the account, and it says so."""
    names = {"Magic Formula": "magic", "Trend Overlay": "trend", "Options VRP": "options"}
    b = " · ".join(f"{k} ${v:,.0f}" for k, v in bases.items())
    line1 = (f"Return bases (overlapping — shared collateral, not allocations; sum "
             f"${sum(bases.values()):,.0f}): {b}")
    if net_liq is None:
        return f"<p style='color:#64748b;font-size:11px;margin:4px 0'>{line1}</p>", None
    unattributed = net_liq - inception - sum(s["l_total"] for s in sleeves)
    terms = " + ".join(f"{names.get(s['label'], s['label'])} {_money(s['l_total'])}" for s in sleeves)
    big = abs(unattributed) > UNATTRIBUTED_WARN_FRAC * abs(net_liq)
    line2 = (f"NAV ${net_liq:,.0f} = inception ${inception:,.0f} + {terms} + unattributed "
             f"{_money(unattributed)}"
             + (f" ⚠ over {UNATTRIBUTED_WARN_FRAC:.0%} of NAV — the sleeve ledgers do not explain "
                f"the account (unbooked fill? capital flow?)" if big else ""))
    return (f"<p style='color:#64748b;font-size:11px;margin:4px 0'>{line1}<br>{line2}</p>",
            unattributed)


def _money(x) -> str:
    return f"${x:+,.0f}" if x is not None else "—"


def build() -> tuple[str, str]:
    today = datetime.now().strftime("%Y-%m-%d")

    # Account truth FIRST: the book P&L is derived from NetLiq so it ties to NAV exactly.
    m = _ib_margin()
    net_liq = m.get("NetLiquidation") if m else None

    # Per-sleeve P&L from each sleeve's OWN ledger.
    sleeves = []
    series_all = []
    for label, path, base_default in (("Magic Formula", MAGIC_STATE, 0.0),
                                      ("Trend Overlay", TREND_STATE, TREND_BASE),
                                      ("Options VRP", OPTIONS_STATE, OPTIONS_BASE)):
        if not Path(path).exists():
            continue
        st = json.loads(Path(path).read_text())
        series, base, inception = _pnl_series(st, base_default)
        if not series:
            continue
        series_all.append(series)
        sleeves.append({"label": label, "inception": inception, "base": base,
                        "l_daily": _daily(series), "l_total": _latest(series),
                        "mdd": _max_drawdown(series)})
    ledger_total_sum = sum(s["l_total"] for s in sleeves)
    ledger_daily_sum = (sum(s["l_daily"] for s in sleeves)
                        if sleeves and all(s["l_daily"] is not None for s in sleeves) else None)
    book_since = min((s["inception"] for s in sleeves if s["inception"]), default=today)
    sdates = sorted({d for s in series_all for d in s})

    total_drift = daily_drift = 0.0
    if net_liq is not None:
        nl_series = {h["date"]: h["net_liq"] - INCEPTION_CAPITAL
                     for h in _save_netliq(_load_netliq_hist(), today, net_liq)}
        book_total = net_liq - INCEPTION_CAPITAL          # ties to NAV by construction
        # Daily = the account's plain day-over-day NetLiq change (all costs/marking already in it),
        # measured over the sleeves' latest interval so it lines up with their dailies. None until
        # the NetLiq history (new) spans both dates.
        book_daily = (nl_series[sdates[-1]] - nl_series[sdates[-2]]
                      if len(sdates) >= 2 and sdates[-1] in nl_series and sdates[-2] in nl_series
                      else None)
        # Drawdown from the combined sleeve path (full history; NetLiq history is too new).
        book_mdd = _max_drawdown(_combined(series_all))
        total_drift = book_total - ledger_total_sum
        if book_daily is not None and ledger_daily_sum is not None:
            daily_drift = book_daily - ledger_daily_sum
    else:
        book = _combined(series_all)
        book_total, book_daily, book_mdd = _latest(book), _daily(book), _max_drawdown(book)

    # ATTRIBUTE the drift (FX financing, sweep commissions, yfinance-vs-IB marking) to the sleeve
    # that INCURS it — magic-formula holds the foreign stock and does the FX sweeps; trend-overlay
    # is IB-marked with negligible financing. Folding it into magic makes the two sleeves sum
    # EXACTLY to the account (NetLiq - inception), so the book row is genuinely their total.
    for s in sleeves:
        mag = s["label"] == "Magic Formula"
        s["total"] = s["l_total"] + (total_drift if mag else 0.0)
        s["daily"] = ((s["l_daily"] + daily_drift) if (mag and s["l_daily"] is not None)
                      else s["l_daily"])

    book_row = {"label": "WHOLE BOOK", "inception": book_since, "base": INCEPTION_CAPITAL,
                "daily": book_daily, "total": book_total, "mdd": book_mdd}

    def _row(s, bold=False):
        b = "font-weight:700;border-top:2px solid #334155" if bold else ""
        dd = f"{_money(s['mdd'])}" + (f" ({s['mdd'] / s['base']:.1%} of ${s['base']:,.0f})"
                                      if s["base"] else "")
        dcol = "#1a7f37" if (s["daily"] or 0) >= 0 else "#b91c1c"
        tcol = "#1a7f37" if (s["total"] or 0) >= 0 else "#b91c1c"
        return (f"<tr style='{b}'><td style='padding:3px 16px 3px 0'>{s['label']}</td>"
                f"<td style='color:{dcol}'>{_money(s['daily'])}</td>"
                f"<td style='color:{tcol}'>{_money(s['total'])}</td>"
                f"<td style='color:#b91c1c'>{dd}</td>"
                f"<td style='color:#64748b'>{s['inception'] or ''}</td></tr>")

    rows = "".join(_row(s) for s in sleeves) + _row(book_row, bold=True)
    pnl_tbl = (f"<table style='border-collapse:collapse;font-family:monospace;font-size:13px'>"
               f"<tr style='color:#64748b'><td style='padding-right:16px'>Sleeve</td><td>Daily P&amp;L</td>"
               f"<td>Since inception</td><td>Max drawdown</td><td>Since</td></tr>{rows}</table>")

    recon_note = ""
    if net_liq is not None:
        recon_note = (f"<p style='color:#64748b;font-size:11px;margin:4px 0'>WHOLE BOOK = account "
                      f"NetLiq − ${INCEPTION_CAPITAL:,.0f} inception capital (so NAV = ${INCEPTION_CAPITAL:,.0f} "
                      f"+ book P&amp;L), and daily = the day's NetLiq change. The account-vs-ledger "
                      f"difference ({_money(total_drift)} to date) — FX financing and sweep commissions "
                      f"plus yfinance-vs-IB marking (either sign) — is attributed to magic-formula, where "
                      f"it arises (trend is IB-marked), so the two sleeves sum to the book. Max drawdown "
                      f"is each sleeve's own path and is not additive.</p>")
        if book_daily is None:
            recon_note += ("<p style='color:#64748b;font-size:11px;margin:4px 0'>Book daily shows &mdash; "
                           "until the NetLiq history spans two trading days (new; fills from the 21:00 run).</p>")

    _bases = {"magic": next((s["base"] for s in sleeves if s["label"] == "Magic Formula"),
                            CAPITAL_BASES["magic-formula"]["amount"]),
              "trend": TREND_BASE, "options": OPTIONS_BASE}
    recon_html, _unattr = reconciliation_lines(net_liq, sleeves, INCEPTION_CAPITAL, _bases)
    _over = [f"{k} {v:,.0f} (documented {CAPITAL_BASES[cfg]['amount']:,.0f})"
             for k, v, cfg in (("trend", TREND_BASE, "trend-overlay"),
                               ("options", OPTIONS_BASE, "options-vrp"))
             if abs(v - CAPITAL_BASES[cfg]["amount"]) > 0.5]
    if _over:
        recon_html += ("<p style='color:#b45309;font-size:11px;margin:4px 0'>⚠ return base "
                       "OVERRIDDEN by env: " + "; ".join(_over) + " — log it in docs/capital_bases.md"
                       "</p>")
    recon_note += recon_html

    opt_margin = _options_margin(OPTIONS_STATE)   # options-vrp's own defined-risk collateral
    if m:
        nl, mm = m.get("NetLiquidation", 0.0), m.get("FullMaintMarginReq", 0.0)
        xl, gpv = m.get("ExcessLiquidity", 0.0), m.get("GrossPositionValue", 0.0)
        util = mm / nl if nl else 0.0
        opt_cell = (f"${opt_margin:,.0f} ({(opt_margin/nl if nl else 0):.0%} of NetLiq)"
                    if opt_margin is not None else "—")
        risk_tbl = (
            f"<table style='border-collapse:collapse;font-family:monospace;font-size:13px'>"
            f"<tr><td style='padding:2px 16px 2px 0'>Net liquidation</td><td>${nl:,.0f}</td></tr>"
            f"<tr><td>Gross position value</td><td>${gpv:,.0f}</td></tr>"
            f"<tr><td>Maintenance margin (used)</td><td>${mm:,.0f} ({util:.0%} of NetLiq)</td></tr>"
            f"<tr><td>&nbsp;&nbsp;↳ Options VRP margin</td><td>{opt_cell}</td></tr>"
            f"<tr><td>Excess liquidity (buffer)</td><td style='color:{'#1a7f37' if xl > 0.15*nl else '#b45309'}'>"
            f"${xl:,.0f} ({(xl/nl if nl else 0):.0%})</td></tr></table>")
    else:
        risk_tbl = "<i>account margin unavailable (Gateway down?) — P&L above still valid</i>"

    # Headline: the total portfolio value (= account NetLiquidation). Shown up top in plain dollars
    # (unsigned — it's a value, not a P&L), with the same number repeated in the margin table below.
    pv_str = f"${net_liq:,.0f}" if net_liq is not None else "—"
    _twr = ibkr_twr()
    twr_str = (f"{_twr[0]:+.2f}% ({_twr[1]}–{_twr[2]}, IBKR)" if _twr
               else "unavailable (no Flex ChangeInNAV TWR in the archive)")
    pv_line = (f"<p style='font-family:monospace;font-size:15px;margin:8px 0'>"
               f"<b>Total portfolio value:</b> <b style='color:#1a3c5e'>{pv_str}</b><br>"
               f"<b>Account TWR:</b> {twr_str}</p>"
               f"<p style='color:#64748b;font-size:11px;margin:4px 0'>Sleeves are shown in dollars; "
               f"any percentage is against the disclosed notional base in brackets "
               f"(docs/capital_bases.md). Use the account TWR for anything outside-facing.</p>"
               f"<p style='font-family:monospace;font-size:12px;margin:4px 0'>"
               f"{overlap_line(MAGIC_STATE, OPTIONS_STATE)}</p>")

    body = f"""<html><body style='font-family:sans-serif;color:#1e293b'>
    <h2 style='color:#1a3c5e'>Live Book Summary — {today}</h2>
    <p style='color:#64748b;font-size:12px'>Real capital, account U27760647. Magic Formula + Trend
    Overlay + Options VRP.</p>
    {pv_line}
    <h3 style='color:#1a3c5e'>P&amp;L</h3>
    {pnl_tbl}
    {recon_note}
    <h3 style='color:#1a3c5e'>Account risk</h3>
    {risk_tbl}
    <p style='color:#64748b;font-size:11px;margin-top:14px'>Read-only summary — no orders placed.</p>
    </body></html>"""

    bdaily = book_row["daily"]
    subject = (f"Live Book — {today}: value {pv_str}, day {_money(bdaily)}, "
               f"total {_money(book_row['total'])}")
    return subject, body


def main() -> None:
    subject, body = build()
    print(subject)
    u, p, to = os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"), os.getenv("TO_EMAIL")
    if not (u and p and to):
        print("(no email creds — printing only)")
        return
    msg = MIMEText(body, "html"); msg["Subject"] = subject; msg["From"] = u; msg["To"] = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(u, p); s.sendmail(u, [to], msg.as_string())
    print(f"book summary emailed to {to}")


if __name__ == "__main__":
    main()
