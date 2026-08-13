"""Daily entry point for the Magic Formula paper-trading book (US + Europe).

Runs one weekday: refresh ranking (monthly heavy universe pull, cached), fetch marks,
run the staggered daily loop (build/rotate), mark NAV, email the report.

Usage:
    python scripts/run_paper.py            # live paper (needs IB Gateway)
    python scripts/run_paper.py --dry-run  # no orders, no email send (prints/returns)

Schedule: weekdays 20:00 CET. Weekends are skipped automatically.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from paper.broker import Broker, ib_contract_spec  # noqa: E402
from paper.email_report import send_report  # noqa: E402
from paper.live_data import _fx_to_usd, fetch_live_panels  # noqa: E402
from paper.orchestrator import PaperConfig, run_daily  # noqa: E402
from risk_guard import (install_alert_collector, missed_runs,  # noqa: E402
                        push_if_alerts, reconcile, halt_state,
                        HALT_ALL, HALT_NEW, circuit_breaker, peak_equity)
from paper.rank import todays_ranking  # noqa: E402
from paper.state import PortfolioState  # noqa: E402
from paper.universe import paper_universe  # noqa: E402
from strategies.magic_formula import EnhancedMagicConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Collect WARNING+ so the daily email can carry it. Without this the guards write to run.log and
# nobody sees them; an alert that is not delivered is not an alert.
ALERTS = install_alert_collector()
load_dotenv(ROOT / ".env")

STATE_FILE = ROOT / "results" / "paper" / "state.json"
RANK_CACHE = ROOT / "results" / "paper" / "ranking.json"
PANEL_CACHE = ROOT / "results" / "paper" / "panels.pkl"


def _refresh_ranking(today: str, cfg_mf: EnhancedMagicConfig):
    """Monthly heavy universe pull -> cached ranking + panels. Reuse within the month."""
    import pickle
    stale = True
    if RANK_CACHE.exists():
        meta = json.loads(RANK_CACHE.read_text())
        stale = meta.get("month") != today[:7]
    if not stale and PANEL_CACHE.exists():
        panels = pickle.loads(PANEL_CACHE.read_bytes())
        ranking = pd.Series(json.loads(RANK_CACHE.read_text())["ranking"])
        logging.info("Using cached ranking (%d names) for %s", len(ranking), today[:7])
        return ranking, panels
    logging.info("Monthly refresh: pulling universe …")
    tickers = paper_universe()
    logging.info("Universe: %d names — pulling fundamentals (gentle pacing) …", len(tickers))
    panels = fetch_live_panels(tickers, price_days=500, pause=0.15)
    ranking = todays_ranking(panels, cfg_mf, min_mcap_usd=500e6)
    RANK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RANK_CACHE.write_text(json.dumps({"month": today[:7], "ranking": ranking.to_dict()}))
    PANEL_CACHE.write_bytes(pickle.dumps(panels))
    logging.info("Ranking refreshed: %d eligible names", len(ranking))
    return ranking, panels


def _refresh_marks(panels: dict, tickers: set[str]) -> dict:
    """Patch TODAY's prices into the (monthly-cached) price panel for a small set of
    tickers — held names + top candidates — so daily P&L, sizing and marks are current
    while the heavy fundamentals/ranking stays monthly-cached."""
    tickers = sorted({t for t in tickers if t and t in panels["adj"].columns})
    if not tickers:
        return panels
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False, threads=True)
        px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(px, pd.Series):
            px = px.to_frame(tickers[0])
    except Exception as e:  # noqa: BLE001
        logging.warning("mark refresh failed (%s) — using cached prices", e)
        return panels
    adj = panels["adj"]
    day = pd.Timestamp.today().normalize()
    if day not in adj.index:
        adj.loc[day] = np.nan
    n = 0
    for t in px.columns:
        if t in adj.columns and px[t].notna().any():
            adj.loc[day, t] = float(px[t].dropna().iloc[-1])
            n += 1
    panels["adj"] = adj.sort_index()
    logging.info("Refreshed today's prices for %d/%d names.", n, len(tickers))
    return panels


def _spy_returns(inception: str | None):
    """(day_ret, since_inception_ret) for SPY."""
    try:
        spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False)["Close"].dropna()
        spy = spy.iloc[:, 0] if hasattr(spy, "columns") else spy
        day = float(spy.iloc[-1] / spy.iloc[-2] - 1)
        incep = None
        if inception:
            ref = spy[spy.index >= inception]
            if len(ref):
                incep = float(spy.iloc[-1] / ref.iloc[0] - 1)
        return day, incep, float(spy.iloc[-1])
    except Exception as e:  # noqa: BLE001
        logging.warning("SPY fetch failed: %s", e)
        return None, None, None


def main(dry_run: bool = False, force: bool = False) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    if datetime.now().weekday() >= 5 and not force:
        logging.info("Weekend (%s) — skipping (use --force to run anyway).", today)
        return

    cfg_mf = EnhancedMagicConfig(use_graham=False)
    cfg = PaperConfig()
    state = PortfolioState.load(STATE_FILE)

    # KILL SWITCH — FIRST, before the universe refresh. That pull takes ~13 minutes, so checking
    # afterwards would make a halted run do all the expensive work anyway, and would delay a
    # HALT_ALL exit by a quarter of an hour when the point is to stop promptly.
    _halt, _hwhy = halt_state(ROOT)
    if _halt == HALT_ALL:
        logging.error("HALTED (all): %s — exiting without trading", _hwhy)
        push_if_alerts(ALERTS, "Magic Formula")
        return
    if _halt == HALT_NEW:
        logging.warning("HALTED (new risk): %s — managing existing positions only", _hwhy)
        cfg.max_new_buys_per_day = 0

    # CIRCUIT BREAKER — drawdown from the book's own peak NAV. Thresholds sit OUTSIDE the range
    # the strategy is expected to produce (15/25/35%), because this is an operational failsafe for
    # "something is wrong", not a risk tool for normal losses; sizing handles those. It NEVER
    # auto-flattens: liquidating at a drawdown threshold is capitulating at the bottom, the same
    # mistake as the 2x stop removed from options-vrp. It stops NEW risk and shouts.
    _adj = panels["adj"]
    _marks = {t: float(_adj[t].dropna().iloc[-1]) for t in _adj.columns if _adj[t].notna().any()}
    _eq = state.nav(_marks, fx)
    _peak = peak_equity(state.nav_history, cfg.budget, key="nav", absolute=True)
    _blvl, _bscale, _bwhy = circuit_breaker(_eq, _peak)
    if _bwhy:
        (logging.error if _blvl == "halt" else logging.warning)("circuit breaker: %s", _bwhy)
    if _bscale <= 0:                      # reduce_only / halt: no NEW risk, closes still run
        cfg.max_new_buys_per_day = 0
    elif _bscale < 1.0:                   # derisk: smaller new positions
        cfg.vol_target *= _bscale

    ranking, panels = _refresh_ranking(today, cfg_mf)
    # Daily light refresh: current prices for held names + top candidates (marks/sizing/P&L).
    refresh_set = state.tickers | set(list(ranking.index)[:100])
    panels = _refresh_marks(panels, refresh_set)
    fx = _fx_to_usd(set(panels["currency"].values()))

    broker = Broker(host=os.getenv("IB_HOST", "127.0.0.1"),
                    port=int(os.getenv("IB_PORT", "7497")),
                    client_id=int(os.getenv("IB_CLIENT_ID", "5")),
                    dry_run=dry_run)
    if not dry_run:
        if not broker.connect():
            logging.error("Could not connect to IB — aborting run.")
            return
    try:
        summary = run_daily(state, ranking, panels, fx, broker, cfg, today)
        # RECONCILE while the IB connection is still OPEN — this must sit inside the try, before
        # disconnect(). State is what the strategy BELIEVES; when it is wrong, nothing inside the
        # strategy can tell. Report only, never auto-correct a shared account.
        if not dry_run:
            _actual = broker.stock_positions()
            if _actual is None:
                logging.warning("reconcile: IB positions unavailable — state NOT verified")
            else:
                # Key by the IB SYMBOL, not the yfinance ticker: state stores foreign names with
                # an exchange suffix (ASML.AS, AVIO.MI, SCYR.MC) while IB reports the bare symbol
                # (ASML, AVIO, SCYR). Comparing raw made every European holding show up as BOTH a
                # phantom (state's suffixed key) and an orphan (IB's bare key) — 12 false alarms
                # that drowned the one real orphan. ib_contract_spec is the same mapping the order
                # path uses, so the keys now line up.
                _exp: dict[str, float] = {}
                for p in state.positions:
                    _exp[ib_contract_spec(p.ticker)[0]] = _exp.get(ib_contract_spec(p.ticker)[0], 0.0) + float(p.shares)
                _d, _rnote = reconcile(_exp, _actual, label="equities")
                if _rnote:
                    logging.warning("%s", _rnote)
    finally:
        broker.disconnect()

    marks = summary["marks"]
    spy_day, spy_incep, spy_close = _spy_returns(state.inception_date)
    state.record_nav(today, state.nav(marks, fx), spy_close)

    if not dry_run:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state.save(STATE_FILE)

    # Retroactive heartbeat: logged as a WARNING so the alert collector carries it into the
    # email body AND the subject. Catches skipped days; cannot catch a permanently dead task,
    # which needs an external dead-man's switch.
    _missed, _last, _note = missed_runs(state.nav_history, today)
    if _note:
        logging.warning("heartbeat: %s", _note)
    body = send_report(state, marks, fx, spy_day, spy_incep, today, dry_run=dry_run,
                       alerts=ALERTS)
    # Out-of-band push, AFTER the email attempt so an SMTP failure is itself in what gets pushed.
    # The email cannot report its own failure; this is the only channel that can.
    if not dry_run:
        push_if_alerts(ALERTS, "Magic Formula")
    if dry_run:
        out = ROOT / "results" / "paper" / f"report_{today}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        logging.info("[DRY RUN] report written to %s", out)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv, force="--force" in sys.argv)
