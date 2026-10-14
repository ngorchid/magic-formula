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

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from paper.broker import Broker, ib_contract_spec  # noqa: E402
from paper.email_report import send_report  # noqa: E402
from paper.live_data import _fx_to_usd, fetch_live_panels  # noqa: E402
from paper.orchestrator import PaperConfig, run_daily  # noqa: E402
from risk_guard import (code_version, install_alert_collector, email_if_alerts,  # noqa: E402
                        missed_runs,
                        push_if_alerts, reconcile, halt_state,
                        HALT_ALL, HALT_HARD, HALT_NEW, circuit_breaker, peak_equity,
                        documented_sizing, log_sizing,
                        data_fresh, RiskLimits, write_equity, book_drawdown, book_vol,
                        BreakerLevels, blended_vol)
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

# One id per run, stamped on every order as orderRef "magic-formula:<RUN_ID>" — so each fill in
# IB's executions and Flex statements traces back to this sleeve AND to this run's log section.
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
ORDER_REF = f"magic-formula:{RUN_ID}"


def make_broker(**kw) -> Broker:
    """The run's broker, tagged with this run's ORDER_REF. A factory rather than two inline lines
    so the tagging is testable: scripts/test_order_links.py asserts main() builds its broker here
    and that the tag is set, because a runner that forgets it sends every order out untagged
    while the broker-level tests stay green."""
    b = Broker(**kw)
    b.order_ref = ORDER_REF
    return b

# Annualised vol prior for the circuit-breaker levels: 19.0%, from the AUTHORITATIVE backtest —
# scripts/run_best_magic.py on the PIT survivorship-corrected S&P 500 with EDGAR deep-history
# fundamentals, 2012-01..2026-07 (14.6 yrs): ann +17.7%, vol 19.0%, Sharpe 0.96, maxDD -35.8%.
# Series: results/best_magic/best_sp500_pit_all.csv.
#
# ⚠ CORRECTED 2026-08-13 from 21.6%. That figure came from scripts/enhanced_magic_backtest.py,
# which I wrote after wrongly concluding no enhanced backtest existed — it does, and had since
# July. The substitute ran on the free SimFin BROAD universe over 4.9 years (Sharpe 0.49, vol
# 21.6%), a different and much weaker dataset. Prefer the PIT series for anything that matters.
#
# ⚠ UPDATE THIS if the configuration materially changes the book's risk (top_n, weighting,
# universe floor) — the recorded history will not reflect it for months.
#
# ⚠ CORRECTED 2026-08-28 from 19.0%, measured in scripts/magic_vol_prior_universe_lab.py
# (commit 656a504). The 19.0% came from the S&P 500 PIT backtest and so describes a US-ONLY
# book, but the live sleeve has held US + eurozone for months and now holds all of Europe.
# Measured vol of a 30-name equal-weight book, USD returns with unhedged FX included:
#     US only         19.22%   <- what 19.0% described; the method's validity check, since it
#                                 reproduces the PIT backtest's number almost exactly
#     US + eurozone   16.92%   <- ALREADY the live universe, so most of the staleness
#     US + all Europe 16.62%   <- the 2026-08-28 widening adds only a further -0.30pp
# Vol goes DOWN: diversification beats unhedged FX comfortably, and European large caps are
# individually LESS volatile in USD than US ones (17.92%) even carrying the currency move.
# Robust to the European share of the book — monotonic and saturating by ~40%, so anything
# from 20% upward implies 16.3-17.5%. 16.5% is the middle of that range.
#
# CONSEQUENCE: this TIGHTENS the breaker (derisk 22.80% -> 19.80%). The old levels were too
# WIDE, i.e. the book would have fallen further than intended before halving exposure.
#
# NB this is not a backtest and could not be — run_best_magic.py loads fundamentals with
# sources=("edgar",), US filings only, and the European names have no PIT membership data.
# It measures book vol from prices, which is what the prior actually claims. A random draw
# carries no value/quality tilt, so treat the deltas as the result and the levels as the check.
VOL_PRIOR = 0.165
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


def _ensure_held_prices(panels: dict, held: dict[str, str]) -> dict:
    """Add price history for HELD names that are missing from the panel. `held`: {ticker: ccy}.

    WHY. The monthly pull fetches the CURRENT universe only. A held name that has left its
    index (or changed ticker) is then absent from `adj`, gets no mark, and `PortfolioState.nav`
    skips it -- i.e. values it at ZERO until its clock sells it, up to 21 trading days. That
    understates NAV by ~3% per name and feeds the breaker a loss that is not real.

    PRICES ONLY. The ranking was computed (and cached) before this runs, so an added name gets no
    rank, stays out of the hold band, and is sold on its clock exactly as before: exits do not
    change. A held name still without a price afterwards is logged as an ERROR, never silent."""
    adj = panels["adj"]
    missing = sorted(t for t in held if t and t not in adj.columns)
    if not missing:
        return panels
    px = pd.DataFrame()
    try:
        raw = yf.download(missing, start=adj.index[0], auto_adjust=True, progress=False,
                          threads=True)
        if raw is not None and not raw.empty:
            px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if isinstance(px, pd.Series):
                px = px.to_frame(missing[0])
            if list(px.columns) == ["Close"]:
                px.columns = [missing[0]]
    except Exception as e:  # noqa: BLE001
        logging.warning("held-name price download failed (%s)", e)
    added = []
    for t in missing:
        if t in px.columns and px[t].notna().any():
            # Aligned to the panel's own dates: adding rows would change every other column.
            adj[t] = px[t].reindex(adj.index)
            panels["currency"].setdefault(t, held[t])
            added.append(t)
    panels["adj"] = adj
    if added:
        logging.warning("held name(s) not in this month's universe, prices added for valuation "
                        "only (no rank; sold on their clock): %s", ", ".join(added))
    lost = [t for t in missing if t not in added]
    if lost:
        logging.error("HELD position(s) with NO price: %s — valued at ZERO in NAV. Check for a "
                      "ticker change or delisting and handle by hand", ", ".join(lost))
    return panels


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
    # Every fetched bar goes on its OWN date (2026-10-13). price_sane compares the newest bar with
    # the one before it: if those sat on different split bases, a mid-month split read as a ~50%
    # crash and every buy of that name was rejected until the monthly rebuild (APH 2:1 on
    # 2026-09-03). The newest two bars now both come from this one split-adjusted fetch, so they
    # always share a basis; a genuine bad print in the newest bar still differs from its prior.
    #
    # Until 2026-10-13 only the newest close went on TODAY's row and the prior close overwrote "the
    # latest earlier row with data". The panel is reloaded from the monthly cache every run, so
    # that row was always the CACHE DATE: the days between were missing, and whenever the newest
    # close equalled the prior one (a London name at 16:00 CET, before its day's bar exists),
    # stale_columns read "unchanged since the cache date" and dropped a good candidate from buys
    # (DVN 10d, EN.PA 10d, BWLPG.OL 5d, INCY 8d, AAF.L 11d -- each exactly the days since the
    # rebuild). A name with no bar yet today now leaves today's row empty; marks and priors read
    # the last non-empty bars, so the mark is unchanged.
    adj = panels["adj"]
    day = pd.Timestamp.today().normalize()
    pidx = pd.DatetimeIndex(px.index)
    if pidx.tz is not None:
        pidx = pidx.tz_localize(None)
    px = px.set_axis(pidx.normalize(), axis=0)
    px = px[~px.index.duplicated(keep="last")]
    adj = adj.reindex(adj.index.union(px.index).union([day]))
    n = 0
    for t in px.columns:
        if t not in adj.columns:
            continue
        s = px[t].dropna()
        if s.empty:
            continue
        adj.loc[s.index, t] = s.values
        n += 1
    panels["adj"] = adj.sort_index()
    logging.info("Refreshed today's prices for %d/%d names.", n, len(tickers))
    return panels


def _repair_splits(panels: dict, tickers: set[str], lookback: int = 70) -> dict:
    """Back-adjust the monthly-cached panel for a split that landed AFTER the last rebuild.

    The cache is frozen between monthly rebuilds and is NOT re-split-adjusted. A mid-month split
    therefore leaves a ~2x step in the series: the bars `_refresh_marks` has patched since the
    split are post-split, the older cached bars are pre-split. `_refresh_marks` repairs the 2-bar
    price-sanity comparison, but `_annual_vol`'s 63-day window and the gross / inverse-vol sizing
    read the whole step as one enormous daily return — inflating that name's vol, shrinking its
    tilt, and dragging the blended gross scalar. (Momentum ranking is unaffected: it is rebuilt
    monthly from a fresh split-adjusted pull, not from this cached panel.)

    We find the step on the SMALL refreshed set, CONFIRM it is a real split via yfinance's
    authoritative actions — never from the price alone, since a genuine -50% crash must not be
    silently rescaled — and divide every pre-split bar by the split ratio. Idempotent: once the
    series is continuous the step is gone and nothing re-triggers, so re-running is a no-op.
    """
    adj = panels.get("adj")
    if adj is None:
        return panels
    repaired = 0
    for t in sorted(tickers):
        if t not in adj.columns:
            continue
        s = adj[t].dropna()
        if len(s) < 3:
            continue
        tail = s.tail(lookback)
        ratios = (tail.shift(1) / tail).dropna()          # prev/current; ~2.0 at a 2:1 split
        cand = ratios[(ratios > 1.35) | (ratios < 1.0 / 1.35)]   # steps past the price-sanity band
        if cand.empty:
            continue
        try:
            splits = yf.Ticker(t).splits
        except Exception as e:                                    # noqa: BLE001
            logging.warning("split check failed for %s: %s — vol may stay distorted", t, e)
            continue
        if splits is None or len(splits) == 0:
            continue                                             # a real move, not a split — leave it
        sp_idx = pd.DatetimeIndex(splits.index)
        sp_idx = sp_idx.tz_convert(None) if sp_idx.tz is not None else sp_idx
        sp = pd.Series(splits.values, index=sp_idx.normalize())
        for jump_date in cand.index:
            d = pd.Timestamp(jump_date).normalize()
            near = sp[(sp.index >= d - pd.Timedelta(days=5)) & (sp.index <= d + pd.Timedelta(days=5))]
            if near.empty:
                continue                                         # not a split — don't touch it
            ratio, ex = float(near.iloc[-1]), near.index[-1]
            if ratio <= 0:
                continue
            pre = adj.index < ex
            adj.loc[pre, t] = adj.loc[pre, t] / ratio
            repaired += 1
            logging.warning("SPLIT-ADJUSTED %s: divided %d pre-%s bars by %.3g — the monthly "
                            "cache was not re-adjusted for the mid-month split", t,
                            int(pre.sum()), ex.date(), ratio)
            break                                                # one split per name per run
    if repaired:
        panels["adj"] = adj.sort_index()
    return panels


def _spy_returns(inception: str | None):
    """(day_ret, since_inception_ret, latest_close) for SPY, on RAW (unadjusted) closes.

    auto_adjust=False is deliberate and was a BUG when True. yfinance's adjusted series is
    re-derived on EVERY fetch: when SPY pays a dividend the whole history is retroactively
    scaled down, so the fixed inception bar (e.g. 2026-08-17) returned a DIFFERENT level each
    run — ~0.56% lower six weeks in — with no market cause. That drift flowed straight into the
    "since inception vs SPY" line, which then jumped day to day even when the strategy tracked
    SPY almost exactly. Raw closes give a FIXED anchor (SPY does not split), so the since-
    inception number only moves with the actual price path.

    It also makes the comparison apples-to-apples: this book's NAV is PRICE return — it never
    credits dividends — so SPY must be measured price-return too. The adjusted series was
    handing SPY ~1.4%/yr of reinvested dividends the strategy does not receive. (Cost: on SPY's
    own ex-dividend day the raw close dips by the dividend, a small one-day artifact; the
    strategy's holdings take the same un-credited dip on THEIR ex-div days, so it stays fair.)
    """
    try:
        spy = yf.download("SPY", period="1y", auto_adjust=False, progress=False)["Close"].dropna()
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


def paper_config() -> PaperConfig:
    """magic-formula's config, its budget read from config/capital_bases.json (the single source
    for sizing budgets; owner's decision 2026-10-07).

    HOW THIS SLEEVE READS THE SHARED CONFIG -- so it is not flattened into a fixed 50k:
      * `budget` is the CAP on the deployment base. Sizing stays gap-to-NAV against
        min(NAV, budget) (orchestrator._slot_usd): below budget the book sizes to its own equity
        (it shrinks after losses), above it the base stops at budget (no compounding past it).
      * `budget` seeds inception_nav ONLY on a fresh, empty book (ensure_inception).
      * The RETURN base is the sleeve's own inception_nav from state.json; the config's `amount`
        documents it and is never used for sizing."""
    budget, src = documented_sizing(ROOT, "magic-formula")
    log_sizing("magic-formula", budget, src)
    return PaperConfig(budget=budget, no_buy=NO_BUY)


# Owner exclusions from NEW buys (orchestrator PaperConfig.no_buy): {ticker: (until, reason)}.
NO_BUY = {
    # Owner's decision 2026-10-13. Corteva separated on 2026-10-01 (Yahoo books it as a 6.665
    # "split": $77.65 -> $12.57). The October ranking (built 2026-10-01 10:20 CET, before the US
    # open) scores the PRE-separation company (rank 23), and stays frozen all month.
    # Extended to 2027-03-01 (owner, same day): the November-February rebuilds may still pair the
    # old company's reported earnings (the Jul-Sep quarter predates the separation) with the much
    # smaller post-separation market value -- an inflated earnings yield that could rank CTVA near
    # the top. By March its first post-separation results should be in the fundamentals.
    "CTVA": ("2027-03-01", "Corteva separation 2026-10-01; rankings until the first "
                           "post-separation results score the pre-separation company"),
}


def main(dry_run: bool = False, force: bool = False) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    if datetime.now().weekday() >= 5 and not force:
        logging.info("Weekend (%s) — skipping (use --force to run anyway).", today)
        return

    cfg_mf = EnhancedMagicConfig(use_graham=False)
    # BUDGET from config/capital_bases.json (an env BUDGET still overrides, loudly) -- see
    # paper_config() for how this NAV-linked sleeve uses it.
    cfg = paper_config()
    state = PortfolioState.load(STATE_FILE)

    # KILL SWITCH — FIRST, before the universe refresh. That pull takes ~13 minutes, so checking
    # afterwards would make a halted run do all the expensive work anyway, and would delay a
    # HALT_ALL exit by a quarter of an hour when the point is to stop promptly.
    # Report WHICH COMMIT is running before anything else. Placed above the kill switch so it
    # is recorded even on a halted run: "the box is running week-old code" is exactly the kind of
    # thing you want to learn from a halted day's log, not discover a month later.
    code_version(ROOT)
    _halt, _hwhy = halt_state(ROOT)
    # magic-formula has no SAFETY-flagged closes, so HALT_ALL and HALT_HARD both stop it entirely.
    if _halt in (HALT_ALL, HALT_HARD):
        logging.error("HALTED (%s): %s — exiting without trading", _halt, _hwhy)
        email_if_alerts(ALERTS, f"Magic Formula {_halt.upper()}", today)
        push_if_alerts(ALERTS, "Magic Formula")
        return
    if _halt == HALT_NEW:
        logging.warning("HALTED (new risk): %s — managing existing positions only", _hwhy)
        cfg.max_new_buys_per_day = 0

    ranking, panels = _refresh_ranking(today, cfg_mf)
    # Held names that left the universe at the monthly pull: prices only, AFTER the ranking.
    panels = _ensure_held_prices(panels, {p.ticker: p.currency for p in state.positions})
    # Daily light refresh: current prices for held names + top candidates (marks/sizing/P&L).
    refresh_set = state.tickers | set(list(ranking.index)[:100])
    panels = _refresh_marks(panels, refresh_set)
    # Back-adjust the cached panel for any split that landed after the last monthly rebuild, so the
    # 63-day vol window and the inverse-vol / gross sizing do not read the split step as a return.
    panels = _repair_splits(panels, refresh_set)
    fx = _fx_to_usd(set(panels["currency"].values()))

    # CIRCUIT BREAKER / STALENESS run AFTER the refresh above: both read `panels` (prices) and the
    # breaker reads `fx`, which _refresh_ranking / _refresh_marks / _fx_to_usd create. (The kill
    # switch stays first, before the ~13-min pull, because it needs neither. They cannot run
    # earlier — staleness is precisely a property of the freshly refreshed price panel.)
    # CIRCUIT BREAKER — drawdown from the book's own peak NAV. Thresholds sit OUTSIDE the range
    # the strategy is expected to produce (15/25/35%), because this is an operational failsafe for
    # "something is wrong", not a risk tool for normal losses; sizing handles those. It NEVER
    # auto-flattens: liquidating at a drawdown threshold is capitulating at the bottom, the same
    # mistake as the 2x stop removed from options-vrp. It stops NEW risk and shouts.
    # STALENESS. A frozen feed keeps returning plausible prices, so every downstream number —
    # marks, sizing, NAV, the rankings — still computes and still looks reasonable. price_sane
    # cannot catch it: each individual price is perfectly valid, just old. Fails CLOSED, because
    # trading on a stale picture of the market is worse than missing a day.
    _fresh = data_fresh(panels["adj"].index, pd.Timestamp(today), RiskLimits(budget=cfg.budget))
    if not _fresh:
        logging.error("data staleness: %s — NOT trading today", _fresh.reason)
        push_if_alerts(ALERTS, "Magic Formula")
        return

    _adj = panels["adj"]
    _marks = {t: float(_adj[t].dropna().iloc[-1]) for t in _adj.columns if _adj[t].notna().any()}
    _eq = state.nav(_marks, fx)
    _peak = peak_equity(state.nav_history, cfg.budget, key="nav", absolute=True)
    _peak = max(_peak, _eq)
    write_equity(ROOT.parent, "magic-formula", _eq, _peak)
    # Levels scaled to THIS book's own volatility. A fixed 15/25/35 is 1.2/2.0/2.8 sigma at
    # trend's 12.4% vol but only 0.7/1.2/1.6 sigma on a ~21% equity book — routine moves, not
    # failures. Backtested: the fixed levels cost 0.34 Sharpe here and fired at the bottom on
    # 13 of 13 triggers.
    _lv = BreakerLevels.from_vol(blended_vol(state.nav_history, cfg.budget, VOL_PRIOR,
                                             key="nav", absolute=True))
    _blvl, _bscale, _bwhy = circuit_breaker(_eq, _peak, _lv)
    if _bwhy:
        (logging.error if _blvl == "halt" else logging.warning)("circuit breaker: %s", _bwhy)
    # BOOK-level: three books each down 20% all sit under their own 25% threshold while the
    # total is down 20% — the correlated crash no single strategy can see. Worse of the two.
    _bdd, _beq, _bpk, _bnote = book_drawdown(ROOT.parent)
    if _bdd is not None:
        # Book levels scale to the BOOK's OWN vol, not a hard-coded tighter set. With one
        # strategy live the book curve IS that strategy's, so the levels come out identical
        # and this adds nothing — correct, since there is no diversification to reward.
        # Skipped entirely until the book has enough history to estimate its vol.
        _bvol = book_vol(ROOT.parent)
        _lvl2, _sc2, _why2 = (circuit_breaker(_beq, _bpk, BreakerLevels.from_vol(_bvol))
                              if _bvol else ('ok', 1.0, ''))
        if _why2:
            logging.warning("BOOK circuit breaker: %s | %s", _why2, _bnote)
        _bscale = min(_bscale, _sc2)
    if _bscale <= 0:                      # reduce_only / halt: no NEW risk, closes still run
        cfg.max_new_buys_per_day = 0
    elif _bscale < 1.0:                   # derisk: smaller new positions
        cfg.vol_target *= _bscale

    broker = make_broker(host=os.getenv("IB_HOST", "127.0.0.1"),
                         port=int(os.getenv("IB_PORT", "7497")),
                         client_id=int(os.getenv("IB_CLIENT_ID", "5")),
                         dry_run=dry_run)
    logging.info("run id %s — orders tagged orderRef=%s", RUN_ID, ORDER_REF)
    if not dry_run:
        if not broker.connect():
            logging.error("Could not connect to IB — aborting run.")
            email_if_alerts(ALERTS, "Magic Formula", today)
            return
    try:
        summary = run_daily(state, ranking, panels, fx, broker, cfg, today)
        # RECONCILE while the IB connection is still OPEN — this must sit inside the try, before
        # disconnect(). State is what the strategy BELIEVES; when it is wrong, nothing inside the
        # strategy can tell. Report only, never auto-correct a shared account.
        if not dry_run:
            # Let IB's position feed catch up to fills just placed above. Without this the
            # read races the just-filled order and reports it as a PHANTOM (state N vs broker 0)
            # — a false alarm that lands in the LIVE email. Same feed-lag fix as trend-overlay.
            broker.ib.sleep(3)
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
