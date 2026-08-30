"""IB paper-trading bridge (ib_insync) — connection, marks, account, orders.

Reuses the contract-strategy patterns (auto-launch Gateway, retry). Adds US + European
contract resolution: a yfinance ticker like ``SAP.DE`` maps to an IB Stock on the right
exchange/currency. In dry-run mode no orders are placed and no connection is required —
so the daily loop and email can be tested offline.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time

# yfinance suffix -> (IB currency, IB primaryExchange). SMART routing + primaryExchange
# resolves most European listings.
SUFFIX_MAP: dict[str, tuple[str, str]] = {
    ".DE": ("EUR", "IBIS"),   # Xetra
    ".PA": ("EUR", "SBF"),    # Euronext Paris
    ".AS": ("EUR", "AEB"),    # Euronext Amsterdam
    ".MC": ("EUR", "BM"),     # Bolsa de Madrid
    ".MI": ("EUR", "BVME"),   # Borsa Italiana
    ".BR": ("EUR", "ENEXT.BE"),
    ".HE": ("EUR", "HEX"),
    ".LS": ("EUR", "BVLP"),
    ".IR": ("EUR", "ISE"),
    ".VI": ("EUR", "VSE"),
    ".SW": ("CHF", "EBS"),    # SIX Swiss
    ".L":  ("GBP", "LSE"),    # London
    ".ST": ("SEK", "SFB"),    # Stockholm
    ".CO": ("DKK", "CPH"),    # Copenhagen
    ".OL": ("NOK", "OSE"),    # Oslo
}


def ib_contract_spec(yf_ticker: str) -> tuple[str, str, str]:
    """(ib_symbol, currency, primaryExchange) for a yfinance ticker. US => SMART/USD.

    The symbol returned is the FIRST candidate; see `ib_symbol_candidates` for why one is not
    enough.
    """
    for suf, (ccy, exch) in SUFFIX_MAP.items():
        if yf_ticker.endswith(suf):
            return yf_ticker[: -len(suf)], ccy, exch
    return yf_ticker, "USD", ""


def ib_symbol_candidates(yf_ticker: str) -> list[str]:
    """Candidate IB symbols for a yfinance ticker, most likely first.

    yfinance and IB DISAGREE on share-class separators, and the disagreement is silent: an
    unresolvable contract makes `qualify` return None, the order is refused, and the name is
    skipped with a log line — so it sits in the universe and can never trade. Measured
    2026-08-30 against live IB, `VOLV-B.ST` / `NOVO-B.CO` / `RYA.IR` all failed with "no
    security definition found".

    yfinance writes the class with a HYPHEN (VOLV-B). MEASURED against live IB 2026-08-30 by
    sweeping all 32 hyphenated European names (scripts/ib_symbol_probe.py): IB uses a DOT for
    the overwhelming majority — 29 of 32 (VOLV.B, ASSA.B, BT.A, MAERSK.B ...) — a SPACE for one
    (NDA-FI.HE -> "NDA FI") and NO separator for two (NDA-SE.ST -> NDASE, ROCK-B.CO -> ROCKB).
    IB never uses the hyphen. So the dot leads the candidate list, and all 32 resolve across the
    four forms; the earlier order (which omitted the dot entirely) left every share class silent.

    `qualify` tries these in order and keeps whichever IB accepts, logging the winner. Cases
    where the TICKER ITSELF differs (not just its separator) cannot be derived and need
    `IB_SYMBOL_FIX`. NB Euronext Dublin (.IR) resolves NONE of its 20 names — the account has no
    Dublin permission — so that venue is dropped at the universe level (data/universe.py), not
    patched here.
    """
    base, _, _ = ib_contract_spec(yf_ticker)
    fixed = IB_SYMBOL_FIX.get(yf_ticker)
    out = [fixed] if fixed else []
    # Dot FIRST: it is IB's actual share-class convention, so for a hyphenated ticker it resolves
    # on the first attempt instead of after a failed hyphen probe. For a name with no hyphen all
    # four forms collapse to `base`, so plain and US tickers still yield exactly one candidate.
    for cand in (base.replace("-", "."), base, base.replace("-", " "), base.replace("-", "")):
        if cand and cand not in out:
            out.append(cand)
    return out


# Tickers where IB's symbol is not a formatting variant of yfinance's but a DIFFERENT symbol.
# Only add entries VERIFIED against live IB — a wrong entry here silently trades the wrong
# instrument, which is far worse than the unresolved-contract failure it replaces.
IB_SYMBOL_FIX: dict[str, str] = {}


# ccy -> (IB pair, is the ccy the BASE of that pair?). IB quotes minor currencies against USD
# as base (USDCHF, USDSEK...), majors the other way (EURUSD, GBPUSD), and an FX order quantity
# is ALWAYS in the pair's base currency. Getting this backwards does not fail loudly -- it
# doubles the exposure it was meant to close -- so the direction is derived here, as a pure
# function, and tested rather than assumed.
FX_PAIRS = {"EUR": ("EURUSD", True), "GBP": ("GBPUSD", True),
            "CHF": ("USDCHF", False), "SEK": ("USDSEK", False),
            "DKK": ("USDDKK", False), "NOK": ("USDNOK", False)}


def fx_order_spec(ccy: str, amount_ccy: float, rate_usd: float) -> tuple[str, str, int]:
    """(pair, action, quantity) to trade `amount_ccy` units of `ccy` against USD.

    Positive `amount_ccy` = ACQUIRE that currency (covering a short financed at the debit
    rate); negative = sell it back to USD. `rate_usd` is USD per 1 unit of ccy and is used only
    when USD is the pair's base, where the order size must be expressed in USD.
    """
    spec = FX_PAIRS.get(ccy)
    if spec is None:
        return ("", "", 0)
    pair, ccy_is_base = spec
    if ccy_is_base:
        action, qty = ("BUY" if amount_ccy > 0 else "SELL"), abs(amount_ccy)
    else:
        # USD is the base, so the direction INVERTS: acquiring CHF means selling USDCHF, and
        # the quantity is the USD amount, not the CHF amount.
        action = "SELL" if amount_ccy > 0 else "BUY"
        qty = abs(amount_ccy) * (rate_usd or 0.0)
    return (pair, action, int(round(qty)))


class Broker:
    def __init__(self, host="127.0.0.1", port=7497, client_id=5,
                 gateway_bat: str | None = None, dry_run: bool = False):
        self.host, self.port, self.client_id = host, port, client_id
        self.gateway_bat, self.dry_run = gateway_bat, dry_run
        self.ib = None
        self._contracts: dict[str, object] = {}

    # ---- connection ----
    def connect(self, max_retries: int = 3, startup_wait: int = 40) -> bool:
        from ib_insync import IB
        self.ib = IB()
        for attempt in range(1, max_retries + 1):
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=15)
                logging.info("IB connected (clientId %s, port %s).", self.client_id, self.port)
                return True
            except Exception as e:  # noqa: BLE001
                logging.warning("IB connect attempt %d/%d failed: %s", attempt, max_retries, e)
                if attempt == 1 and self.gateway_bat and os.path.exists(self.gateway_bat):
                    logging.info("Launching IB Gateway: %s", self.gateway_bat)
                    subprocess.Popen([self.gateway_bat], shell=True)
                    time.sleep(startup_wait)
                else:
                    time.sleep(10)
        return False

    def disconnect(self) -> None:
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()

    # ---- contracts / marks ----
    def qualify(self, ticker: str):
        if ticker in self._contracts:
            return self._contracts[ticker]
        from ib_insync import Stock
        _, ccy, exch = ib_contract_spec(ticker)
        cands = ib_symbol_candidates(ticker)
        self._contracts[ticker] = None
        for i, sym in enumerate(cands):
            c = (Stock(sym, "SMART", ccy, primaryExchange=exch) if exch
                 else Stock(sym, "SMART", ccy))
            try:
                q = self.ib.qualifyContracts(c)
            except Exception as e:  # noqa: BLE001
                logging.debug("qualify %s as %r: %s", ticker, sym, e)
                continue
            if q:
                if i > 0:
                    # Worth an INFO line: it means the primary form is wrong for this venue and
                    # the mapping should eventually be encoded rather than rediscovered daily.
                    logging.info("qualified %s as %r (candidate %d of %d)",
                                 ticker, sym, i + 1, len(cands))
                self._contracts[ticker] = q[0]
                break
        if self._contracts[ticker] is None:
            logging.warning("cannot resolve %s at IB — tried %s. It will NEVER trade; add a "
                            "verified entry to IB_SYMBOL_FIX if the ticker itself differs.",
                            ticker, cands)
        return self._contracts[ticker]

    def price(self, ticker: str) -> float | None:
        """Latest close (local currency) via a 1-day historical bar."""
        c = self.qualify(ticker)
        if c is None:
            return None
        try:
            bars = self.ib.reqHistoricalData(c, endDateTime="", durationStr="2 D",
                                             barSizeSetting="1 day", whatToShow="TRADES",
                                             useRTH=True, formatDate=1)
            return bars[-1].close if bars else None
        except Exception as e:  # noqa: BLE001
            logging.warning("price failed for %s: %s", ticker, e)
            return None

    def stock_positions(self) -> dict[str, float] | None:
        """{ticker -> signed shares} for STOCKS at IB, or None if unavailable.

        Filtered to secType STK on purpose: this account is shared with the options and futures
        strategies, and an unfiltered read would report their positions as orphans — which is
        exactly the pattern that once had one strategy flattening another's book.

        Returns None (not {}) when it cannot be read, so the caller can distinguish "flat" from
        "could not check"; an empty dict would make every held position look like a phantom.
        """
        if getattr(self, "dry_run", False) or self.ib is None:
            return None
        try:
            out: dict[str, float] = {}
            for it in self.ib.portfolio():
                c = it.contract
                if c.secType == "STK" and it.position:
                    out[c.symbol] = out.get(c.symbol, 0.0) + float(it.position)
            return out
        except Exception as e:  # noqa: BLE001
            logging.warning("stock_positions failed: %s", e)
            return None

    def margin_cushion(self) -> tuple[float, float] | None:
        """(excess liquidity, net liquidation) for the WHOLE account, or None.

        Account-wide on purpose: several strategies share this account, so the constraint
        genuinely is shared.

        ⚠ ExcessLiquidity, NOT MaintMarginReq (changed 2026-08-14). Maintenance margin on long
        stock is 25% of position value regardless of leverage, so a fully-invested unborrowed
        equity book read as 25% "used" and tripped the ceiling in normal operation. Excess
        liquidity is what an actual liquidation is measured against and is leverage-aware:
        it goes to zero only when the account is genuinely near forced liquidation.

        Returns None on any failure — the caller treats that as "unknown" and logs it, rather
        than assuming healthy.

        The dry-run / not-connected guard is NOT optional: without it this raises on every dry
        run, the exception is caught, and a WARNING is logged — which the alert collector then
        puts in the email subject and pushes to your phone. An alert channel that cries wolf on
        every offline run is worse than none, because you learn to ignore it.
        """
        if self.dry_run or self.ib is None:
            return None
        try:
            rows = {r.tag: r for r in self.ib.accountSummary()}
            xl = rows.get("ExcessLiquidity") or rows.get("FullExcessLiquidity")
            nl = rows.get("NetLiquidation")
            if not xl or not nl:
                return None
            return float(xl.value), float(nl.value)
        except Exception as e:  # noqa: BLE001
            logging.warning("margin_cushion failed: %s", e)
            return None

    def net_liq(self) -> tuple[float, str] | None:
        """(value, currency) of NetLiquidation — informational only (account base ccy;
        commingled if the account is shared with other strategies). Not used for sizing."""
        try:
            rows = [r for r in self.ib.accountSummary() if r.tag == "NetLiquidation"]
            if not rows:
                return None
            r = next((r for r in rows if r.currency in ("USD", "BASE")), rows[0])
            return float(r.value), r.currency
        except Exception as e:  # noqa: BLE001
            logging.warning("net_liq failed: %s", e)
            return None

    def ib_positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            for p in self.ib.positions():
                out[p.contract.symbol] = out.get(p.contract.symbol, 0.0) + p.position
        except Exception as e:  # noqa: BLE001
            logging.warning("positions failed: %s", e)
        return out

    # ---- orders ----
    # order-status buckets: live = filled or will fill; dead = will not fill
    _DEAD = {"Cancelled", "ApiCancelled", "Inactive", "Rejected"}

    def order(self, ticker: str, action: str, shares: int, wait: float = 20.0) -> dict:
        """Place a market order (BUY/SELL). Returns
            {ok, status, fill_price}
        ok=True if the order is live (filled or queued to fill); False if rejected/cancelled.
        fill_price is the actual avg fill when already filled, else None (caller falls back to
        the mark). Honours dry_run (logs, places nothing)."""
        if shares <= 0:
            return {"ok": False, "status": "zero_qty", "fill_price": None}
        if self.dry_run:
            logging.info("[DRY RUN] %s %d %s", action, shares, ticker)
            return {"ok": True, "status": "dryrun", "fill_price": None}
        from ib_insync import MarketOrder
        c = self.qualify(ticker)
        if c is None:
            logging.warning("cannot order %s — contract unresolved", ticker)
            return {"ok": False, "status": "unresolved", "fill_price": None}
        try:
            order = MarketOrder(action, shares)
            order.tif = "DAY"                      # explicit — avoids the preset TIF cancel/resubmit
            trade = self.ib.placeOrder(c, order)
            # Poll up to `wait`s, returning as soon as the order reaches a terminal state. A single
            # fixed sleep read the status while still PreSubmitted, so the email showed unfilled
            # orders that had actually filled a second later. Liquid names return in ~1s.
            waited = 0.0
            while waited < wait:
                self.ib.sleep(1.0)
                waited += 1.0
                if trade.orderStatus.status == "Filled" or trade.orderStatus.status in self._DEAD:
                    break
            st = trade.orderStatus.status
            if st in self._DEAD:
                logging.warning("%s %d %s NOT live (status=%s) — not recorded", action, shares, ticker, st)
                return {"ok": False, "status": st, "fill_price": None}
            fill = trade.orderStatus.avgFillPrice or None
            logging.info("%s %d %s -> %s%s", action, shares, ticker, st,
                         f" @ {fill}" if fill else " (queued, fills at next open)")
            return {"ok": True, "status": st, "fill_price": float(fill) if fill else None}
        except Exception as e:  # noqa: BLE001
            logging.error("order failed %s %s %d: %s", action, ticker, shares, e)
            return {"ok": False, "status": "error", "fill_price": None}

    # ---- FX cash sweep -------------------------------------------------------------------
    # WHY THIS EXISTS. IB does not auto-convert. Buying a EUR-denominated stock from a
    # USD-funded margin account leaves a NEGATIVE EUR cash balance financed at the first-tier
    # debit rate (measured 2026-08-28 on the IBKR UK schedule: EUR 3.697%, GBP 5.227%, CHF
    # 1.500%, SEK 3.154%, DKK 4.796% (BM+3%, the only one), NOK 5.636%). On a $50k book that is
    # ~50% European that runs ~$1,094/yr gross, ~$822/yr net of the USD credit forgone — 1.64%
    # of NAV against a strategy whose whole edge is a few percent. Sweeping costs $2 per FX
    # order, so ~$144/yr at six currencies swept monthly. Converting saves ~1.36% of NAV/yr.
    #
    # The borrow IS a partial FX hedge (long stock in EUR, short EUR cash leaves the principal
    # naturally hedged) and closing it raises measured book vol 15.68% -> 16.55%. That is a
    # real cost, and it loses decisively: 1.36% of NAV/yr to avoid 0.87pp of vol on a 16.5%-vol
    # book. Converting ALSO makes the book's own accounting correct — Position.pnl_usd models a
    # fully unhedged position, which is only true once the balance is swept.

    # ccy -> (IB pair, is the ccy the BASE of that pair?). IB quotes minor currencies against
    # USD as base (USDCHF, USDSEK...), majors the other way (EURUSD, GBPUSD), and the order
    # quantity is always in the pair's BASE currency — which is why the direction has to be
    # tracked rather than assumed.
    FX_PAIRS = FX_PAIRS      # module-level; kept as an attribute for callers that had it

    def cash_balances(self) -> dict[str, float]:
        """{currency: settled cash balance}. Empty dict if unavailable — callers must treat
        that as "do not sweep" rather than "nothing to sweep"."""
        if self.dry_run:
            return {}
        out: dict[str, float] = {}
        try:
            for v in self.ib.accountValues():
                if v.tag == "CashBalance" and v.currency not in ("BASE", ""):
                    out[v.currency] = out.get(v.currency, 0.0) + float(v.value)
        except Exception as e:  # noqa: BLE001
            logging.warning("cash_balances failed: %s", e)
            return {}
        return out

    def convert_fx(self, ccy: str, amount_ccy: float, rate_usd: float,
                   wait: float = 20.0) -> dict:
        """Trade `amount_ccy` units of `ccy` against USD. Positive = ACQUIRE that currency
        (covers a short balance); negative = sell it back to USD.

        `rate_usd` is USD per 1 unit of ccy, used only to express the order size when USD is
        the pair's base currency.
        """
        if ccy == "USD" or abs(amount_ccy) < 1.0:
            return {"ok": False, "status": "noop", "filled": 0.0}
        spec = self.FX_PAIRS.get(ccy)
        if spec is None:
            logging.warning("no FX pair mapped for %s — not swept", ccy)
            return {"ok": False, "status": "unmapped", "filled": 0.0}
        pair, action, qty = fx_order_spec(ccy, amount_ccy, rate_usd)
        if qty <= 0:
            return {"ok": False, "status": "zero_qty", "filled": 0.0}
        if self.dry_run:
            logging.info("[DRY RUN] FX %s %d %s (%+.0f %s)", action, qty, pair, amount_ccy, ccy)
            return {"ok": True, "status": "dryrun", "filled": float(amount_ccy)}
        from ib_insync import Forex, MarketOrder
        try:
            c = Forex(pair)
            self.ib.qualifyContracts(c)
            order = MarketOrder(action, qty)
            order.tif = "DAY"
            trade = self.ib.placeOrder(c, order)
            waited = 0.0
            while waited < wait:
                self.ib.sleep(1.0)
                waited += 1.0
                if trade.orderStatus.status == "Filled" or trade.orderStatus.status in self._DEAD:
                    break
            st = trade.orderStatus.status
            if st in self._DEAD:
                logging.warning("FX %s %d %s NOT live (status=%s)", action, qty, pair, st)
                return {"ok": False, "status": st, "filled": 0.0}
            logging.info("FX %s %d %s -> %s", action, qty, pair, st)
            return {"ok": True, "status": st, "filled": float(amount_ccy)}
        except Exception as e:  # noqa: BLE001
            logging.error("FX convert failed %s %s: %s", ccy, amount_ccy, e)
            return {"ok": False, "status": "error", "filled": 0.0}
