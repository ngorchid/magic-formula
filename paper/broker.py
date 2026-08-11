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
    """(ib_symbol, currency, primaryExchange) for a yfinance ticker. US => SMART/USD."""
    for suf, (ccy, exch) in SUFFIX_MAP.items():
        if yf_ticker.endswith(suf):
            return yf_ticker[: -len(suf)], ccy, exch
    return yf_ticker, "USD", ""


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
        sym, ccy, exch = ib_contract_spec(ticker)
        c = Stock(sym, "SMART", ccy, primaryExchange=exch) if exch else Stock(sym, "SMART", ccy)
        try:
            q = self.ib.qualifyContracts(c)
            self._contracts[ticker] = q[0] if q else None
        except Exception as e:  # noqa: BLE001
            logging.warning("qualify failed for %s: %s", ticker, e)
            self._contracts[ticker] = None
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

    def margin_usage(self) -> tuple[float, float] | None:
        """(maintenance margin used, net liquidation) for the WHOLE account, or None.

        Account-wide on purpose: several strategies share this account, so the margin constraint
        genuinely is shared. `MaintMarginReq` rather than `FullInitMarginReq` because maintenance
        is what an actual liquidation is measured against.

        Returns None on any failure — the caller treats that as "unknown" and logs it, rather
        than assuming healthy.
        """
        try:
            rows = {r.tag: r for r in self.ib.accountSummary()}
            mm = rows.get("MaintMarginReq") or rows.get("FullMaintMarginReq")
            nl = rows.get("NetLiquidation")
            if not mm or not nl:
                return None
            return float(mm.value), float(nl.value)
        except Exception as e:  # noqa: BLE001
            logging.warning("margin_usage failed: %s", e)
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

    def order(self, ticker: str, action: str, shares: int, wait: float = 4.0) -> dict:
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
            self.ib.sleep(wait)                    # let it fill (RTH) or reach PreSubmitted (queued)
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
