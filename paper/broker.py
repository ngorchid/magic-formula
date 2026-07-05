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

    def net_liq_usd(self) -> float | None:
        try:
            for row in self.ib.accountSummary():
                if row.tag == "NetLiquidation" and row.currency in ("USD", "BASE"):
                    return float(row.value)
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
    def order(self, ticker: str, action: str, shares: int) -> bool:
        """Place a market order (BUY/SELL). Honours dry_run (logs, places nothing)."""
        if shares <= 0:
            return False
        if self.dry_run:
            logging.info("[DRY RUN] %s %d %s", action, shares, ticker)
            return True
        from ib_insync import MarketOrder
        c = self.qualify(ticker)
        if c is None:
            logging.warning("cannot order %s — contract unresolved", ticker)
            return False
        try:
            self.ib.placeOrder(c, MarketOrder(action, shares))
            self.ib.sleep(1)
            logging.info("%s %d %s placed", action, shares, ticker)
            return True
        except Exception as e:  # noqa: BLE001
            logging.error("order failed %s %s %d: %s", action, ticker, shares, e)
            return False
