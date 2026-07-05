"""Portfolio state that persists between daily runs (the strategy's memory).

Stores the live book (holdings, entry price/date/FX, per-name 21-trading-day clock),
realized P&L + closed-trade log, cash, and the inception anchor — everything the
orchestrator, the P&L math and the daily email need but a fresh process wouldn't know.

Multi-currency: entry_price is in the stock's local currency and entry_fx is USD-per-
local at entry, so USD P&L captures both the equity move and the FX move (borne on a
USD book). USD names have entry_fx = 1.0.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

HOLD_DAYS = 21  # trading days ≈ 1 month; each position's clock


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float          # local currency (fill price)
    entry_date: str             # ISO date
    entry_fx: float = 1.0       # USD per 1 unit local ccy, at entry
    currency: str = "USD"
    exchange: str = "SMART"

    def clock_expiry(self, hold_days: int = HOLD_DAYS) -> str:
        """Date the position's ~1-month clock is up (entry + hold_days business days)."""
        d = pd.bdate_range(start=self.entry_date, periods=hold_days + 1)[-1]
        return d.date().isoformat()

    def clock_up(self, today: str, hold_days: int = HOLD_DAYS) -> bool:
        return today >= self.clock_expiry(hold_days)

    def cost_usd(self) -> float:
        return self.shares * self.entry_price * self.entry_fx

    def value_usd(self, price: float, fx: float) -> float:
        return self.shares * price * fx

    def pnl_usd(self, price: float, fx: float) -> float:
        return self.value_usd(price, fx) - self.cost_usd()


@dataclass
class PortfolioState:
    inception_date: str | None = None
    inception_nav: float = 100_000.0
    cash: float = 100_000.0
    positions: list[Position] = field(default_factory=list)
    realized_pnl: float = 0.0
    trade_log: list[dict] = field(default_factory=list)
    nav_history: list[dict] = field(default_factory=list)   # [{date, nav, spy}]

    # ---- persistence ----
    @classmethod
    def load(cls, path: str | Path) -> "PortfolioState":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        d["positions"] = [Position(**x) for x in d.get("positions", [])]
        return cls(**d)

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=2, default=str))

    # ---- queries ----
    @property
    def tickers(self) -> set[str]:
        return {p.ticker for p in self.positions}

    def get(self, ticker: str) -> Position | None:
        return next((p for p in self.positions if p.ticker == ticker), None)

    def clocks_up(self, today: str, hold_days: int = HOLD_DAYS) -> list[Position]:
        """Positions whose ~1-month clock has expired — the ones to re-evaluate today."""
        return [p for p in self.positions if p.clock_up(today, hold_days)]

    # ---- mutations ----
    def ensure_inception(self, today: str) -> None:
        if self.inception_date is None:
            self.inception_date = today

    def open_position(self, pos: Position) -> None:
        self.cash -= pos.cost_usd()
        self.positions.append(pos)

    def close_position(self, ticker: str, exit_price: float, exit_fx: float,
                       exit_date: str, reason: str = "") -> dict | None:
        pos = self.get(ticker)
        if pos is None:
            return None
        proceeds = pos.value_usd(exit_price, exit_fx)
        pnl = proceeds - pos.cost_usd()
        self.cash += proceeds
        self.realized_pnl += pnl
        rec = {"ticker": ticker, "shares": pos.shares, "currency": pos.currency,
               "entry_price": pos.entry_price, "entry_date": pos.entry_date, "entry_fx": pos.entry_fx,
               "exit_price": exit_price, "exit_fx": exit_fx, "exit_date": exit_date,
               "pnl_usd": round(pnl, 2), "reason": reason}
        self.trade_log.append(rec)
        self.positions = [p for p in self.positions if p.ticker != ticker]
        return rec

    # ---- valuation (marks: {ticker: price_local}, fx: {ccy: usd_per_local}) ----
    def unrealized_pnl(self, marks: dict[str, float], fx: dict[str, float]) -> float:
        tot = 0.0
        for p in self.positions:
            px, f = marks.get(p.ticker), fx.get(p.currency, p.entry_fx)
            if px is not None:
                tot += p.pnl_usd(px, f)
        return tot

    def positions_value_usd(self, marks: dict[str, float], fx: dict[str, float]) -> float:
        tot = 0.0
        for p in self.positions:
            px, f = marks.get(p.ticker), fx.get(p.currency, p.entry_fx)
            if px is not None:
                tot += p.value_usd(px, f)
        return tot

    def nav(self, marks: dict[str, float], fx: dict[str, float]) -> float:
        return self.cash + self.positions_value_usd(marks, fx)

    def record_nav(self, date: str, nav: float, spy: float | None = None) -> None:
        """Append/replace today's NAV (and SPY close) for the daily/inception comparison."""
        self.nav_history = [h for h in self.nav_history if h["date"] != date]
        self.nav_history.append({"date": date, "nav": round(nav, 2), "spy": spy})
        self.nav_history.sort(key=lambda h: h["date"])
