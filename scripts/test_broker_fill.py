"""Tests for Broker.order()'s book-on-CONFIRMED-fill behaviour — no IB, no network.

WHY THIS EXISTS. Until 2026-09-07 order() returned ok=True for a merely QUEUED order and the
caller booked it at the mark. A market order into a CLOSED market (US holiday, a European local
holiday, Oslo after its 16:20 close) sits PreSubmitted, was booked, then expired overnight
unfilled — a PHANTOM (state holds a position the broker never bought). This bit TGS.OL on
2026-09-01. The fix: only a FILLED order is booked; anything still open after the poll is
cancelled and reported ok=False, so both the buy and sell paths skip it and retry next run.

Driven through the REAL Broker.order() against a fake IB that scripts the order-status
transitions. Real asserts, non-zero exit on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from paper.broker import Broker  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


class _OrderStatus:
    def __init__(self):
        self.status = "PreSubmitted"
        self.avgFillPrice = 0.0
        self.filled = 0.0


class _Trade:
    def __init__(self):
        self.orderStatus = _OrderStatus()


class _FakeIB:
    """Scripts order-status over successive sleeps.

    `fill_after`  : sleeps after which status flips to Filled (None = never on its own).
    `fill_on_cancel`: if True, the order fills in the race exactly when cancelOrder is called
                      (models a fast market filling just as we give up).
    `partial`     : shares shown as `filled` while never reaching Filled (models a stuck partial).
    """

    def __init__(self, fill_after=None, fill_on_cancel=False, partial=0, fill_px=100.0):
        self.fill_after, self.fill_on_cancel = fill_after, fill_on_cancel
        self.partial, self.fill_px = partial, fill_px
        self.trade = _Trade()
        self.sleeps = 0
        self.cancelled = False

    def placeOrder(self, contract, order):
        return self.trade

    def sleep(self, _s):
        self.sleeps += 1
        if self.fill_after is not None and self.sleeps >= self.fill_after:
            self._fill()
        elif self.partial:
            self.trade.orderStatus.filled = float(self.partial)

    def cancelOrder(self, order):
        self.cancelled = True
        if self.fill_on_cancel:
            self._fill()
        elif self.trade.orderStatus.status not in ("Filled",):
            self.trade.orderStatus.status = "Cancelled"

    def _fill(self):
        self.trade.orderStatus.status = "Filled"
        self.trade.orderStatus.avgFillPrice = self.fill_px
        self.trade.orderStatus.filled = 100.0


def _broker(ib) -> Broker:
    b = Broker.__new__(Broker)
    b.ib, b.dry_run, b._contracts = ib, False, {}
    b.qualify = lambda ticker: object()      # non-None dummy contract
    return b


print("=" * 88)
print("BROKER.order() — book only on a CONFIRMED fill")
print("=" * 88)

# 1. Normal RTH: fills on the first poll -> ok=True with the real fill price.
ib = _FakeIB(fill_after=1, fill_px=123.45)
r = _broker(ib).order("AAPL", "BUY", 10, wait=5)
check("a filled order returns ok=True with the fill price",
      r["ok"] is True and r["status"] == "Filled" and r["fill_price"] == 123.45, str(r))
check("...and it is NOT cancelled", ib.cancelled is False, f"cancelled={ib.cancelled}")

# 2. THE FIX — closed market: never fills, stays PreSubmitted -> cancelled, ok=False, no price.
ib = _FakeIB(fill_after=None)
r = _broker(ib).order("NTAP", "BUY", 10, wait=3)
check("an order that never fills is CANCELLED", ib.cancelled is True, f"cancelled={ib.cancelled}")
check("...and returns ok=False so the caller books nothing",
      r["ok"] is False and r["fill_price"] is None, str(r))

# 3. Cancel race: still open at the poll's end, but fills exactly as we cancel -> honour the fill.
ib = _FakeIB(fill_after=None, fill_on_cancel=True, fill_px=55.5)
r = _broker(ib).order("XYZ", "BUY", 10, wait=3)
check("a fill in the cancel race is honoured (ok=True, real price)",
      r["ok"] is True and r["status"] == "Filled" and r["fill_price"] == 55.5, str(r))

# 4. A dead (Rejected) order is not booked and not 'cancel-raced' into a fill.
ib = _FakeIB(fill_after=None)
ib.trade.orderStatus.status = "Rejected"
r = _broker(ib).order("BAD", "BUY", 10, wait=3)
check("a Rejected order returns ok=False", r["ok"] is False and r["status"] == "Rejected", str(r))
check("...and a dead order is NOT sent a cancel", ib.cancelled is False, f"cancelled={ib.cancelled}")

# 5. A stuck PARTIAL (never reaches Filled) is treated as not-filled: ok=False, nothing booked.
#    The broker holds the partial, so the next run's reconcile surfaces it as an ORPHAN (visible),
#    which is strictly better than the silent phantom booking it replaced.
ib = _FakeIB(fill_after=None, partial=3)
r = _broker(ib).order("THIN", "BUY", 10, wait=3)
check("a stuck partial is not booked (ok=False)", r["ok"] is False, str(r))
check("...and the order is cancelled", ib.cancelled is True, f"cancelled={ib.cancelled}")

# 6. SELL path: an unfilled sell must also be ok=False (position stays intact, retries next run —
#    this is the 'never block a close' rule: we defer the close, we do not book a fake one).
ib = _FakeIB(fill_after=None)
r = _broker(ib).order("CVS", "SELL", 32, wait=3)
check("an unfilled SELL returns ok=False (no false close)", r["ok"] is False, str(r))

# 7. dry_run books at the mark as before (no IB interaction, ok=True, no fill price).
b = Broker.__new__(Broker); b.ib, b.dry_run, b._contracts = None, True, {}
r = b.order("AAPL", "BUY", 10)
check("dry_run stays ok=True with no fill price (books at mark)",
      r["ok"] is True and r["status"] == "dryrun" and r["fill_price"] is None, str(r))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} broker-fill checks behaved as expected")
