"""Tests for the magic-formula SIZING path — the code that decides order size.

WHY THIS EXISTS. `_slot_usd`, `_size_shares`, `_normalised_tilts` and `_gross_scalar` had NO
coverage of any kind before 2026-08-14 — they were referenced nowhere outside `orchestrator.py`
— despite three real bugs having been found in them by inspection, each of which reached the
live path. This is also the first strategy going to real money, and sizing errors are the
expensive kind: a guard that fails wrong blocks a trade, a sizer that fails wrong sends one.

⚠ EVERY CASE USES THE REAL `Check`, `PortfolioState` AND `Position`. The risk_guard suite was
found on 2026-08-14 to contain four assertions handed bare `type("C", (), {...})()` objects,
which define no `__bool__` and are therefore ALWAYS truthy — those cases had never been able to
fail. Do not introduce an ad-hoc stand-in here. If a real object is awkward to build, that is
information about the design, not a reason for a fake.

⚠ THIS SUITE IS MUTATION-TESTED. Seeded faults must FAIL it. Run
`python3 scripts/mutate_magic_sizing.py` after changing anything here; a case that survives its
own mutation is decoration.

THE THREE REGRESSIONS, each marked REGRESSION below:
  1. Tilts did not average 1, so gross landed anywhere in 0.92x-1.09x of budget (theoretically
     0.5x-2.0x) and `_gross_scalar` is clipped to <=1 so it can only reduce, never correct an
     overshoot.
  2. The cash cap was applied to `slot_usd` BEFORE the inverse-vol tilt multiplied it, so a
     high-tilt name could overdraw. Live cash reached -$2,370.
  3. A FULL book collapsed `slots_free` to `max(0, 1) = 1`, aiming the entire accumulated-cash
     gap at ONE order (~16% of NAV), which breached the 15% single-order cap and rejected the
     whole ranked list — 362 alerts on 2026-08-13.

Run: python3 scripts/test_magic_sizing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dataclasses import replace
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.orchestrator import (PaperConfig, _gross_scalar,  # noqa: E402
                                _normalised_tilts, _size_shares, _slot_usd)
from paper.state import PortfolioState, Position  # noqa: E402
from risk_guard import Check, RiskLimits, check_order  # noqa: E402

CFG = PaperConfig()
fails, ran = [], 0


def expect(label: str, got, want_ok: bool = True) -> None:
    """`got` MUST define __bool__ (use Check). A bare object is always truthy — see module docs."""
    global ran
    ran += 1
    ok = bool(got)
    if ok != want_ok:
        fails.append(f"{label}: expected {'PASS' if want_ok else 'REJECT'}, got "
                     f"{'PASS' if ok else 'REJECT'} ({getattr(got, 'reason', '')})")
    print(f"  [{'ok ' if ok == want_ok else 'FAIL'}] {label:60} "
          f"-> {'PASS' if ok else 'REJECT'}"
          f"{'' if ok == want_ok else '  | ' + getattr(got, 'reason', '')}")


def close(a: float, b: float, tol: float = 1e-9) -> Check:
    return Check(abs(a - b) <= tol, f"{a!r} vs {b!r} (tol {tol})")


def book(cash: float, holdings: dict[str, tuple[float, float]],
         ccy: dict[str, str] | None = None) -> PortfolioState:
    """A real PortfolioState. holdings = {ticker: (shares, entry_price)}."""
    ccy = ccy or {}
    return PortfolioState(cash=cash, positions=[
        Position(t, sh, px, "2026-08-01", 1.0, ccy.get(t, "USD"))
        for t, (sh, px) in holdings.items()])


# =====================================================================================
print("=" * 92)
print("MAGIC-FORMULA SIZING")
print("=" * 92)
print("\n--- _normalised_tilts: must average EXACTLY 1 for ANY vol cross-section ---")
print("    (REGRESSION 1: when it did not, gross landed off budget and nothing could correct it)")

# The cap and the floor bind asymmetrically, and which one dominates depends entirely on the
# cross-section of whichever names rank top that day. These two are the adversarial extremes.
LOWSKEW = pd.Series([0.01] * 25 + [1.00] * 5, index=[f"L{i}" for i in range(30)])
HIGHSKEW = pd.Series([1.00] * 25 + [0.01] * 5, index=[f"H{i}" for i in range(30)])
RANDOM = pd.Series(np.random.default_rng(0).uniform(0.05, 0.9, 30),
                   index=[f"R{i}" for i in range(30)])

for lab, v in [("low-vol-skewed (2.0 cap binds)", LOWSKEW),
               ("high-vol-skewed (0.5 floor binds)", HIGHSKEW),
               ("random cross-section", RANDOM),
               ("uniform vol", pd.Series([0.2] * 30, index=[f"U{i}" for i in range(30)]))]:
    t = _normalised_tilts(v, CFG)
    expect(f"tilts average exactly 1 — {lab}", close(float(t.mean()), 1.0, 1e-12))
    expect(f"tilts sum to n — {lab}", close(float(t.sum()), float(len(v)), 1e-9))

# ---- tilt-mechanism tests -------------------------------------------------------------
# The LIVE config pins the tilt to 1.0 (`inv_vol_clip=(1.0,1.0)`, equal weight, 2026-08-28).
# The tilt code is retained and reversible, so these tests must exercise it with an EXPLICIT
# wide clip rather than the live default -- otherwise they silently stop testing anything
# the moment the default changes, which is exactly what happened when equal weight went in.
TILT_CFG = replace(CFG, inv_vol_clip=(0.5, 2.0))

# Show the raw overshoot the normalisation removes, so the number is on the record.
_ref = float(np.nanmedian(HIGHSKEW.values))
_raw = np.clip(_ref / HIGHSKEW, *TILT_CFG.inv_vol_clip)
print(f"    raw (un-normalised) mean on the high-skew book = {_raw.mean():.4f} "
      f"-> book would run {_raw.mean():.1%} of budget with no way to correct it")
expect("the adversarial case really does overshoot un-normalised (guard is not vacuous)",
       Check(_raw.mean() > 1.10, f"raw mean {_raw.mean():.4f}"))

# Shape must be preserved: normalising changes the level, not the structure.
t_hs = _normalised_tilts(HIGHSKEW, TILT_CFG)
expect("tilt SHAPE preserved (rank corr with raw = 1)",
       close(float(pd.Series(_raw.values).corr(pd.Series(t_hs.values), method="spearman")), 1.0, 1e-9))
expect("tilts stay within the clip band after rescaling (no name exceeds 2x the mean)",
       Check(float(t_hs.max()) <= 2.0 / float(_raw.mean()) + 1e-9, f"max {t_hs.max():.4f}"))

print("\n--- _normalised_tilts: degenerate inputs must not produce a poisoned size ---")
expect("empty vol -> empty tilts", Check(len(_normalised_tilts(pd.Series(dtype=float), CFG)) == 0))
_alln = _normalised_tilts(pd.Series([np.nan] * 5, index=list("abcde")), CFG)
expect("all-NaN vol -> all tilts 1.0", Check(bool((_alln == 1.0).all()), f"{_alln.tolist()}"))
_zero = _normalised_tilts(pd.Series([0.0] * 5, index=list("abcde")), CFG)
expect("zero-vol (zero median) -> all tilts 1.0", Check(bool((_zero == 1.0).all()), f"{_zero.tolist()}"))
_one = _normalised_tilts(pd.Series([0.2], index=["X"]), CFG)
expect("single name -> tilt exactly 1.0", close(float(_one.iloc[0]), 1.0, 1e-12))
_mixed = _normalised_tilts(pd.Series([0.1, 0.2, np.nan, 0.4], index=list("abcd")), CFG)
expect("NaN in one name -> finite tilts, mean still 1",
       Check(bool(np.isfinite(_mixed.values).all()) and abs(_mixed.mean() - 1.0) < 1e-12,
             f"{_mixed.tolist()}"))
_inf = _normalised_tilts(pd.Series([0.0, 0.2, 0.3, 0.4], index=list("abcd")), CFG)
expect("zero vol in one name -> no inf leaks through",
       Check(bool(np.isfinite(_inf.values).all()), f"{_inf.tolist()}"))

print("\n--- _size_shares: the cash cap must bind AFTER the tilt ---")
print("    (REGRESSION 2: capping slot_usd first let slot x tilt overdraw; cash hit -$2,370)")

TIL = pd.Series({"AAA": 1.70, "BBB": 1.00, "CCC": 0.50})
# slot 1,000 x tilt 1.7 = 1,700 wanted, but only 1,200 of cash exists.
sh = _size_shares("AAA", price_local=10.0, fx=1.0, tilts=TIL, slot_usd=1000.0, cash_usd=1200.0)
expect("REGRESSION: high tilt cannot spend more than cash",
       Check(sh * 10.0 <= 1200.0 + 1e-9, f"{sh} sh x $10 = ${sh*10:,.0f} vs $1,200 cash"))
expect("  ... and it does spend the cash it has (not silently zero)", Check(sh == 120, f"{sh}"))
# The pre-fix arithmetic, stated explicitly so the regression cannot be reintroduced unnoticed.
_buggy = int(min(1000.0, 1200.0) * 1.70 // 10.0)
expect("  ... the OLD cap-before-tilt arithmetic really did overdraw (guard is not vacuous)",
       Check(_buggy * 10.0 > 1200.0, f"old path would buy {_buggy} sh = ${_buggy*10:,.0f}"))

for lab, tilt_v, slot, cash, px, fxr, want in [
        ("tilt 1.0, ample cash", 1.00, 1000.0, 99_999.0, 10.0, 1.0, 100),
        ("tilt 0.5 shrinks the order", 0.50, 1000.0, 99_999.0, 10.0, 1.0, 50),
        ("FX applied to the price", 1.00, 1000.0, 99_999.0, 10.0, 2.0, 50),
        ("rounds DOWN to whole shares", 1.00, 1005.0, 99_999.0, 10.0, 1.0, 100)]:
    got = _size_shares("Z", px, fxr, pd.Series({"Z": tilt_v}), slot, cash)
    expect(f"_size_shares: {lab}", Check(got == want, f"got {got}, want {want}"))

for lab, kw in [("zero cash -> 0 shares", dict(cash_usd=0.0)),
                ("negative cash -> 0 shares, never negative", dict(cash_usd=-5_000.0)),
                ("zero price -> 0 shares (no ZeroDivisionError)", dict(price_local=0.0)),
                ("NaN price -> 0 shares", dict(price_local=float("nan"))),
                ("zero fx -> 0 shares", dict(fx=0.0)),
                ("zero slot -> 0 shares", dict(slot_usd=0.0)),
                ("negative slot -> 0 shares", dict(slot_usd=-1000.0))]:
    base = dict(ticker="Z", price_local=10.0, fx=1.0, tilts=pd.Series({"Z": 1.0}),
                slot_usd=1000.0, cash_usd=50_000.0)
    base.update(kw)
    got = _size_shares(**base)
    expect(f"_size_shares: {lab}", Check(got == 0, f"got {got}"))

for lab, tv in [("NaN tilt -> treated as 1.0", float("nan")),
                ("zero tilt -> treated as 1.0", 0.0),
                ("negative tilt -> treated as 1.0", -1.0)]:
    got = _size_shares("Z", 10.0, 1.0, pd.Series({"Z": tv}), 1000.0, 50_000.0)
    expect(f"_size_shares: {lab}", Check(got == 100, f"got {got}, want 100"))
got = _size_shares("MISSING", 10.0, 1.0, TIL, 1000.0, 50_000.0)
expect("_size_shares: ticker absent from tilts -> defaults to 1.0", Check(got == 100, f"got {got}"))

print("\n--- _slot_usd: a FULL book must not aim the whole gap at one order ---")
print("    (REGRESSION 3: slots_free collapsed to 1, sizing one order at ~16% of NAV)")

# 30 held of top_n 30, and cash has accumulated from sells.
full = book(cash=16_000.0, holdings={f"T{i}": (100.0, 28.0) for i in range(30)})
marks = {f"T{i}": 28.0 for i in range(30)}
nav_full = full.nav(marks, {})
slot_full = _slot_usd(full, marks, {}, CFG, 1.0)
eq_w = nav_full / CFG.top_n
print(f"    NAV ${nav_full:,.0f}, 30/30 held, ${full.cash:,.0f} idle "
      f"-> slot ${slot_full:,.0f} (equal-weight ${eq_w:,.0f})")
expect("REGRESSION: full-book slot capped at 2x equal weight",
       Check(slot_full <= 2.0 * eq_w + 1e-9, f"${slot_full:,.0f} vs 2x EW ${2*eq_w:,.0f}"))
expect("REGRESSION: full-book slot stays under the 15% single-order cap",
       Check(slot_full < 0.15 * nav_full, f"${slot_full:,.0f} = {slot_full/nav_full:.1%} of NAV"))
# The pre-fix arithmetic, so the regression is pinned rather than merely absent today.
_old_slot = max(nav_full * 1.0 - full.positions_value_usd(marks, {}), 0.0) / 1
expect("  ... the OLD slots_free=1 arithmetic really did breach 15% (guard is not vacuous)",
       Check(_old_slot > 0.15 * nav_full, f"old slot ${_old_slot:,.0f} = {_old_slot/nav_full:.1%}"))
# And prove it end-to-end against the actual order guard that produced the 362 alerts.
_lim = RiskLimits.for_equities(nav_full)
_shares_new = _size_shares("T0", 28.0, 1.0, pd.Series({"T0": 1.0}), slot_full, full.cash)
_shares_old = _size_shares("T0", 28.0, 1.0, pd.Series({"T0": 1.0}), _old_slot, full.cash)
expect("REGRESSION end-to-end: the fixed slot PASSES check_order",
       check_order("T0", "BUY", _shares_new, 28.0, 1.0, _lim, gross_notional=0.0))
expect("  ... and the old slot would have been REJECTED by it",
       check_order("T0", "BUY", _shares_old, 28.0, 1.0, _lim, gross_notional=0.0), want_ok=False)

print("\n--- _slot_usd: the ordinary cases ---")
empty = book(cash=100_000.0, holdings={})
expect("empty book -> slot = NAV/top_n",
       close(_slot_usd(empty, {}, {}, CFG, 1.0), 100_000.0 / CFG.top_n, 1e-6))
half = book(cash=50_000.0, holdings={f"T{i}": (100.0, 33.3333333333) for i in range(15)})
m_half = {f"T{i}": 33.3333333333 for i in range(15)}
_nav_h = half.nav(m_half, {})
_inv_h = half.positions_value_usd(m_half, {})
expect("partial book -> remaining gap split across FREE slots",
       close(_slot_usd(half, m_half, {}, CFG, 1.0),
             min((_nav_h - _inv_h) / (CFG.top_n - 15), 2.0 * _nav_h / CFG.top_n), 1e-6))
over = book(cash=0.0, holdings={f"T{i}": (100.0, 50.0) for i in range(20)})
m_over = {f"T{i}": 50.0 for i in range(20)}
expect("over-invested (gross_scalar shrinks the target) -> slot floors at 0, never negative",
       Check(_slot_usd(over, m_over, {}, CFG, 0.5) >= 0.0, ""))
expect("gross_scalar 0 -> slot 0", close(_slot_usd(empty, {}, {}, CFG, 0.0), 0.0, 1e-9))
expect("gross_scalar 0.5 halves the slot",
       close(_slot_usd(empty, {}, {}, CFG, 0.5), 50_000.0 / CFG.top_n, 1e-6))
# The gap is measured against NAV, not cfg.budget: a drawn-down book must not top up to the
# original number with money the account no longer has.
drawn = book(cash=10_000.0, holdings={f"T{i}": (100.0, 20.0) for i in range(10)})
m_drawn = {f"T{i}": 20.0 for i in range(10)}
expect("drawn-down book sizes off NAV, not cfg.budget",
       Check(_slot_usd(drawn, m_drawn, {}, CFG, 1.0) < 100_000.0 / CFG.top_n,
             f"slot ${_slot_usd(drawn, m_drawn, {}, CFG, 1.0):,.0f}"))
expect("slot is never negative for any gross_scalar",
       Check(all(_slot_usd(over, m_over, {}, CFG, g) >= 0.0 for g in (0.0, 0.25, 0.5, 1.0)), ""))

print("\n--- _gross_scalar: clipped to <=1, so it can only ever REDUCE ---")
expect("high vol -> scales down",
       Check(_gross_scalar(pd.Series([0.60] * 30), CFG) < 1.0,
             f"{_gross_scalar(pd.Series([0.60]*30), CFG):.4f}"))
expect("low vol -> clipped at 1.0 (no leverage)",
       close(_gross_scalar(pd.Series([0.05] * 30), CFG), 1.0, 1e-12))
expect("never exceeds 1.0 for any vol",
       Check(all(_gross_scalar(pd.Series([v] * 30), CFG) <= 1.0
                 for v in (0.001, 0.01, 0.05, 0.2, 0.5, 2.0)), ""))
expect("never negative for any vol",
       Check(all(_gross_scalar(pd.Series([v] * 30), CFG) >= 0.0
                 for v in (0.001, 0.01, 0.05, 0.2, 0.5, 2.0)), ""))
expect("empty vol -> 1.0", close(_gross_scalar(pd.Series(dtype=float), CFG), 1.0, 1e-12))
expect("NaN median -> 1.0", close(_gross_scalar(pd.Series([np.nan] * 5), CFG), 1.0, 1e-12))
expect("zero median -> 1.0", close(_gross_scalar(pd.Series([0.0] * 5), CFG), 1.0, 1e-12))

print("\n--- integration: fill a book from cash and check the invariants hold throughout ---")
print("    (REGRESSION 1 end-to-end: gross must land ON budget, never ~2x)")

st = book(cash=100_000.0, holdings={})
tickers = [f"N{i}" for i in range(CFG.top_n)]
vols = pd.Series(np.random.default_rng(7).uniform(0.10, 0.70, CFG.top_n), index=tickers)
tilts = _normalised_tilts(vols, CFG)
px = {t: 40.0 for t in tickers}
g = _gross_scalar(vols, CFG)
min_cash, max_order_frac = float("inf"), 0.0
for t in tickers:
    marks_now = {k: px[k] for k in st.tickers | {t}}
    slot = _slot_usd(st, marks_now, {}, CFG, g)
    n = _size_shares(t, px[t], 1.0, tilts, slot, st.cash)
    if n <= 0:
        continue
    nav_now = st.nav(marks_now, {})
    max_order_frac = max(max_order_frac, n * px[t] / nav_now)
    st.open_position(Position(t, n, px[t], "2026-08-02", 1.0, "USD"))
    min_cash = min(min_cash, st.cash)

final_marks = {t: px[t] for t in st.tickers}
nav_f = st.nav(final_marks, {})
inv_f = st.positions_value_usd(final_marks, {})
print(f"    filled {len(st.positions)} names, NAV ${nav_f:,.0f}, invested ${inv_f:,.0f} "
      f"({inv_f/nav_f:.1%} of NAV), gross_scalar {g:.3f}, min cash ${min_cash:,.0f}")
expect("REGRESSION: gross never exceeds NAV (was reaching ~2x budget)",
       Check(inv_f <= nav_f + 1e-6, f"invested ${inv_f:,.0f} vs NAV ${nav_f:,.0f}"))
expect("REGRESSION: cash never went negative during the fill (was -$2,370)",
       Check(min_cash >= -1e-9, f"min cash ${min_cash:,.2f}"))
expect("gross lands ON target, not far under (>=90% of the vol-scaled aim)",
       Check(inv_f >= 0.90 * nav_f * g, f"{inv_f/(nav_f*g):.1%} of target"))
expect("no single order exceeded the 15% cap during the fill",
       Check(max_order_frac <= 0.15, f"largest order {max_order_frac:.2%} of NAV"))
expect("every order passed the real check_order guard",
       Check(all(check_order(p.ticker, "BUY", p.shares, p.entry_price, 1.0,
                             RiskLimits.for_equities(nav_f), gross_notional=0.0).ok
                 for p in st.positions), ""))

# =====================================================================================
print("\n--- stale prices: excluded from BUYS, never blocking a SELL ---")
print("    (data_fresh sees only the INDEX; one frozen ticker left the panel looking current)")


class _StubBroker:
    """Records orders. A stub is legitimate HERE because the broker is a collaborator, not the
    thing under test — run_daily's stale handling is. Nothing about its behaviour is asserted."""

    dry_run = True

    def __init__(self):
        self.orders = []

    def order(self, ticker, action, shares, wait=20.0):
        self.orders.append((action, ticker, shares))
        return {"ok": True, "status": "dryrun", "fill_price": None}


def _run(px, ranking, state, today="2026-08-17", **cfg_kw):
    brk = _StubBroker()
    panels = {"adj": px, "currency": {c: "USD" for c in px.columns}}
    from paper.orchestrator import run_daily
    # hold_n must be SMALL here: it is the no-trade band, and with the default 45 against a
    # 7-name fixture every name stays in the band and the rotation never sells anything.
    cfg = PaperConfig(max_new_buys_per_day=3, **cfg_kw)
    run_daily(state, ranking, panels, {"USD": 1.0}, brk, cfg, today)
    return brk.orders


_idx = pd.bdate_range(end=pd.Timestamp("2026-08-17"), periods=400)
_rng = np.random.default_rng(5)
_names = [f"N{i}" for i in range(6)]
_px = pd.DataFrame({n: 100 * np.exp(np.cumsum(_rng.standard_normal(len(_idx)) * 0.01))
                    for n in _names}, index=_idx)
_px["FROZEN"] = 50.0                                   # reports every day, never moves
# FROZEN must rank FIRST. The buy loop walks `ranking.index` IN ORDER and stops at
# max_new_buys_per_day, so a frozen name ranked last is never reached — with or without the
# skip — and the case cannot fail. That is exactly how the first version of it survived both
# stale mutations.
_rank = pd.Series(range(len(_px.columns)),
                  index=["FROZEN"] + _names, dtype=float)

_orders = _run(_px, _rank, book(cash=100_000.0, holdings={}))
_bought = {t for a, t, _ in _orders if a == "BUY"}
print(f"    bought: {sorted(_bought)}")
expect("a FROZEN-price name is not bought EVEN WHEN RANKED FIRST",
       Check("FROZEN" not in _bought, f"{sorted(_bought)}"))
expect("  ... while live names still are (not a blanket freeze)",
       Check(len(_bought) > 0, f"{sorted(_bought)}"))

# THE INVARIANT: a stale price must never block a SELL. Hold the frozen name past its clock
# and out of the band, so the rotation wants it gone.
_held = book(cash=10_000.0, holdings={"FROZEN": (10.0, 50.0)})
_held.positions[0].entry_date = "2026-01-01"                     # clock long expired
_rank_out = pd.Series(range(len(_px.columns)), index=_names + ["FROZEN"], dtype=float)
_sells = [o for o in _run(_px, _rank_out, _held, hold_n=3, top_n=3) if o[0] == "SELL"]
print(f"    sells: {_sells}")
expect("INVARIANT: a stale price does NOT block the SELL of that position",
       Check(any(t == "FROZEN" for _, t, _ in _sells), f"{_sells}"))




# ---------------------------------------------------------------------------------------
# FX CASH SWEEP
# ---------------------------------------------------------------------------------------
# IB does not auto-convert, so a European buy leaves a NEGATIVE balance in that currency
# financed at the first-tier debit rate (IBKR UK 2026-08-28: EUR 3.697% ... NOK 5.636%). Left
# alone that is ~1.64% of NAV a year on a half-European $50k book. `plan_fx_sweep` decides what
# to convert; it is a pure function precisely so this policy can be tested without a broker,
# because unlike a mis-report a bug here SPENDS MONEY.
print("\n" + "=" * 92)
print("FX CASH SWEEP")
print("=" * 92)

from paper.orchestrator import plan_fx_sweep  # noqa: E402

_FX = {"EUR": 1.16, "GBP": 1.35, "CHF": 1.24, "SEK": 0.104, "DKK": 0.155,
       "NOK": 0.107, "USD": 1.0}

# The core case: a short balance from financing a European buy is covered.
_p = plan_fx_sweep({"USD": 25_000.0, "EUR": -12_000.0}, _FX)
expect("short EUR balance is swept to zero", close(_p.get("EUR", 0.0), 12_000.0, 1e-6))
expect("USD is never itself swept", Check("USD" not in _p, str(_p)))

# Direction: an IDLE foreign balance (e.g. after a sell) goes back to USD, not further out.
_p = plan_fx_sweep({"GBP": 4_000.0}, _FX)
expect("idle long GBP is sold back to USD", close(_p.get("GBP", 0.0), -4_000.0, 1e-6))

# The threshold is on USD VALUE, not on units — the whole point of currencies like SEK and
# NOK, where 4,000 units is ~$420 and 40,000 units is ~$4,200.
_p = plan_fx_sweep({"SEK": -4_000.0}, _FX)           # ~$416
expect("sub-threshold balance left alone (USD value, not units)", Check("SEK" not in _p, str(_p)))
_p = plan_fx_sweep({"SEK": -40_000.0}, _FX)          # ~$4,160
expect("above-threshold SEK IS swept", close(_p.get("SEK", 0.0), 40_000.0, 1e-6))

# REGRESSION GUARD: threshold must be symmetric, or idle long balances accumulate untouched.
_p = plan_fx_sweep({"NOK": 40_000.0}, _FX)           # ~$4,280 long
expect("threshold applies to LONG balances too", close(_p.get("NOK", 0.0), -40_000.0, 1e-6))

# An unknown/absent rate must SKIP, never sweep a guessed size. Not sweeping costs interest;
# sweeping the wrong size trades real money in the wrong amount.
_p = plan_fx_sweep({"JPY": -900_000.0}, _FX)
expect("unknown-rate currency is skipped, not guessed", Check("JPY" not in _p, str(_p)))
_p = plan_fx_sweep({"EUR": -12_000.0}, {"EUR": float("nan")})
expect("NaN rate is skipped", Check("EUR" not in _p, str(_p)))
_p = plan_fx_sweep({"EUR": -12_000.0}, {"EUR": 0.0})
expect("zero rate is skipped", Check("EUR" not in _p, str(_p)))

# Multi-currency: each is decided on its own USD value, independently.
_p = plan_fx_sweep({"USD": 30_000.0, "EUR": -16_000.0, "GBP": -5_000.0,
                    "CHF": -300.0, "NOK": -1_000.0}, _FX)
expect("EUR swept", close(_p.get("EUR", 0.0), 16_000.0, 1e-6))
expect("GBP swept", close(_p.get("GBP", 0.0), 5_000.0, 1e-6))
expect("tiny CHF (~$372) left alone", Check("CHF" not in _p, str(_p)))
expect("NOK ~$107 left alone", Check("NOK" not in _p, str(_p)))

# Zero and empty are no-ops rather than zero-size orders.
expect("zero balance produces no order", Check("EUR" not in plan_fx_sweep({"EUR": 0.0}, _FX)))
expect("empty balances produce an empty plan", Check(plan_fx_sweep({}, _FX) == {}))

# The threshold is configurable, and raising it must actually suppress a sweep — this is the
# knob to turn if the FX commission is not IB's $2 (LYNX marks it up).
_p = plan_fx_sweep({"EUR": -1_000.0}, _FX, min_usd=500.0)
expect("EUR ~$1,160 swept at the $500 default", close(_p.get("EUR", 0.0), 1_000.0, 1e-6))
_p = plan_fx_sweep({"EUR": -1_000.0}, _FX, min_usd=5_000.0)
expect("same balance suppressed by a raised threshold", Check("EUR" not in _p, str(_p)))

# The plan must net to zero: applying it leaves no residual financing charge.
_bal = {"EUR": -16_000.0, "GBP": 5_000.0}
_p = plan_fx_sweep(_bal, _FX)
_resid = {c: _bal[c] + _p.get(c, 0.0) for c in _bal}
expect("applying the plan leaves every swept balance at zero",
       Check(all(abs(v) < 1e-6 for v in _resid.values()), str(_resid)))

# ORDER DIRECTION. The highest-consequence code in the sweep: IB expresses an FX order in the
# PAIR'S BASE currency, and quotes minor currencies with USD as base (USDCHF) but majors the
# other way (EURUSD). Getting it backwards does not fail loudly — it DOUBLES the exposure it
# was meant to close. Hence a pure function, tested in all four combinations.
from paper.broker import FX_PAIRS, fx_order_spec  # noqa: E402

# ccy-as-base (EURUSD, GBPUSD): quantity is in the foreign currency, direction is natural.
expect("cover short EUR -> BUY EURUSD in EUR",
       Check(fx_order_spec("EUR", 12_000, 1.16) == ("EURUSD", "BUY", 12_000),
             str(fx_order_spec("EUR", 12_000, 1.16))))
expect("sell long EUR -> SELL EURUSD in EUR",
       Check(fx_order_spec("EUR", -12_000, 1.16) == ("EURUSD", "SELL", 12_000),
             str(fx_order_spec("EUR", -12_000, 1.16))))
expect("cover short GBP -> BUY GBPUSD in GBP",
       Check(fx_order_spec("GBP", 5_000, 1.35) == ("GBPUSD", "BUY", 5_000),
             str(fx_order_spec("GBP", 5_000, 1.35))))

# USD-as-base (USDCHF, USDSEK, USDDKK, USDNOK): direction INVERTS and the quantity is USD.
expect("cover short CHF -> SELL USDCHF, sized in USD",
       Check(fx_order_spec("CHF", 4_000, 1.24) == ("USDCHF", "SELL", 4_960),
             str(fx_order_spec("CHF", 4_000, 1.24))))
expect("sell long CHF -> BUY USDCHF, sized in USD",
       Check(fx_order_spec("CHF", -4_000, 1.24) == ("USDCHF", "BUY", 4_960),
             str(fx_order_spec("CHF", -4_000, 1.24))))
expect("cover short SEK -> SELL USDSEK, sized in USD (not SEK units)",
       Check(fx_order_spec("SEK", 40_000, 0.104) == ("USDSEK", "SELL", 4_160),
             str(fx_order_spec("SEK", 40_000, 0.104))))

# REGRESSION: the two conventions must not collapse to the same direction. If a refactor drops
# the ccy_is_base flag, EUR and CHF shorts would both map to BUY and one of them would be wrong.
expect("the two pair conventions give OPPOSITE actions for the same sign",
       Check(fx_order_spec("EUR", 1_000, 1.16)[1] != fx_order_spec("CHF", 1_000, 1.24)[1]))

# An unmapped currency yields a zero quantity, which convert_fx refuses — never a guessed pair.
expect("unmapped currency yields no order", Check(fx_order_spec("JPY", 900_000, 0.0068)[2] == 0))
expect("every FX_PAIRS entry contains USD",
       Check(all("USD" in p for p, _ in FX_PAIRS.values()), str(FX_PAIRS)))
expect("every swept currency in SUFFIX_MAP has an FX pair",
       Check({c for c, _ in __import__("paper.broker", fromlist=["x"]).SUFFIX_MAP.values()}
             - {"USD"} <= set(FX_PAIRS),
             str({c for c, _ in __import__("paper.broker", fromlist=["x"]).SUFFIX_MAP.values()})))



# ---------------------------------------------------------------------------------------
# BROKER-vs-FEED PRICE UNITS
# ---------------------------------------------------------------------------------------
# Sizing divides a USD slot by the yfinance mark; the position is then recorded at the IB
# fill. Those must be denominated identically. The LSE quotes in PENCE while the IB contract
# currency reads GBP, and IB ships ContractDetails.priceMagnifier precisely to reconcile
# execution prices with market data -- but that only makes IB self-consistent, not consistent
# with yfinance. A 100x disagreement makes cost_usd 100x too small, so cash barely moves and
# the sizer keeps buying. Nothing raises.
from paper.orchestrator import price_units_agree  # noqa: E402

expect("identical prices agree", Check(price_units_agree(3344.5, 3344.5)))
expect("ordinary intraday drift still agrees", Check(price_units_agree(3350.0, 3344.5)))
expect("a stale bar within 5x still agrees", Check(price_units_agree(3000.0, 3344.5)))
expect("IB in POUNDS vs mark in PENCE is REJECTED",
       Check(not price_units_agree(33.445, 3344.5), "the 100x LSE trap"))
expect("IB in PENCE vs mark in POUNDS is REJECTED (symmetric)",
       Check(not price_units_agree(3344.5, 33.445)))
# Boundaries: the tolerance must actually bind on both sides.
expect("just inside 5x is accepted", Check(price_units_agree(4.99, 1.0)))
expect("just outside 5x is rejected", Check(not price_units_agree(5.01, 1.0)))
expect("just inside 1/5x is accepted", Check(price_units_agree(1.0, 4.99)))
expect("just outside 1/5x is rejected", Check(not price_units_agree(1.0, 5.01)))
# Missing/degenerate inputs are NOT this guard's job — other guards own them, and returning
# False here would block every name whenever the broker is merely unreachable.
expect("missing broker price passes through", Check(price_units_agree(None, 3344.5)))
expect("missing mark passes through", Check(price_units_agree(3344.5, None)))
expect("zero price passes through", Check(price_units_agree(0.0, 3344.5)))
expect("negative price passes through", Check(price_units_agree(-1.0, 3344.5)))


# INTEGRATION: the guard must actually be WIRED INTO the buy loop. Testing price_units_agree
# as a pure function proves the arithmetic and nothing else — a mutation that stopped the loop
# from ever calling it survived exactly that way. These drive run_daily end to end.
class _UnitBroker(_StubBroker):
    """Stub whose quoted price disagrees with the mark by `factor`, mimicking IB reporting
    LSE fills in pounds while the yfinance mark is in pence."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def price(self, ticker):
        return 100.0 * self.factor        # fixture marks are ~100


def _run_ccy(px, ranking, state, factor, ccy="GBp", today="2026-08-17", **cfg_kw):
    brk = _UnitBroker(factor)
    panels = {"adj": px, "currency": {c: ccy for c in px.columns}}
    from paper.orchestrator import run_daily
    # top_n must be large enough that one order clears the 15% single-order cap: at top_n=3
    # every order is 33% of budget and the guard rejects the whole list, so BOTH arms would
    # emit zero buys and the comparison could not fail. budget is pinned to the fixture cash
    # for the same reason -- the default $100k would make the cap bind on a $50k book.
    cfg = PaperConfig(max_new_buys_per_day=3, budget=50_000.0, **cfg_kw)
    run_daily(state, ranking, panels, {ccy: 1.0, "USD": 1.0}, brk, cfg, today)
    return brk.orders


_empty = book(50_000.0, {})
_buys_ok = [o for o in _run_ccy(_px, _rank, _empty, factor=1.0, hold_n=12, top_n=10)
            if o[0] == "BUY"]
expect("foreign names DO trade when broker and mark agree", Check(len(_buys_ok) > 0,
       f"{_buys_ok}"))

_empty2 = book(50_000.0, {})
_buys_bad = [o for o in _run_ccy(_px, _rank, _empty2, factor=0.01, hold_n=12, top_n=10)
             if o[0] == "BUY"]
expect("NO buy is placed when the broker price is 100x off the mark",
       Check(len(_buys_bad) == 0, f"{_buys_bad}"))

# USD names must be unaffected — the guard is for foreign venues and must not gate the S&P.
_empty3 = book(50_000.0, {})
_buys_usd = [o for o in _run_ccy(_px, _rank, _empty3, factor=0.01, ccy="USD",
                                 hold_n=12, top_n=10) if o[0] == "BUY"]
expect("USD names are NOT gated by the unit guard", Check(len(_buys_usd) > 0, f"{_buys_usd}"))

# INTEGRATION: the RISK guard (check_order/price_sane) — a DIFFERENT guard from the unit check —
# must also be wired into the buy loop for US names. A 2026-08-28 refactor nested it inside the
# `ccy != USD` branch, so US buys (the bulk of the book) reached the broker with no pre-trade
# check and use_risk_guard became dead. The USD unit-guard case above could not catch it: it only
# asserts the UNIT check leaves USD alone. This asserts the SIZE check BITES on a US name — at
# top_n=3 a single order is 33% of budget, over check_order's 15% single-order cap, so a guarded
# USD buy is rejected. factor=1.0 so the unit guard is a no-op and only the risk guard can reject.
_empty4 = book(50_000.0, {})
_buys_capped = [o for o in _run_ccy(_px, _rank, _empty4, factor=1.0, ccy="USD",
                                    hold_n=12, top_n=3) if o[0] == "BUY"]
expect("US buy over the 15% single-order cap is REJECTED by the wired risk guard",
       Check(len(_buys_capped) == 0, f"{_buys_capped}"))
# The mutation-killer: the SAME oversized order DOES reach the broker with the guard OFF, proving
# the rejection above is the guard doing its job (not the sizer quietly shrinking the order) and
# that use_risk_guard is live again rather than dead. Under the nesting bug BOTH arms showed buys.
_empty5 = book(50_000.0, {})
_buys_noguard = [o for o in _run_ccy(_px, _rank, _empty5, factor=1.0, ccy="USD",
                                     hold_n=12, top_n=3, use_risk_guard=False) if o[0] == "BUY"]
expect("...and the SAME US order reaches the broker when use_risk_guard=False",
       Check(len(_buys_noguard) > 0, f"{_buys_noguard}"))

print("\n" + "=" * 92)
if fails:
    print(f"{len(fails)} FAILURE(S) of {ran}:")
    for f in fails:
        print("   " + f)
    sys.exit(1)
print(f"all {ran} sizing checks behaved as expected")
