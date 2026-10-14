"""Tests for `_ensure_held_prices`: a held name that left the universe keeps a price (2026-10-12).

WHY THIS EXISTS. The monthly pull fetches the CURRENT universe only. A held name that has left its
index was then missing from the price panel, got no mark, and `PortfolioState.nav` skipped it --
valued at ZERO until its clock sold it (up to 21 trading days), and the breaker saw a false loss.
The fix adds price history for missing held names AFTER the ranking, so they get no rank and are
still sold on their clock. scripts/mutate_held_prices.py seeds the faults. No IB, no network.

Run: python scripts/test_held_prices.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import risk_guard as rg  # noqa: E402
import run_paper as runner  # noqa: E402
from paper.state import PortfolioState, Position  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        return f"crashed: {type(e).__name__}: {e}"


IDX = pd.bdate_range(end="2026-10-09", periods=6)


def panels():
    adj = pd.DataFrame({"AAA": [10, 11, 12, 13, 14, 15.0], "BBB": [20, 21, 22, 23, 24, 25.0]},
                       index=IDX)
    return {"adj": adj, "currency": {"AAA": "USD", "BBB": "EUR"}}


def yf_frame(cols: dict, index=IDX, multi=True):
    """Shaped like yf.download: (field, ticker) MultiIndex, or flat columns for one ticker."""
    df = pd.DataFrame(cols, index=index)
    if multi:
        df.columns = pd.MultiIndex.from_product([["Close"], list(cols)])
    else:
        df.columns = ["Close"]
    return df


LOG: dict[str, list[str]] = {"warning": [], "error": []}
runner.logging.warning = lambda f, *a: LOG["warning"].append(f % a if a else f)
runner.logging.error = lambda f, *a: LOG["error"].append(f % a if a else f)
CALLS: list = []


def fake_download(ret):
    def dl(tickers, *a, **k):
        CALLS.append((list(tickers), k))
        if isinstance(ret, Exception):
            raise ret
        return ret
    return dl


def reset():
    LOG["warning"].clear()
    LOG["error"].clear()
    CALLS.clear()


# =============================================================================================
print("A HELD NAME MISSING FROM THE PANEL GETS ITS PRICES")
reset()
p = panels()
before = p["adj"].copy()
xs = [30, 31, 32, 33, 34, 35.0]
runner.yf.download = fake_download(yf_frame({"XXX": xs}))
p = safe(runner._ensure_held_prices, p, {"AAA": "USD", "XXX": "GBp"})
ok = isinstance(p, dict)
check("it runs", ok, str(p))
adj = p["adj"] if ok else pd.DataFrame()
check("XXX is now a column with its prices", "XXX" in adj.columns and list(adj["XXX"]) == xs,
      str(adj.get("XXX")))
check("only the missing name was downloaded, from the panel's first date",
      CALLS and CALLS[0][0] == ["XXX"] and pd.Timestamp(CALLS[0][1].get("start")) == IDX[0], str(CALLS))
check("no rows added, the other columns unchanged",
      ok and adj.index.equals(before.index) and adj[["AAA", "BBB"]].equals(before), "")
check("its currency comes from the position; existing currencies untouched",
      ok and p["currency"].get("XXX") == "GBp" and p["currency"]["BBB"] == "EUR", str(p.get("currency")))
check("a WARNING names it (prices for valuation only)",
      any("XXX" in w and "valuation only" in w for w in LOG["warning"]) and not LOG["error"], str(LOG))

print("\nTHE POSITION IS VALUED, NOT COUNTED AS ZERO")
st = PortfolioState(cash=1000.0, positions=[Position("AAA", 10, 10.0, "2026-10-01"),
                                            Position("XXX", 10, 30.0, "2026-10-01")])
marks = {t: float(adj[t].dropna().iloc[-1]) for t in adj.columns}
check("NAV = cash + 10 x 15 + 10 x 35 = 1,500", abs(st.nav(marks, {"USD": 1.0}) - 1500.0) < 1e-9,
      str(st.nav(marks, {"USD": 1.0})))

print("\nDATES ARE ALIGNED TO THE PANEL")
reset()
extra = IDX.union(pd.DatetimeIndex(["2026-10-05 12:00"]))     # a date the panel does not have
runner.yf.download = fake_download(yf_frame({"XXX": np.arange(len(extra), dtype=float)}, extra))
p = safe(runner._ensure_held_prices, panels(), {"XXX": "USD"})
check("a date only the download has is dropped (rows of other columns would turn NaN)",
      isinstance(p, dict) and p["adj"].index.equals(IDX) and p["adj"]["AAA"].notna().all(), "")

print("\nONE TICKER, FLAT COLUMNS (older yfinance shape)")
reset()
runner.yf.download = fake_download(yf_frame({"XXX": xs}, multi=False))
p = safe(runner._ensure_held_prices, panels(), {"XXX": "USD"})
check("a flat 'Close' column is taken as that ticker",
      isinstance(p, dict) and "XXX" in p["adj"].columns and list(p["adj"]["XXX"]) == xs, str(p)[:200])

print("\nNOTHING MISSING -> NOTHING DOWNLOADED")
reset()
runner.yf.download = fake_download(AssertionError("must not download"))
p = safe(runner._ensure_held_prices, panels(), {"AAA": "USD", "BBB": "EUR"})
check("no download, no log, panel as before",
      isinstance(p, dict) and not CALLS and not LOG["warning"] and not LOG["error"]
      and p["adj"].equals(panels()["adj"]), str(CALLS))

print("\nNO PRICE AFTER ALL -> LOUD, NEVER SILENT")
reset()
runner.yf.download = fake_download(ConnectionError("down"))
p = safe(runner._ensure_held_prices, panels(), {"XXX": "USD"})
check("a failed download: no crash, panel unchanged",
      isinstance(p, dict) and "XXX" not in p["adj"].columns, str(p)[:200])
check("...and an ERROR names it as valued at ZERO", any("XXX" in e and "ZERO" in e for e in LOG["error"]),
      str(LOG))
reset()
runner.yf.download = fake_download(yf_frame({"XXX": [np.nan] * 6}))
p = safe(runner._ensure_held_prices, panels(), {"XXX": "USD"})
check("an all-NaN download (renamed / delisted ticker) is not added, and is an ERROR",
      isinstance(p, dict) and "XXX" not in p["adj"].columns
      and any("XXX" in e for e in LOG["error"]), str(LOG))
reset()
runner.yf.download = fake_download(yf_frame({"YYY": xs}))
p = safe(runner._ensure_held_prices, panels(), {"XXX": "USD", "YYY": "USD"})
check("of two missing names, the one with prices is added and only the other is an ERROR",
      isinstance(p, dict) and "YYY" in p["adj"].columns and "XXX" not in p["adj"].columns
      and any("XXX" in e for e in LOG["error"]) and not any("YYY" in e for e in LOG["error"]), str(LOG))

# =============================================================================================
print("\nWIRING IN main(): AFTER THE RANKING, BEFORE THE DAILY MARKS")
reset()


class Stop(Exception):
    pass


SEEN: dict = {}
tmp = Path(tempfile.mkdtemp())
sf = tmp / "state.json"
PortfolioState(cash=1000.0, positions=[Position("AAA", 10, 10.0, "2026-10-01"),
                                       Position("XXX", 10, 30.0, "2026-10-01", currency="GBp")]).save(sf)
RANK = pd.Series({"AAA": 2.1, "BBB": 1.7})
runner.STATE_FILE = sf
runner.code_version = lambda *a, **k: None
runner.halt_state = lambda root: (rg.HALT_NONE, "")
runner._refresh_ranking = lambda today, cfg: (RANK.copy(), panels())
runner.yf.download = fake_download(yf_frame({"XXX": xs}))


def stop_at_marks(pn, tickers):
    SEEN["adj"], SEEN["tickers"], SEEN["ccy"] = pn["adj"].copy(), set(tickers), dict(pn["currency"])
    raise Stop()


runner._refresh_marks = stop_at_marks
r = safe(runner.main, dry_run=True, force=True)
check("main() reaches the daily mark refresh", "Stop" in str(r), str(r))
check("the held name that left the universe is in the panel by then, with its currency",
      "XXX" in SEEN.get("adj", pd.DataFrame()).columns and SEEN.get("ccy", {}).get("XXX") == "GBp",
      str(SEEN.get("adj", pd.DataFrame()).columns.tolist()))
check("...and it is in the daily refresh set, so it gets today's price",
      "XXX" in SEEN.get("tickers", set()), str(SEEN.get("tickers")))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} held-price checks behaved as expected")
