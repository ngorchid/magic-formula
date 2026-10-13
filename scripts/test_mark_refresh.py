"""Tests for `_refresh_marks` — the daily live-price splice into the monthly-cached panel.

WHY THIS EXISTS. A mid-month stock split (APH 2:1 on 2026-09-03) made `price_sane` reject EVERY
buy of that name for weeks. The guard was right; the input was not. `_refresh_marks` patched only
TODAY's bar with a fresh split-adjusted close while the prior bar stayed on the frozen monthly
cache (pre-split), so the mark and its own prior sat on different split bases and the comparison
read as a ~48% crash. The fix refreshes the prior bar from the SAME fetch; this suite pins that.

Run: python scripts/test_mark_refresh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_paper as run_paper  # noqa: E402
from risk_guard import RiskLimits, price_sane  # noqa: E402

fails, ran = [], 0


def expect(label: str, cond: bool) -> None:
    global ran
    ran += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        fails.append(label)


def _yf_frame(tickers, closes_by_ticker, end="2026-09-28", periods=5):
    """Shape a frame like yf.download(multiple tickers): MultiIndex (field, ticker) columns."""
    dates = pd.bdate_range(end=end, periods=periods)
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    data = np.column_stack([closes_by_ticker[t] for t in tickers])
    return pd.DataFrame(data, index=dates, columns=cols)


# --- the APH split scenario -------------------------------------------------------------------
# Monthly cache: APH pre-split, last cached bar 2026-09-02 at 157.11 (the split was 09-03).
cache_idx = pd.bdate_range(end="2026-09-02", periods=7)
adj = pd.DataFrame({
    "APH":  [150, 152, 155, 156, 157, 158, 157.11],
    "MSFT": [495, 496, 497, 498, 499, 500, 501.0],
}, index=cache_idx)
panels = {"adj": adj.copy()}

# Fresh 5d fetch is fully split-adjusted (all ~82-84), consistent within itself.
fresh = _yf_frame(["APH", "MSFT"], {
    "APH":  [82.84, 82.19, 83.08, 84.09, 83.16],
    "MSFT": [500, 501, 502, 503, 504.0],
})
run_paper.yf.download = lambda *a, **k: fresh      # monkeypatch the network call

panels = run_paper._refresh_marks(panels, {"APH", "MSFT"})
out = panels["adj"]
lim = RiskLimits.for_equities(50_000)

aph = out["APH"].dropna()
mark, prior = float(aph.iloc[-1]), float(aph.iloc[-2])
expect("today's APH mark is the fresh post-split close", abs(mark - 83.16) < 1e-6)
expect("APH prior bar refreshed to the SAME-basis close (not the cached 157.11)",
       abs(prior - 84.09) < 1e-6)
expect("price_sane now PASSES for the split name (was a phantom 48% reject)",
       price_sane("APH", mark, prior, lim).ok)

# The guard must still do its job: a genuine bad print in today's bar, against the fresh prior,
# is still caught. (Mutation guard: if the fix wrongly copied the mark into the prior, this fails.)
expect("a real bad print in today's bar is still rejected",
       not price_sane("APH", 8.31, prior, lim).ok)

# A non-split name is untouched in spirit: mark current, prior sane, guard passes.
msft = out["MSFT"].dropna()
expect("non-split name marks cleanly and passes",
       price_sane("MSFT", float(msft.iloc[-1]), float(msft.iloc[-2]), lim).ok)

# --- every fetched bar on its own date: no false "stale since the cache date" (2026-10-13) ------
# AAF.L on 2026-10-12: cache rebuilt 2026-10-01; the 16:00 fetch had no 12 Oct bar for London yet
# and Yahoo repeated the 9 Oct close as the newest value. The old splice put that close on TODAY
# and the prior close on the CACHE DATE, so stale_columns read "unchanged since 1 Oct" -> 11d.
from risk_guard import stale_columns  # noqa: E402

cache2 = pd.DataFrame({"AAF.L": [306.6, 311.0, 311.0, 296.5], "FROZ": [50.0, 51.0, 52.0, 53.0],
                       "SPLITX": [100.0, 101.0, 102.0, 103.0]},
                      index=pd.to_datetime(["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01"]))
fetch_dates = pd.to_datetime(["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-12"])
fetch2 = pd.DataFrame(
    np.column_stack([[np.nan, 312.8, 309.6, 290.2, 290.2],        # newest repeats the prior close
                     [60.0, 60.0, 60.0, 60.0, 60.0],              # a feed that really is frozen
                     [51.0, 51.5, 52.0, 52.5, 53.0]]),            # post-split basis (cache pre-split)
    index=fetch_dates, columns=pd.MultiIndex.from_product([["Close"], ["AAF.L", "FROZ", "SPLITX"]]))
run_paper.yf.download = lambda *a, **k: fetch2
out2 = run_paper._refresh_marks({"adj": cache2.copy()}, {"AAF.L", "FROZ", "SPLITX"})["adj"]
stale2, _ = stale_columns(out2, pd.Timestamp("2026-10-12"), lim)
expect("each fetched close lands on its own date (AAF.L 8 Oct = 309.6, 9 Oct = 290.2)",
       out2.loc["2026-10-08", "AAF.L"] == 309.6 and out2.loc["2026-10-09", "AAF.L"] == 290.2)
expect("the cache-date bar is NOT overwritten by a later close (1 Oct stays 296.5)",
       out2.loc["2026-10-01", "AAF.L"] == 296.5)
expect("AAF.L moved on 9 Oct -> NOT stale on 12 Oct (was a false 11d)", "AAF.L" not in stale2)
expect("a feed frozen for a week IS still flagged stale", "FROZ" in stale2)
spl = out2["SPLITX"].dropna()
expect("newest bar and its prior share the fetch's basis (split guard still holds)",
       float(spl.iloc[-2]) == 52.5 and price_sane("SPLITX", float(spl.iloc[-1]), float(spl.iloc[-2]), lim).ok)
expect("a name with no bar yet today keeps its last close as the mark",
       float(out2["AAF.L"].dropna().iloc[-1]) == 290.2)
expect("other columns gain no values on the new dates", out2.loc[fetch_dates].notna().sum().sum()
       == fetch2.notna().sum().sum())

# A fetch that OVERLAPS the cache (a run the day after a rebuild) with a missing bar, and a
# duplicated date (yfinance can repeat the live row): a gap must not wipe a cached close, and the
# duplicate must neither crash nor create a second row.
fetch3 = pd.DataFrame({("Close", "AAF.L"): [np.nan, 300.0, 301.0, 302.0]},
                      index=pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-05"]))
fetch3.columns = pd.MultiIndex.from_tuples(fetch3.columns)
run_paper.yf.download = lambda *a, **k: fetch3
try:
    out3 = run_paper._refresh_marks({"adj": cache2[["AAF.L"]].copy()}, {"AAF.L"})["adj"]
except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here
    out3 = None
    print(f"  (crashed: {type(e).__name__}: {e})")
expect("a missing bar in the fetch does not wipe the cached close (1 Oct stays 296.5)",
       out3 is not None and out3.loc["2026-10-01", "AAF.L"] == 296.5)
expect("a duplicated fetch date gives ONE row holding the later value",
       out3 is not None and out3.index.is_unique and out3.loc["2026-10-05", "AAF.L"] == 302.0)

# --- split back-adjustment of the cached panel (vol / sizing de-distortion) -------------------
# Build an ~80-bar series on a clean post-split basis, then make the pre-split half 2x larger to
# mimic the frozen monthly cache that was never re-adjusted. _repair_splits must recover the
# continuous path once yfinance confirms the split.
dates = pd.bdate_range(end="2026-10-01", periods=80)
true = pd.Series(np.linspace(80.0, 84.0, 80), index=dates)   # smooth, no real jumps
ex = dates[40]
cached = true.copy()
cached.loc[cached.index < ex] *= 2.0                          # pre-split bars un-adjusted (2x)


class _FakeTicker:
    """Stand-in for yf.Ticker; .splits returns an authoritative 2:1 on the ex-date."""
    def __init__(self, *a, **k):
        pass

    @property
    def splits(self):
        return pd.Series({pd.Timestamp(ex, tz="America/New_York"): 2.0})


class _NoSplitTicker:
    def __init__(self, *a, **k):
        pass

    @property
    def splits(self):
        return pd.Series(dtype=float)


# BEFORE repair, the step is a ~-50% daily return that would wreck the 63-day vol.
pre_rets = cached.pct_change().dropna()
expect("setup: the un-adjusted cache has a split-sized daily return", pre_rets.abs().max() > 0.4)

run_paper.yf.Ticker = _FakeTicker
p = run_paper._repair_splits({"adj": pd.DataFrame({"APH": cached})}, {"APH"})
got = p["adj"]["APH"]
expect("confirmed split: pre-split bars divided by 2 -> continuous with the true path",
       np.allclose(got.values, true.values, atol=1e-6))
expect("confirmed split: no split-sized return remains (vol no longer distorted)",
       got.pct_change().dropna().abs().max() < 0.1)

# A genuine crash (same step, but yfinance reports NO split) must be left untouched.
run_paper.yf.Ticker = _NoSplitTicker
p2 = run_paper._repair_splits({"adj": pd.DataFrame({"XYZ": cached.copy()})}, {"XYZ"})
expect("unconfirmed step (real crash) is NOT rescaled",
       p2["adj"]["XYZ"].pct_change().dropna().abs().max() > 0.4)

# Idempotent: re-running on an already-continuous series changes nothing and makes no yf call.
run_paper.yf.Ticker = _FakeTicker
p3 = run_paper._repair_splits({"adj": pd.DataFrame({"APH": true.copy()})}, {"APH"})
expect("idempotent: a continuous series is left unchanged",
       np.allclose(p3["adj"]["APH"].values, true.values, atol=1e-9))

print("=" * 70)
print(f"{ran} ran, {'ALL PASS' if not fails else f'{len(fails)} FAIL: ' + ', '.join(fails)}")
if fails:
    raise SystemExit(1)
