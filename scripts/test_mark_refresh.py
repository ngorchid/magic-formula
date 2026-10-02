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

print("=" * 70)
print(f"{ran} ran, {'ALL PASS' if not fails else f'{len(fails)} FAIL: ' + ', '.join(fails)}")
if fails:
    raise SystemExit(1)
