"""Volume-confirmed drift — do big moves on big volume CONTINUE, and over what horizon?

Hypothesis: a large price move ON UNUSUALLY HIGH VOLUME is informed/news-driven → it DRIFTS
(continues); a large move on quiet volume is noise. The next-day test (v1) found ~nothing —
but the classic "news drifts" effect (PEAD) plays out over WEEKS. So here we measure cumulative
MARKET-EXCESS continuation over multiple horizons (t+1, t+5, t+10, t+20), signed by the move
direction (>0 = drift, <0 = reversal), for HIGH- vs NORMAL-volume big moves, up vs down.
DRIFT => positive AND growing with horizon for high-volume events. Run: python scripts/volume_drift_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import download_ohlcv, sp500_tickers


def main(start="2011-01-01", end=None, move_z=2.0, horizons=(1, 5, 10, 20)):
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tk = sp500_tickers()
    print(f"loading {len(tk)} S&P500 names + SPY, {start}->{end} …")
    panel = download_ohlcv(sorted(set(tk + ["SPY"])), start, end)
    px = panel["adj_close"].dropna(how="all", axis=1)
    vol = panel["volume"].reindex_like(px)
    spy_px = px["SPY"]
    px = px.drop(columns=["SPY"], errors="ignore")
    vol = vol.drop(columns=["SPY"], errors="ignore")

    rets = px.pct_change(fill_method=None)
    move_sigma = rets.rolling(63).std()
    vol_ratio = vol / vol.rolling(63).mean()
    big = rets.abs() > (move_z * move_sigma)
    sign = np.sign(rets)

    # Market-excess cumulative continuation over each horizon h (close t -> close t+h).
    cont_ex = {}
    for h in horizons:
        stock_fwd = px.shift(-h) / px - 1.0
        spy_fwd = spy_px.shift(-h) / spy_px - 1.0
        cont_ex[h] = sign * stock_fwd.sub(spy_fwd, axis=0)

    events = [("vol > 2.0x (HIGH)", vol_ratio > 2.0),
              ("vol > 3.0x (V.HIGH)", vol_ratio > 3.0),
              ("vol 0.7-1.3x (normal)", (vol_ratio > 0.7) & (vol_ratio < 1.3))]

    print("\n" + "=" * 92)
    print(f"VOLUME-CONFIRMED DRIFT — multi-horizon  (S&P500, move>|{move_z}sigma|; market-EXCESS cont, bps)")
    print("=" * 92)
    hdr = f"  {'event / dir':26s} {'N':>7s}" + "".join(f"{'h='+str(h):>10s}" for h in horizons)
    print(hdr + "\n  " + "-" * (35 + 10 * len(horizons)))
    for vlabel, vmask in events:
        for dlabel, dmask in [("up", rets > 0), ("down", rets < 0)]:
            m = (big & vmask & dmask).values
            n = int(np.nansum(m))
            cells = ""
            for h in horizons:
                v = cont_ex[h].values[m]
                v = v[~np.isnan(v)]
                cells += f"{(v.mean() * 1e4 if len(v) else np.nan):>+10.1f}"
            print(f"  {vlabel + ' / ' + dlabel:26s} {n:>7d}{cells}")
        print()
    print("  Cells = mean cumulative market-excess return in the move's direction, bps, at horizon h days.")
    print("  DRIFT hypothesis => positive AND growing with h for high-vol events (vs normal-vol control).")


if __name__ == "__main__":
    main()
