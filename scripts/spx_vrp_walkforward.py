"""Out-of-sample validation of the SPX put-spread parameters.

The full-sample sweep picked VRP>2pts and 16D/10D as best over 137 trades. That is a 42-cell
grid searched on a sample of ~10 trades a year — precisely the setup that manufactures
optima. This asks whether the choices survive when they are made WITHOUT seeing the test data.

TWO TESTS
  1. Single split — select on 2013-2019, report on 2020-2026. One clean holdout.
  2. Expanding walk-forward — re-select each year on everything before it, trade the next
     year, stitch the out-of-sample trades together. More folds, each thinner.

Reported alongside every selected result:
  - the SAME cell's train Sharpe (so the degradation is visible)
  - the BEST cell on test (the unreachable ceiling — how much was left on the table)
  - the FIXED live config (16D/10D, VRP>2) held constant, i.e. no selection at all
If selection cannot beat the fixed config out-of-sample, the sweep was fitting noise.

The stop is held OFF throughout: it is a separate, theoretically-motivated, monotone result
(see spx_vrp_lab.py) and folding it into the search would just widen the grid.

Run: python scripts/spx_vrp_walkforward.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from spx_vrp_lab import Config, load, run, regime_ratio  # noqa: E402

OUT = ROOT / "results" / "spx_vrp"

VRP_GRID = [-99.0, 0.0, 0.01, 0.02, 0.03, 0.05]
DELTA_GRID = [(0.30, 0.20), (0.25, 0.15), (0.20, 0.10), (0.16, 0.10),
              (0.16, 0.08), (0.12, 0.06), (0.10, 0.05)]
COST = 0.25          # index points per leg per side; the middle of the plausible band
REGIME_THR = 1.00    # the LIVE market-wide gate (VIX/VIX3M < 1.00). Earlier runs omitted it.
BASELINE = (0.02, (0.16, 0.10))


def cell_label(v, d) -> str:
    vs = "none" if v < -1 else f"{v*100:+.0f}pt"
    return f"vrp{vs} {d[0]:.2f}/{d[1]:.2f}"


def net_pnl(t: pd.DataFrame) -> pd.Series:
    """Re-cost the stored P&L to the COST used here (lab stored it at 0.50)."""
    return t.pnl + 4 * 0.50 * 100 - 4 * COST * 100


def sharpe(t: pd.DataFrame) -> float:
    if len(t) < 8:
        return np.nan
    p = net_pnl(t)
    yrs = max((t.exit_date.max() - t.entry_date.min()).days / 365.25, 0.5)
    return p.mean() / p.std() * np.sqrt(len(t) / yrs) if p.std() else np.nan


def build_grid(ch, spot, vrp, ratio) -> dict:
    """Run each cell ONCE over full history; slice by date afterwards. Valid because a
    cell's trade list is fixed — only the date window changes between folds."""
    grid = {}
    for v in VRP_GRID:
        for d in DELTA_GRID:
            cfg = Config(vrp_min=v, short_delta=d[0], long_delta=d[1], stop_mult=0.0,
                         regime_thr=REGIME_THR)
            t = run(cfg, ch, spot, vrp, ratio)
            grid[(v, d)] = t
            print(f"    {cell_label(v,d):22s} {len(t):>4d} trades", flush=True)
    return grid


def slice_(t, lo, hi):
    return t[(t.entry_date >= lo) & (t.entry_date < hi)] if len(t) else t


def main() -> None:
    print("Loading chain...")
    ch, spot, vrp = load()
    ratio = regime_ratio()
    print(f"Running {len(VRP_GRID)*len(DELTA_GRID)} grid cells over full history "
          f"(LIVE spec: regime gate VIX/VIX3M < {REGIME_THR:.2f})...")
    grid = build_grid(ch, spot, vrp, ratio)

    # ---------- 1. single split ----------
    SPLIT = pd.Timestamp("2020-01-01")
    lo, hi = pd.Timestamp("2013-01-01"), pd.Timestamp("2027-01-01")
    tr = {k: slice_(t, lo, SPLIT) for k, t in grid.items()}
    te = {k: slice_(t, SPLIT, hi) for k, t in grid.items()}

    ranked = sorted([(sharpe(v), k) for k, v in tr.items() if not np.isnan(sharpe(v))],
                    reverse=True)
    best_k = ranked[0][1]
    best_te = max(((sharpe(v), k) for k, v in te.items() if not np.isnan(sharpe(v))))

    print("\n" + "=" * 96)
    print(f"1. SINGLE SPLIT — select on 2013-2019 ({sum(len(v) for v in tr.values())//len(tr)} "
          f"trades/cell avg), report on 2020-2026")
    print("=" * 96)
    print(f"  {'':34s} {'train Sh':>9s} {'TEST Sh':>9s} {'test n':>7s} {'test $/tr':>10s}")
    print("  " + "-" * 76)
    for lab, k in (("SELECTED on train", best_k),
                   ("fixed live config (no selection)", BASELINE),
                   ("best on TEST (unreachable)", best_te[1])):
        print(f"  {lab + '  ' + cell_label(*k):34s} {sharpe(tr[k]):>+9.2f} "
              f"{sharpe(te[k]):>+9.2f} {len(te[k]):>7d} {net_pnl(te[k]).mean():>10,.0f}")
    print(f"\n  train ranking (top 5): " + ", ".join(
        f"{cell_label(*k)}={s:+.2f}" for s, k in ranked[:5]))

    # ---------- 2. expanding walk-forward ----------
    print("\n" + "=" * 96)
    print("2. EXPANDING WALK-FORWARD — re-select each year on all prior data, trade next year")
    print("=" * 96)
    print(f"  {'year':6s} {'selected on prior':24s} {'train Sh':>9s} | {'OOS n':>6s} "
          f"{'OOS $/tr':>10s} {'baseline $/tr':>14s}")
    print("  " + "-" * 82)
    oos_sel, oos_base = [], []
    for yr in range(2018, 2027):
        a, b = pd.Timestamp("2013-01-01"), pd.Timestamp(f"{yr}-01-01")
        c = pd.Timestamp(f"{yr+1}-01-01")
        trn = {k: slice_(t, a, b) for k, t in grid.items()}
        tst = {k: slice_(t, b, c) for k, t in grid.items()}
        cand = [(sharpe(v), k) for k, v in trn.items() if not np.isnan(sharpe(v))]
        if not cand:
            continue
        s_tr, k = max(cand)
        sel, base = tst[k], tst[BASELINE]
        if len(sel):
            oos_sel.append(net_pnl(sel))
        if len(base):
            oos_base.append(net_pnl(base))
        print(f"  {yr:<6d} {cell_label(*k):24s} {s_tr:>+9.2f} | {len(sel):>6d} "
              f"{(net_pnl(sel).mean() if len(sel) else np.nan):>10,.0f} "
              f"{(net_pnl(base).mean() if len(base) else np.nan):>14,.0f}")

    sel_all = pd.concat(oos_sel) if oos_sel else pd.Series(dtype=float)
    base_all = pd.concat(oos_base) if oos_base else pd.Series(dtype=float)
    yrs = 2026 - 2018 + 1
    print("  " + "-" * 82)
    for lab, s in (("SELECTED (walk-forward)", sel_all), ("FIXED live config", base_all)):
        if len(s) < 5:
            continue
        shp = s.mean() / s.std() * np.sqrt(len(s) / yrs)
        print(f"  {lab:32s} n={len(s):>4d}  total ${s.sum():>9,.0f}  "
              f"$/trade {s.mean():>7,.0f}  Sharpe {shp:>+5.2f}")

    print(f"\n  NB LIVE spec: market-wide regime gate VIX/VIX3M < {REGIME_THR:.2f} applied at entry.")
    print("  NB cost fixed at 0.25 pts/leg/side (assumed — ohlcv-1d has no quotes).")
    print("  Stop held OFF throughout; it is a separate monotone result.")


if __name__ == "__main__":
    main()
