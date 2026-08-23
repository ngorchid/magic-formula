"""Lab: does the circuit breaker punish the UNHEDGED book more than the hedged one?

The partial-hedge lab showed hedging costs ~7.5%/yr of return for a Sharpe that is flat,
and that the Sharpe gain is not convertible at IB margin rates. The one remaining
argument for hedging was operational: the h=0 book has a -37% backtested drawdown, and a
circuit breaker that fires there would de-risk near the bottom and turn a recovered
drawdown into a realised loss — a cost the partial-hedge backtest does not contain.

That would be a real, mechanical reason to prefer a shallower book. This tests it.

The breaker levels are VOL-SCALED (risk_guard.BreakerLevels.from_vol, 1.2σ/2.0σ/2.8σ),
which matters here: a lower-vol book gets proportionally lower thresholds, so scaling may
neutralise the whole effect. Whether it does is an empirical question, because drawdown
does not scale linearly with vol.

Simulation is sequential and path-dependent — de-risking slows the recovery, which keeps
you de-risked longer, which is exactly the capitulation trap the breaker can create.
`halt` needs a manual restart in production; here it is modelled as staying flat until
the drawdown recovers past `reduce_only`, which is the OPTIMISTIC assumption (a real halt
would need a human, likely later).

Run: python scripts/breaker_hedge_interaction_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from risk_guard import BreakerLevels, circuit_breaker

CAP = 300_000


def simulate(returns: pd.Series, levels: BreakerLevels) -> tuple[pd.Series, pd.DataFrame]:
    """Apply the breaker sequentially. Returns (realised series, per-day state log)."""
    eq, peak, scale = 1.0, 1.0, 1.0
    halted = False
    out, log = [], []
    for dt, r in returns.items():
        realised = r * scale
        eq *= (1 + realised)
        peak = max(peak, eq)
        out.append(realised)

        level, new_scale, _ = circuit_breaker(eq, peak, levels)
        dd = 1.0 - eq / peak
        if level == "halt":
            halted = True
        if halted:
            # manual restart: stay flat until the drawdown recovers past reduce_only
            if dd < levels.reduce_only:
                halted, scale = False, 1.0
                level = "restart"
            else:
                new_scale = 0.0
        scale = new_scale
        log.append({"date": dt, "dd": dd, "level": level, "scale": scale})
    return pd.Series(out, index=returns.index), pd.DataFrame(log).set_index("date")


def capitulation_test(raw: pd.Series, log: pd.DataFrame, horizon: int = 63) -> tuple[int, float]:
    """After each fresh trigger, what did the UNCONSTRAINED book do over the next 63d?"""
    trig = log["level"].isin(["derisk", "reduce_only", "halt"])
    fresh = trig & ~trig.shift(1).fillna(False).astype(bool)
    fwd = []
    idx = raw.index
    for dt in log.index[fresh]:
        i = idx.get_loc(dt)
        if i + horizon < len(idx):
            fwd.append((1 + raw.iloc[i:i + horizon]).prod() - 1)
    return len(fwd), float(np.mean(fwd)) if fwd else float("nan")


def main() -> None:
    B = pd.read_csv(ROOT / "results" / "partial_hedge" / "books.csv",
                    index_col=0, parse_dates=True)

    print("=" * 112)
    print("CIRCUIT-BREAKER INTERACTION BY HEDGE FRACTION")
    print("=" * 112)
    print(f"  {'book':8s} {'vol':>7s} {'maxDD':>7s} {'DD/vol':>7s} | "
          f"{'derisk':>7s} {'reduce':>7s} {'halt':>7s} | {'days derisk+':>12s} {'triggers':>9s}")

    rows = {}
    for c in B.columns:
        raw = B[c].fillna(0.0)
        vol = raw.std() * np.sqrt(252)
        lv = BreakerLevels.from_vol(vol)
        realised, log = simulate(raw, lv)
        dd_raw = summary_stats(raw)["max_drawdown"]
        constrained = int((log["scale"] < 1.0).sum())
        trig = log["level"].isin(["derisk", "reduce_only", "halt"])
        n_trig = int((trig & ~trig.shift(1).fillna(False).astype(bool)).sum())
        print(f"  {c.replace('book_',''):8s} {vol:>7.2%} {dd_raw:>7.1%} "
              f"{abs(dd_raw)/vol:>6.2f}σ | {lv.derisk:>7.1%} {lv.reduce_only:>7.1%} "
              f"{lv.halt:>7.1%} | {constrained:>12,d} {n_trig:>9d}")
        rows[c] = (raw, realised, log, lv)

    print("\n" + "=" * 112)
    print("COST OF THE BREAKER — unconstrained vs breaker-applied")
    print("=" * 112)
    print(f"  {'book':8s} | {'CAGR raw':>9s} {'CAGR brk':>9s} {'cost':>8s} | "
          f"{'Sh raw':>7s} {'Sh brk':>7s} | {'DD raw':>7s} {'DD brk':>7s} | "
          f"{'final raw':>11s} {'final brk':>11s}")
    for c, (raw, realised, log, lv) in rows.items():
        s0, s1 = summary_stats(raw), summary_stats(realised)
        f0 = CAP * (1 + raw).prod()
        f1 = CAP * (1 + realised).prod()
        print(f"  {c.replace('book_',''):8s} | {s0['ann_return']:>+9.2%} {s1['ann_return']:>+9.2%} "
              f"{s1['ann_return']-s0['ann_return']:>+8.2%} | {s0['sharpe']:>+7.2f} {s1['sharpe']:>+7.2f} | "
              f"{s0['max_drawdown']:>+7.1%} {s1['max_drawdown']:>+7.1%} | {f0:>11,.0f} {f1:>11,.0f}")

    print("\n" + "=" * 112)
    print("CAPITULATION TEST — forward 63d of the UNCONSTRAINED book after each fresh trigger")
    print("  (positive = the breaker de-risked into a recovery, i.e. it fired at the bottom)")
    print("=" * 112)
    for c, (raw, realised, log, lv) in rows.items():
        n, avg = capitulation_test(raw, log)
        verdict = "fired at the bottom" if avg > 0 else "fired before further losses"
        print(f"  {c.replace('book_',''):8s}  {n:2d} triggers, mean forward 63d "
              f"{avg:>+7.2%}   {verdict if n else ''}")

    out = ROOT / "results" / "partial_hedge"
    pd.DataFrame({c.replace("book_", ""): rows[c][1] for c in B.columns}).to_csv(
        out / "books_with_breaker.csv")
    print(f"\n  wrote {out}/books_with_breaker.csv")


if __name__ == "__main__":
    main()
