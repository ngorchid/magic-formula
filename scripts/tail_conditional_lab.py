"""Tail-conditional correlation: what do the overlays do in the CORE's worst months?

Ordinary correlation averages over all conditions, and correlations rise under stress -- so two
streams can look uncorrelated across a decade and still lose together every time it matters.
The question that decides whether a sleeve belongs in the book is narrower:

    WHEN THE CORE IS HAVING ITS WORST MONTHS, WHAT DOES THE CANDIDATE DO?

`book-assessment` already recorded an answer -- trend -0.37 against the core in bad months, VRP
+0.23 -- and that answer is NOT TRUSTWORTHY FOR VRP. It was computed on the variance-swap proxy
(`book_assess2.vrp_series`), whose losses are UNBOUNDED, while the deployed sleeve trades
defined-risk spreads capped at ~4x the credit. Capping the proxy moved its skew from -4.42 to
-0.18, so it overstates exactly the behaviour this test measures. Since then the real OPRA
backtest exists, so the test can finally run on the traded payoff.

THREE THINGS THIS DOES DIFFERENTLY FROM THE EARLIER PASS:

  real payoff   runs `spx_vrp_lab` at the CURRENT live spec -- stop_mult=0 (the 2x stop was
                removed on evidence) and regime_thr=None (the gate was turned off 2026-08-15,
                REGIME_THR=99). Neither cached file matches that: trades_baseline.parquet
                carries the stop, trade_pool.csv carries the gate.
  daily marks   uses `run(..., daily_out=)` rather than booking P&L on the exit date. Exit-date
                attribution hides intra-trade drawdown, which for a 30-45 day short-put hold is
                most of the drawdown there is -- and it is precisely the tail months that would
                be mis-dated.
  proxy beside  reports the variance-swap proxy alongside the real series, so the size of the
                error in the recorded -0.37/+0.23 figures is visible rather than asserted.

Usage:  python scripts/tail_conditional_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import spx_vrp_lab as L  # noqa: E402
from book_assess2 import vrp_series as proxy_vrp_series  # noqa: E402

TARGET_VOL = 0.10          # every stream scaled to this, so tail means are comparable
COST_PTS = 0.25            # the documented backtest assumption; real SPX combos measure ~0.125


def live_spec_daily() -> pd.Series:
    """Daily P&L of the REAL sleeve at the current live spec, as an honest equity curve.

    equity = realised (booked at exit) + unrealised (marked daily while open); the return
    series is its first difference. Days with no position contribute 0, which is correct --
    the sleeve is genuinely flat about three quarters of the time.
    """
    ch, spot, vrp = L.load()
    cfg = L.Config(stop_mult=0.0, regime_thr=None, cost_pts=COST_PTS)
    daily: list = []
    trades = L.run(cfg, ch, spot, vrp, daily_out=daily)
    print(f"  live spec: {len(trades)} trades, mean ${trades.pnl.mean():+,.0f}/trade, "
          f"total ${trades.pnl.sum():+,.0f}")

    unreal = pd.Series({pd.Timestamp(d["date"]): float(d["unrealized"]) for d in daily}
                       ).sort_index() if daily else pd.Series(dtype=float)
    realised = (trades.assign(exit_date=pd.to_datetime(trades.exit_date))
                .groupby("exit_date").pnl.sum().sort_index())
    idx = spot.index
    cum_real = realised.reindex(idx).fillna(0.0).cumsum()
    eq = cum_real.add(unreal.reindex(idx).fillna(0.0), fill_value=0.0)
    return eq.diff().fillna(0.0)


def scale_to_vol(s: pd.Series, target: float = TARGET_VOL) -> pd.Series:
    v = s.std() * np.sqrt(252)
    return s * (target / v) if v > 0 else s


def tail_block(core: pd.Series, cand: pd.Series, q: float, label: str) -> dict:
    """Stats for `cand` restricted to the worst `q` fraction of CORE months."""
    both = pd.concat([core.rename("core"), cand.rename("cand")], axis=1).dropna()
    if len(both) < 24:
        return {}
    cut = both["core"].quantile(q)
    t = both[both["core"] <= cut]
    return {"label": label, "n": len(t),
            "core_mean": t["core"].mean(), "cand_mean": t["cand"].mean(),
            "corr": t["core"].corr(t["cand"]),
            "hit": float((t["cand"] > 0).mean()),
            "uncond_corr": both["core"].corr(both["cand"])}


def main() -> None:
    print("Running the real OPRA backtest at the CURRENT live spec (no gate, no stop)...")
    real = live_spec_daily()

    mf = pd.read_csv(ROOT / "results" / "best_magic" / "best_sp500_pit_all.csv",
                     index_col=0, parse_dates=True)["net_return"]
    trend = pd.read_csv(ROOT / "results" / "trend_overlay" / "trend_overlay_net.csv",
                        index_col=0, parse_dates=True)["trend"]
    proxy = proxy_vrp_series()

    # Monthly, and every stream vol-matched so the tail MEANS are directly comparable
    streams = {"trend": trend, "vrp (REAL OPRA)": real, "vrp (variance-swap PROXY)": proxy}
    core_m = scale_to_vol(mf).resample("ME").sum()
    cand_m = {k: scale_to_vol(v).resample("ME").sum() for k, v in streams.items()}

    span = pd.concat([core_m] + list(cand_m.values()), axis=1).dropna()
    print(f"  common monthly window: {span.index.min():%Y-%m} -> {span.index.max():%Y-%m} "
          f"({len(span)} months), all streams scaled to {TARGET_VOL:.0%} vol\n")

    print("=" * 94)
    print("TAIL-CONDITIONAL BEHAVIOUR vs the CORE (enhanced magic formula)")
    print("=" * 94)
    for q, lab in ((0.10, "worst DECILE of core months"), (0.25, "worst QUARTILE")):
        print(f"\n  --- {lab} ---")
        print(f"  {'sleeve':28}{'n':>4}{'core mean':>11}{'sleeve mean':>13}"
              f"{'corr in tail':>14}{'uncond corr':>13}{'up months':>11}")
        for name, s in cand_m.items():
            r = tail_block(core_m, s, q, name)
            if not r:
                continue
            print(f"  {name:28}{r['n']:>4}{r['core_mean']:>+10.2%}{r['cand_mean']:>+13.2%}"
                  f"{r['corr']:>+14.2f}{r['uncond_corr']:>+13.2f}{r['hit']:>10.0%}")

    # ---- the decision-relevant consequence: does bolting it on WORSEN the core's drawdown? ----
    print("\n" + "=" * 94)
    print("BOOK LEVEL — core + overlay, at two overlay sizes (core left at its own vol)")
    print("=" * 94)
    core_d = mf.copy()

    def stats(s: pd.Series) -> tuple[float, float, float]:
        s = s.dropna()
        ann = s.mean() * 252
        vol = s.std() * np.sqrt(252)
        cum = (1 + s).cumprod()
        dd = float((cum / cum.cummax() - 1).min())
        return ann, vol, (ann / vol if vol else np.nan), dd

    print(f"  {'book':40}{'ann':>8}{'vol':>8}{'Sharpe':>9}{'maxDD':>9}")
    a, v, sh, dd = stats(core_d)
    print(f"  {'core alone (magic formula)':40}{a:>+8.1%}{v:>8.1%}{sh:>+9.2f}{dd:>+9.1%}")
    for ov_vol in (0.05, 0.10):
        for name, s in streams.items():
            add = scale_to_vol(s, ov_vol).reindex(core_d.index).fillna(0.0)
            a, v, sh, dd = stats(core_d.add(add, fill_value=0.0))
            print(f"  {f'core + {name} @ {ov_vol:.0%} vol':40}"
                  f"{a:>+8.1%}{v:>8.1%}{sh:>+9.2f}{dd:>+9.1%}")

    print("\n" + "=" * 94)
    print("READ")
    print("=" * 94)
    print("  A diversifier must have a NEGATIVE mean or a NEGATIVE correlation in the core's")
    print("  worst months. Positive on both means it loses WITH the core and concentrates the")
    print("  book's risk rather than spreading it. Compare the two VRP rows: the gap between")
    print("  them is the error in the recorded -0.37/+0.23 figures, which came from the proxy.")

    out = ROOT / "results" / "tail_conditional"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([core_m.rename("core")] + [v.rename(k) for k, v in cand_m.items()],
              axis=1).to_csv(out / "monthly_streams.csv")
    print(f"\nWrote {out / 'monthly_streams.csv'}")


if __name__ == "__main__":
    main()
