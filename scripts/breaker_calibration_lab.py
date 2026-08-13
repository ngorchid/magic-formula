"""Would the circuit-breaker thresholds actually have helped?

THE PROBLEM WITH THE NUMBERS. derisk 15% / reduce_only 25% / halt 35% (and the book's
10/18/25) are JUDGMENT CALLS. Unlike the hysteresis band (backtested, robust across four
sub-period cells) or min_iv (bracketed by observed junk vs historical minimum IV), nothing
empirical sits behind them. This script asks the only question that matters: applied to the
actual historical equity curves, would they have HELPED or HURT?

WHY THIS IS NOT OBVIOUS. A drawdown-triggered de-risk is a momentum bet on your own P&L: it
assumes losses persist. If drawdowns MEAN-REVERT — which is the norm for a positive-skew,
trend-following book — the breaker de-risks into the recovery and systematically sells the low.
That is the same mechanism that made the 2x stop harmful in options-vrp, applied at portfolio
level. For a negative-skew, short-vol book the reverse may hold, since losses there do cluster.

So the honest prior is that the breaker COSTS return, and the question is how much, and whether
it buys enough drawdown reduction to be worth it. A breaker is insurance; insurance has a
premium. What we must not accept is paying the premium AND getting a worse drawdown.

WHAT IS MEASURED
  * how often each level fires, and how much time is spent de-risked
  * return / Sharpe / maxDD with the breaker vs without
  * THE CAPITULATION TEST: the forward return over the 21 and 63 days AFTER each trigger. If
    those are strongly POSITIVE, the breaker is selling the bottom and the level is too tight.

Run: python scripts/breaker_calibration_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from backtest import summary_stats  # noqa: E402

LEVELS = {"derisk": (0.15, 0.5), "reduce_only": (0.25, 0.0), "halt": (0.35, 0.0)}
BOOK_LEVELS = {"derisk": (0.10, 0.5), "reduce_only": (0.18, 0.0), "halt": (0.25, 0.0)}


def apply_breaker(ret: pd.Series, levels: dict, lag: int = 1) -> tuple[pd.Series, pd.Series]:
    """(net return with the breaker, exposure scale path).

    Sequential, because the breaker reads the equity curve IT ITSELF produces — de-risking
    shrinks subsequent losses, which changes the drawdown, which changes the next decision.
    Computing the drawdown from the UNBREAKERED curve would flatter the result.

    `lag` = act on yesterday's drawdown: the daily run sees the prior close.
    """
    eq, peak = 1.0, 1.0
    scales, nets = [], []
    scale = 1.0
    hist_dd = 0.0
    for r in ret.values:
        scales.append(scale)
        net = r * scale
        nets.append(net)
        eq *= (1.0 + net)
        peak = max(peak, eq)
        hist_dd = 1.0 - eq / peak
        s = 1.0
        for _, (thr, sc) in sorted(levels.items(), key=lambda kv: -kv[1][0]):
            if hist_dd >= thr:
                s = sc
                break
        scale = s
    return pd.Series(nets, index=ret.index), pd.Series(scales, index=ret.index)


def capitulation(ret: pd.Series, levels: dict) -> pd.DataFrame:
    """Forward returns after each first-touch of a level, on the UNBREAKERED curve.

    Strongly positive forward returns mean the level fires near the bottom, i.e. it is too tight.
    First-touch only: re-triggering every day inside one drawdown would count the same event
    dozens of times and bury the signal.
    """
    eq = (1.0 + ret).cumprod()
    dd = 1.0 - eq / eq.cummax()
    rows = []
    for name, (thr, _) in levels.items():
        over = dd >= thr
        first = over & ~over.shift(1, fill_value=False)
        idx = ret.index[first]
        f21 = [float((1 + ret.loc[d:]).cumprod().iloc[:21].iloc[-1] - 1)
               for d in idx if len(ret.loc[d:]) >= 5]
        f63 = [float((1 + ret.loc[d:]).cumprod().iloc[:63].iloc[-1] - 1)
               for d in idx if len(ret.loc[d:]) >= 5]
        rows.append({"level": name, "thr": thr, "triggers": len(idx),
                     "fwd21_mean": np.mean(f21) if f21 else np.nan,
                     "fwd21_pos": np.mean([x > 0 for x in f21]) if f21 else np.nan,
                     "fwd63_mean": np.mean(f63) if f63 else np.nan,
                     "fwd63_pos": np.mean([x > 0 for x in f63]) if f63 else np.nan})
    return pd.DataFrame(rows)


def report(name: str, ret: pd.Series, levels: dict) -> dict:
    base = summary_stats(ret)
    net, scale = apply_breaker(ret, levels)
    br = summary_stats(net)
    print(f"\n  {name}")
    print(f"    {'':22} {'ann ret':>9} {'vol':>8} {'Sharpe':>8} {'maxDD':>9}")
    print(f"    {'no breaker':22} {base['ann_return']:>+9.2%} {base['ann_vol']:>8.2%} "
          f"{base['sharpe']:>+8.2f} {base['max_drawdown']:>+9.2%}")
    print(f"    {'with breaker':22} {br['ann_return']:>+9.2%} {br['ann_vol']:>8.2%} "
          f"{br['sharpe']:>+8.2f} {br['max_drawdown']:>+9.2%}")
    print(f"    {'delta':22} {br['ann_return']-base['ann_return']:>+9.2%} "
          f"{'':>8} {br['sharpe']-base['sharpe']:>+8.2f} "
          f"{br['max_drawdown']-base['max_drawdown']:>+9.2%}")
    print(f"    time de-risked: {(scale < 1).mean():.1%}   fully out: {(scale <= 0).mean():.1%}")
    cap = capitulation(ret, levels)
    print(f"    CAPITULATION TEST — forward return after each FIRST touch:")
    print(f"      {'level':13} {'thr':>5} {'n':>3} {'fwd 21d':>9} {'pos%':>6} {'fwd 63d':>9} {'pos%':>6}")
    for _, r in cap.iterrows():
        if not r["triggers"]:
            print(f"      {r['level']:13} {r['thr']:>5.0%} {0:>3}   never fired")
            continue
        print(f"      {r['level']:13} {r['thr']:>5.0%} {int(r['triggers']):>3} "
              f"{r['fwd21_mean']:>+9.2%} {r['fwd21_pos']:>6.0%} "
              f"{r['fwd63_mean']:>+9.2%} {r['fwd63_pos']:>6.0%}")
    return {"name": name, "d_sharpe": br["sharpe"] - base["sharpe"],
            "d_ret": br["ann_return"] - base["ann_return"],
            "d_dd": br["max_drawdown"] - base["max_drawdown"]}


def trend_returns() -> pd.Series:
    """Daily net returns of the live trend config, contract-level."""
    from scripts.trend_hysteresis_lab import BASKET, run as _run  # noqa: F401
    from data import download_ohlcv
    px = download_ohlcv([e for e, _ in BASKET], "2011-01-01",
                        pd.Timestamp.today().strftime("%Y-%m-%d"))["adj_close"].dropna(
                            how="all", axis=1).sort_index()
    # reuse the lab's construction but return the series
    import scripts.trend_hysteresis_lab as lab
    rets = px.pct_change(fill_method=None)
    N = px.shape[1]
    sig = sum(np.sign(px / px.shift(lb) - 1.0) for lb in (126, 252)) / 2
    vol = rets.rolling(60).std() * np.sqrt(252)
    floor = vol.expanding(min_periods=252).quantile(0.20)
    vol_used = vol.where(floor.isna(), np.maximum(vol, floor))
    budget = 200_000.0
    held = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for etf, notl in lab.BASKET:
        if etf not in px.columns:
            continue
        expo = (sig[etf] * (budget * 0.10 / np.sqrt(N)) / vol_used[etf]).clip(
            -0.40 * budget, 0.40 * budget)
        ct = lab.contract_path((expo / notl).fillna(0.0), 0.70)
        held[etf] = ct * notl
    return ((held.shift(1) * rets.fillna(0.0)).sum(axis=1) / budget).dropna()


def vrp_returns(budget: float = 100_000.0) -> pd.Series:
    """Daily return series for the SPX VRP sleeve, MARKED DAILY.

    Attributing each trade's P&L to its exit date would hide every intra-trade drawdown — and for
    a 30-45 day short-put hold that IS the drawdown. The lab now emits a daily mark; equity is
    realised-to-date plus the open position's unrealised, over `budget`.
    """
    from scripts.spx_vrp_lab import Config, load, run, regime_ratio
    ch, spot, vrp = load()
    cfg = Config(short_delta=.16, long_delta=.10, vrp_min=.02, stop_mult=0.0,
                 regime_thr=1.00, cost_pts=0.25)
    daily: list = []
    t = run(cfg, ch, spot, vrp, regime_ratio(), daily_out=daily)
    if t.empty:
        return pd.Series(dtype=float)
    d = pd.DataFrame(daily).groupby("date")["unrealized"].last()
    realised = t.set_index("exit_date")["pnl"].groupby(level=0).sum().cumsum()
    idx = sorted(set(d.index) | set(realised.index))
    eq = (realised.reindex(idx).ffill().fillna(0.0)
          + d.reindex(idx).fillna(0.0)) + budget
    return eq.pct_change().dropna()


def magic_returns() -> pd.Series:
    """Daily net returns of the enhanced magic formula backtest."""
    from strategies.magic_formula import MagicFormula, MagicFormulaConfig
    return MagicFormula(MagicFormulaConfig()).run().net_returns.dropna()


def main() -> None:
    print("=" * 96)
    print("CIRCUIT-BREAKER CALIBRATION — would the chosen levels have helped?")
    print("=" * 96)
    print("  A drawdown-triggered de-risk is a MOMENTUM bet on your own P&L: it assumes losses")
    print("  persist. Where drawdowns mean-revert it sells the low. Insurance has a premium; what")
    print("  we must not accept is paying it AND getting a worse drawdown.")

    out = []
    tr = trend_returns()
    print(f"\n  trend: {len(tr)} days, {tr.index[0].date()} -> {tr.index[-1].date()}")
    out.append(report("TREND OVERLAY, per-strategy levels 15/25/35", tr, LEVELS))

    # tighter levels, to see whether the choice is even pivotal
    for lv, lab_ in [({"derisk": (0.10, 0.5), "reduce_only": (0.18, 0.0), "halt": (0.25, 0.0)},
                      "TREND, book-style levels 10/18/25"),
                     ({"derisk": (0.25, 0.5), "reduce_only": (0.35, 0.0), "halt": (0.45, 0.0)},
                      "TREND, loose levels 25/35/45")]:
        out.append(report(lab_, tr, lv))

    # --- the other two sleeves, then the BOOK ---
    series = {"trend": tr}
    for nm, fn in (("magic-formula", magic_returns), ("options-vrp", vrp_returns)):
        try:
            r = fn()
            if len(r) > 60:
                series[nm] = r
                print(f"\n  {nm}: {len(r)} days, {r.index[0].date()} -> {r.index[-1].date()}")
                out.append(report(f"{nm.upper()}, levels 15/25/35", r, LEVELS))
            else:
                print(f"\n  {nm}: too short ({len(r)} days) — skipped")
        except Exception as e:  # noqa: BLE001
            print(f"\n  {nm}: unavailable ({type(e).__name__}: {e})")

    if len(series) > 1:
        # Equal-risk book: scale each sleeve to the same vol before combining, so the book is not
        # dominated by whichever series happens to be most volatile.
        common = None
        for r in series.values():
            common = r.index if common is None else common.intersection(r.index)
        print(f"\n  BOOK: {len(series)} sleeves, {len(common)} overlapping days"
              f" ({common[0].date()} -> {common[-1].date()})" if len(common) else "\n  BOOK: no overlap")
        if len(common) > 60:
            scaled = []
            for nm, r in series.items():
                x = r.reindex(common).fillna(0.0)
                sd = x.std()
                scaled.append(x / sd if sd > 0 else x)
            book = sum(scaled) / len(scaled)
            book = book * (0.10 / (book.std() * np.sqrt(252)))   # 10% vol, comparable to a sleeve
            out.append(report("BOOK (equal-risk), per-strategy levels 15/25/35", book, LEVELS))
            out.append(report("BOOK (equal-risk), BOOK levels 10/18/25", book, BOOK_LEVELS))

    print("\n" + "=" * 96)
    print("SUMMARY — effect of the breaker on each variant")
    print("=" * 96)
    print(f"  {'variant':46} {'d Sharpe':>10} {'d return':>10} {'d maxDD':>10}")
    for r in out:
        print(f"  {r['name']:46} {r['d_sharpe']:>+10.2f} {r['d_ret']:>+10.2%} {r['d_dd']:>+10.2%}")
    print("\n  d maxDD POSITIVE = drawdown improved (less negative). d return negative = the")
    print("  premium paid. Judge the levels on whether that trade is acceptable, and on whether")
    print("  the capitulation test shows them firing into recoveries.")


if __name__ == "__main__":
    main()
