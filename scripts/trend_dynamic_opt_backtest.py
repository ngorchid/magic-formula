"""Trend overlay: choose whole contracts for the PORTFOLIO, not market by market (2026-10-13).

QUESTION. The live sizer rounds each market's target on its own (with a 0.7-contract hysteresis
band). At the live size ($50k x OVERLAY_MULT 1.5 = $75k of notional) that throws away a lot: a
market whose target is 0.4 contracts is simply not held, and nothing compensates for it.
`trend_live_budget_backtest.py` measured the cost: integer sizing at small budgets lands far below
the continuous reference.

METHOD ("dynamic optimisation", after Carver). Each day take the SAME continuous dollar target the
live sizer computes (`compute_targets(...)["dollar_exposure"]`: vol target, vol floor, post-scale
per-market cap and gross cap all as deployed) and choose the integer contract vector n that
minimises the annualised tracking error to it,

    TE(n) = sqrt( (T - x(n))' Sigma (T - x(n)) ) / budget

where x(n) = n * notional and Sigma is the 252-day covariance of the proxy returns. The search is
greedy: from the current holding (and, separately, from zero) repeatedly apply the single
+/-1-contract move that lowers TE most, until none does; the better of the two ends wins.

BUFFER, not a cost penalty. The book trades to that optimum only if the CURRENT holding's TE is
worse than the optimum's by more than a buffer beta (a fraction of the budget); otherwise it holds.
This is the turnover control. (A first design added k * one-off cost to the annual TE; checked on
the 2026-10-13 targets BEFORE any backtest run, it never builds a small position at all, because
an annual risk figure and a one-off cost are not comparable. Replaced before the first run.)
(First run, 2026-10-13: the greedy search started only from the holding and from zero, and was
capped at 400 one-contract steps, so the $100M reference row could not converge (TE 2.1% where
rounding is immaterial). A third start from the per-market rounded target fixes that; the rule and
settings are unchanged. Both runs are reported.) Every
position respects the live per-market cap (40% of the effective budget) and gross cap (3x). A
market with an unusable target (NaN) is HELD, exactly as the live sizer does.

Both arms run in ONE loop on the same daily `compute_targets` call, so the continuous target, the
proxy returns and the cost model are identical; the only difference is how integers are chosen.
The rounding arm is the live behaviour itself (`contracts`, band 0.7, `held` threaded through).

PRE-REGISTERED, before the first run (the hysteresis band was chosen the same way):
  * main setting beta = 1% of budget (a tenth of the 10% vol target); beta = 0 and 2% are
    reported for sensitivity only, not to pick a winner;
  * the decision row is the LIVE effective budget, $75k;
  * "worth building" = the main setting beats the live rounding on Sharpe in BOTH halves of the
    sample at $75k, and does not worsen max drawdown by more than 2 percentage points overall.
  Nothing is tuned after seeing the result; a different beta would be a new, separately reported test.

Proxy returns, not futures, for the same reason as the live-budget backtest: it isolates the
sizing effect. COST_BP 2bp per side on traded notional, as there.

Usage:  python scripts/trend_dynamic_opt_backtest.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "trend-overlay"))

from trend_live_budget_backtest import COST_BP, WARMUP, load_proxies, stats  # noqa: E402
from trend_overlay.contracts import FUTURES  # noqa: E402
from trend_overlay.execution import TrendPaperConfig, compute_targets  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

BUDGETS = [50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 100_000_000]   # effective notional
LIVE = 75_000
B_MAIN, B_GRID = 0.01, (0.0, 0.01, 0.02)
COV_WINDOW = 252
OUT = ROOT / "results" / "trend_dynamic_opt"


def te_of(n, target, notional, sigma, budget, frozen):
    d = np.where(frozen, 0.0, target - n * notional)
    return float(np.sqrt(max(d @ sigma @ d, 0.0)) / budget)


def greedy(target, notional, held, sigma, budget, cap, gross_cap, frozen, max_iter=400):
    """Integer contracts minimising tracking error under the caps. Arrays aligned by market."""
    def f(n):
        return te_of(n, target, notional, sigma, budget, frozen)

    def ok(n):
        x = np.abs(n * notional)
        return (x <= cap * 1.001).all() and x.sum() <= gross_cap * 1.001

    best_n, best_f = None, np.inf
    # Third start: the target rounded per market. Without it a large book cannot converge
    # within max_iter one-contract steps (found on the first run: the $100M reference row).
    rounded = np.where(frozen, held, np.round(np.divide(target, notional)))
    starts = [held.copy(), np.where(frozen, held, 0), rounded]
    for n in starts:
        n = n.astype(float)
        # clip a start that breaches the caps (only possible if caps changed); frozen stay put
        for i in range(len(n)):
            if not frozen[i]:
                fits = np.floor(cap / notional[i])
                n[i] = np.clip(n[i], -fits, fits)
        cur = f(n)
        for _ in range(max_iter):
            move, move_f = None, cur
            for i in range(len(n)):
                if frozen[i]:
                    continue
                for d in (1.0, -1.0):
                    n[i] += d
                    if ok(n):
                        v = f(n)
                        if v < move_f - 1e-12:
                            move, move_f = (i, d), v
                    n[i] -= d
            if move is None:
                break
            n[move[0]] += move[1]
            cur = move_f
        if cur < best_f:
            best_n, best_f = n.copy(), cur
    return best_n.astype(int)


def run(px: pd.DataFrame, budget: float) -> dict[str, pd.DataFrame]:
    cfg = TrendPaperConfig(budget=budget)
    rets = px.pct_change(fill_method=None)
    proxy_of = {f.market: f.proxy_etf for f in FUTURES}
    dates = px.index[WARMUP:]
    arms = ["round"] + [f"opt_b{b:g}" for b in B_GRID]
    held = {a: {} for a in arms}
    prev = {a: {} for a in arms}
    rows = {a: [] for a in arms}

    for i, t in enumerate(dates[:-1]):
        tgt = compute_targets(px.loc[:t], cfg, held=held["round"] or None)
        mk = list(tgt.index)
        notional = tgt["notional"].to_numpy(float)
        target = tgt["dollar_exposure"].to_numpy(float)
        frozen = ~np.isfinite(target)
        target = np.nan_to_num(target)
        cols = [proxy_of[m] for m in mk]
        window = rets.loc[:t, cols].tail(COV_WINDOW)
        sigma = window.cov(min_periods=60).fillna(0.0).to_numpy() * 252
        cap = cfg.per_market_cap * cfg.budget * cfg.overlay_multiple
        gcap = cfg.gross_cap * cfg.budget * cfg.overlay_multiple

        pos = {"round": {m: int(c) for m, c in tgt["contracts"].items()}}
        for beta in B_GRID:
            a = f"opt_b{beta:g}"
            h = np.array([held[a].get(m, 0) for m in mk], float)
            n = greedy(target, notional, h, sigma, budget, cap, gcap, frozen)
            # trade to the optimum only if holding still is worse by more than the buffer
            if te_of(h, target, notional, sigma, budget, frozen) - \
                    te_of(n, target, notional, sigma, budget, frozen) <= beta:
                n = h
            pos[a] = dict(zip(mk, [int(v) for v in n]))

        nxt = dates[i + 1]
        r_next = rets.loc[nxt] if nxt in rets.index else None
        for a in arms:
            held[a] = pos[a]
            expo = {m: pos[a][m] * float(tgt.at[m, "notional"]) for m in mk}
            pnl = 0.0
            if r_next is not None:
                for m, e in expo.items():
                    r = r_next.get(proxy_of[m], np.nan)
                    if e and np.isfinite(r):
                        pnl += e * r
            turn = sum(abs(expo.get(m, 0.0) - prev[a].get(m, 0.0)) for m in set(expo) | set(prev[a]))
            pnl -= turn * COST_BP / 1e4
            prev[a] = expo
            d = target - np.array([expo[m] for m in mk])
            d[frozen] = 0.0
            rows[a].append({"date": nxt, "ret": pnl / budget, "turn": turn / budget,
                            "n_mkts": sum(1 for m in mk if pos[a][m] != 0),
                            "te": float(np.sqrt(max(d @ sigma @ d, 0.0)) / budget)})
    return {a: pd.DataFrame(rows[a]).set_index("date") for a in arms}


def halves(r: pd.Series) -> tuple[float, float]:
    mid = r.index[len(r) // 2]
    return stats(r[r.index < mid])["sharpe"], stats(r[r.index >= mid])["sharpe"]


def main() -> None:
    px = load_proxies()
    print(f"  proxy panel {px.shape}, {px.index.min():%Y-%m} -> {px.index.max():%Y-%m}")
    print(f"  arms: live rounding (band {TrendPaperConfig().hysteresis_band}) vs greedy integer "
          f"optimiser, buffer beta in {B_GRID} (main {B_MAIN:g}); cost {COST_BP}bp/side\n")
    res = {}
    for b in BUDGETS:
        res[b] = run(px, b)
        print(f"  done ${b:,.0f}")

    hdr = (f"  {'budget':>12} {'arm':>9} {'Sharpe':>7} {'1st half':>9} {'2nd half':>9} {'ann':>7} "
           f"{'vol':>6} {'maxDD':>7} {'markets':>8} {'turn/yr':>8} {'TE':>6}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    summary = []
    for b in BUDGETS:
        for a, d in res[b].items():
            s = stats(d["ret"])
            h1, h2 = halves(d["ret"])
            row = {"budget": b, "arm": a, **s, "h1": h1, "h2": h2, "markets": d["n_mkts"].mean(),
                   "turn": d["turn"].mean() * 252, "te": d["te"].mean()}
            summary.append(row)
            lab = "reference" if b >= 10_000_000 else f"${b:,.0f}"
            print(f"  {lab:>12} {a:>9} {s['sharpe']:>+7.2f} {h1:>+9.2f} {h2:>+9.2f} {s['ann']:>+7.1%} "
                  f"{s['vol']:>6.1%} {s['maxdd']:>+7.1%} {row['markets']:>8.1f} {row['turn']:>8.1f}x "
                  f"{row['te']:>6.1%}")
        print()

    df = pd.DataFrame(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame({f"{a}_{b}": res[b][a]["ret"] for b in BUDGETS for a in res[b]}).to_csv(
        OUT / "daily_returns.csv")

    live = df[df.budget == LIVE].set_index("arm")
    rnd, opt = live.loc["round"], live.loc[f"opt_b{B_MAIN:g}"]
    both = opt.h1 > rnd.h1 and opt.h2 > rnd.h2
    dd_ok = opt.maxdd >= rnd.maxdd - 0.02
    print("=" * len(hdr))
    print(f"PRE-REGISTERED DECISION at ${LIVE:,.0f}, main buffer beta={B_MAIN:g}:")
    print(f"  Sharpe  round {rnd.sharpe:+.2f} -> optimiser {opt.sharpe:+.2f}   "
          f"halves {rnd.h1:+.2f}/{rnd.h2:+.2f} -> {opt.h1:+.2f}/{opt.h2:+.2f}   "
          f"maxDD {rnd.maxdd:+.1%} -> {opt.maxdd:+.1%}")
    print(f"  beats rounding in BOTH halves: {'YES' if both else 'NO'};  "
          f"drawdown within 2pp: {'YES' if dd_ok else 'NO'}  ->  "
          f"{'WORTH BUILDING' if both and dd_ok else 'NOT worth building on this evidence'}")
    print(f"\nWrote {OUT / 'summary.csv'} and daily_returns.csv")


if __name__ == "__main__":
    main()
