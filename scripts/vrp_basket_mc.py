"""Would the 13-name basket's CAPITAL Sharpe beat the SPX-only 0.21?

THE QUESTION. `spx_vrp_lab.py` validated a PER-TRADE edge on ONE instrument: +$120/trade,
Sharpe +0.43, holding a single spread at a time and in a position only 24% of the time. It models
no budget, no `max_positions`, no sizing — so it says nothing about return on CAPITAL. Marked
daily on a $100k base the same series gives Sharpe 0.21, because capital sits idle.

Running 13 names should raise deployment. Whether it raises the capital SHARPE depends entirely
on how correlated the names are — both in WHEN they fire and in WHAT they pay. This simulates
that, drawing trade outcomes from SPX's real distribution and scheduling them across 13 names.

⚠ ASSUMPTIONS — this is an approximation, and the first one is load-bearing and unverified:

 1. **Every name's trades are distributed like SPX's.** Not verified: we have option history for
    SPX only. Single names have fatter idiosyncratic tails (earnings gaps, company events) that
    SPX structurally cannot have, so the simulated tail is almost certainly TOO KIND. The
    earnings filter removes the scheduled part of that, not the unscheduled part.
 2. **Every name's filter-pass rate equals SPX's 37.7%/day.** Supported weakly: on the one
    market-hours day sampled (2026-08-10) 5 of 14 names passed, i.e. 36%.
 3. **Signal correlation 0.72**, from RV20 co-movement across the basket — how much the names
    tend to pass the filter together.
 4. **P&L correlation 0.47**, from underlying return correlation — how much concurrent positions
    win and lose together.
 5. ~~Uniform SPX-level cost~~ SUPERSEDED. The per-name run uses each name's own cost —
    commission COMPUTED from its credit/contract ($2.60 round trip), spread assigned by
    liquidity class and calibrated to the four names actually quoted as combos on 2026-08-10
    (SPY 0.4%, SBUX 16.4%, CAT 27.4%, PFE 27.0% spread component). Cost varies day to day
    (lognormal, sd 0.35) and the ADAPTIVE GUARD skips a trade above 25% of credit rather than
    taking it at a loss — which is what the live code does, and the difference between "CAT drags
    the book at 28%" and "CAT simply does not trade".

Assumption 1 still biases UPWARD and is now the binding one: single names carry idiosyncratic
tails SPX structurally cannot have, and only 4 of 13 costs are measured.

Run: python scripts/vrp_basket_mc.py
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

POOL = ROOT / "results" / "spx_vrp" / "trade_pool.csv"
BASKET = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "XLV", "PFE", "XOM", "XLE", "SBUX", "MCD",
          "DE", "CAT"]
SECTOR = {"SPY": "index", "QQQ": "index", "IWM": "index", "NVDA": "tech", "AAPL": "tech",
          "XLV": "health", "PFE": "health", "XOM": "energy", "XLE": "energy",
          "SBUX": "consumer", "MCD": "consumer", "DE": "industrial", "CAT": "industrial"}
# blocked pairs at the 0.80 correlation threshold measured on live data
OVERLAP = {"SPY": {"QQQ", "IWM"}, "QQQ": {"SPY"}, "IWM": {"SPY"}, "XLE": {"XOM"}, "XOM": {"XLE"}}

# PER-NAME EXECUTION COST. Total cost = commission + spread, both as a fraction of CREDIT.
#   commission is COMPUTABLE: $2.60 round trip / credit-per-contract.
#   spread is assigned by liquidity CLASS, calibrated to the four names actually quoted as
#   combos during market hours on 2026-08-10: SPY 0.4%, SBUX 16.4%, CAT 27.4%, PFE 27.0%.
# credit/contract from the same day's dry run where available; MEASURED entries are marked.
CREDIT = {           # $ per contract
    "SPY": 440, "QQQ": 300, "IWM": 77, "NVDA": 200, "AAPL": 180, "XLV": 90, "PFE": 12,
    "XOM": 72, "XLE": 14, "SBUX": 73, "MCD": 120, "DE": 242, "CAT": 822,
}
SPREAD = {           # fraction of credit; class-assigned except where measured
    "SPY": 0.004, "QQQ": 0.010, "IWM": 0.020,                      # index ETFs (SPY measured)
    "XLV": 0.060, "XLE": 0.080,                                     # sector ETFs
    "NVDA": 0.080, "AAPL": 0.080,                                   # mega-cap, deepest chains
    "XOM": 0.150, "MCD": 0.160, "DE": 0.200,                        # liquid single names
    "SBUX": 0.164, "CAT": 0.274, "PFE": 0.270,                      # MEASURED
}
MEASURED = {"SPY", "SBUX", "CAT", "PFE"}


def name_cost(nm: str) -> float:
    """Total round-trip cost as a fraction of credit for one name."""
    return 2.60 / CREDIT[nm] + SPREAD[nm]


def simulate(pool: pd.DataFrame, *, n_names: int, budget: float, years: float,
             pass_rate: float, rho_signal: float, rho_pnl: float,
             max_positions: int, risk_per_trade: float, max_per_sector: int,
             apply_constraints: bool, rng: np.random.Generator,
             cost_frac_of_credit: float | None = 0.06,
             per_name_cost: bool = False, cost_guard: float = 0.25,
             cost_sd: float = 0.35) -> dict:
    days = int(years * 252)
    names = BASKET[:n_names]
    # entry signal: common factor + idiosyncratic, thresholded to hit `pass_rate` marginally.
    thr = -np.sqrt(2) * 0.0  # placeholder, set below
    from scipy.stats import norm
    thr = norm.ppf(1 - pass_rate)
    zc = rng.standard_normal(days)            # common SIGNAL factor
    # SEPARATE common factor for P&L. Reusing the signal factor made "the signal fired" imply
    # "a good outcome was drawn" — a look-ahead edge baked in, which showed up as PERFECT
    # correlation scoring a BETTER Sharpe than measured correlation (backwards) and a 0.00%
    # max drawdown. Signal timing and outcome must be independent unless we have evidence they
    # are not, and the SPX backtest gives none: entry VRP was ~equal on winners and losers.
    zp = rng.standard_normal(days)            # common P&L factor
    zi = rng.standard_normal((days, len(names)))
    sig = np.sqrt(rho_signal) * zc[:, None] + np.sqrt(1 - rho_signal) * zi
    fires = sig > thr

    # P&L draws are SCALE-FREE: return-on-risk, not the SPX contract's absolute P&L. The live
    # system sizes contracts so each position's max loss ~= risk_per_trade x budget, so what
    # transfers across names is the RATIO, not the dollar figure. Using absolute SPX P&L instead
    # rejects every draw (SPX risks ~$8,600/contract vs a $3,000 budget) and keeps only atypically
    # small trades — pure selection bias.
    # COST HAIRCUT. The pool already carries SPX-level execution (~6% of credit). Single names
    # measured 20-28% on 2026-08-10, and credit is only ~9.1% of max loss, so each extra point of
    # cost-on-credit costs ~0.091 points of return-on-risk. The pool's mean ROR is +1.10%, so an
    # ~18pp cost increase erases the edge entirely — and the single names sit 14-22pp above SPX.
    credit_frac = 732.0 / 8006.0
    if per_name_cost:
        ror_pool = np.sort((pool["pnl"] / pool["max_loss"]).values)   # haircut applied per trade
    else:
        haircut = max((cost_frac_of_credit or 0.06) - 0.06, 0.0) * credit_frac
        ror_pool = np.sort((pool["pnl"] / pool["max_loss"]).values - haircut)
    hold_pool = pool["hold"].clip(lower=1).values
    pos_risk = risk_per_trade * budget

    held = {}            # name -> (days_left, pnl)
    daily_pnl = np.zeros(days)
    risk_path = np.zeros(days)
    n_trades = 0
    n_skipped = 0
    traded_names: dict[str, int] = {}
    for d in range(days):
        for nm in list(held):
            dl, p = held[nm]
            if dl <= 1:
                daily_pnl[d] += p
                del held[nm]
            else:
                held[nm] = (dl - 1, p)
        risk_path[d] = len(held) * pos_risk
        cand = [names[i] for i in np.where(fires[d])[0] if names[i] not in held]
        rng.shuffle(cand)
        for nm in cand:
            if len(held) >= max_positions:
                break
            if apply_constraints:
                if OVERLAP.get(nm, set()) & set(held):
                    continue
                if sum(1 for h in held if SECTOR[h] == SECTOR[nm]) >= max_per_sector:
                    continue
            # GAUSSIAN COPULA. Correlate in NORMAL space, THEN map to a uniform — blending two
            # UNIFORMS as sqrt(rho)*U1 + sqrt(1-rho)*U2 is not uniform at all: it piles up at the
            # centre and truncates BOTH tails, so the pool's -60%-of-risk losses are never drawn.
            # That produced a Sharpe of +10 and a 0.1% max drawdown before this was fixed.
            x = np.sqrt(rho_pnl) * zp[d] + np.sqrt(1 - rho_pnl) * rng.standard_normal()
            u = float(np.clip(norm.cdf(x), 1e-6, 1 - 1e-6))
            k = min(int(u * len(ror_pool)), len(ror_pool) - 1)
            ror = ror_pool[k]
            if per_name_cost:
                # THE ADAPTIVE GUARD, as the live strategy actually behaves. Cost varies day to
                # day (lognormal around the name's measured level), and a trade is SKIPPED when
                # it exceeds the guard rather than taken at a loss. This is the difference
                # between "CAT drags the book at 28%" and "CAT simply does not trade" — the
                # former was what the uniform-cost run assumed, and it is not what the code does.
                c = float(name_cost(nm) * np.exp(rng.normal(0, cost_sd) - cost_sd ** 2 / 2))
                if c > cost_guard:
                    n_skipped += 1
                    continue
                ror -= max(c - 0.06, 0.0) * credit_frac   # pool already carries ~6% (SPX)
            held[nm] = (int(hold_pool[rng.integers(len(hold_pool))]), ror * pos_risk)
            n_trades += 1
            traded_names[nm] = traded_names.get(nm, 0) + 1
    r = daily_pnl / budget
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    eq = np.cumprod(1 + r)
    dd = float((1 - eq / np.maximum.accumulate(eq)).max())
    return {"names": n_names, "trades_yr": n_trades / years, "ann_return": ann, "ann_vol": vol,
            "sharpe": ann / vol if vol else np.nan, "maxdd": -dd,
            "avg_risk_pct": risk_path.mean() / budget, "max_risk_pct": risk_path.max() / budget,
            "pct_days_flat": float((risk_path == 0).mean()),
            "skipped_yr": n_skipped / years,
            "skip_rate": n_skipped / max(n_skipped + n_trades, 1),
            "traded_names": traded_names}


def main() -> None:
    if not POOL.exists():
        print(f"missing {POOL} — run the extraction in spx_vrp_lab first")
        return
    pool = pd.read_csv(POOL)
    rng = np.random.default_rng(7)
    common = dict(pool=pool, budget=100_000.0, years=40.0, pass_rate=0.377,
                  max_positions=6, risk_per_trade=0.03, max_per_sector=2)

    print("=" * 104)
    print("VRP BASKET MONTE CARLO — does multi-name deployment lift the CAPITAL Sharpe?")
    print("=" * 104)
    print("  SPX-only reference: per-trade Sharpe +0.43, daily mark-to-market capital Sharpe 0.21,")
    print("  in a position 24% of the time.\n")
    print(f"  {'variant':34} {'trades/yr':>10} {'ann ret':>9} {'vol':>7} {'Sharpe':>8} "
          f"{'maxDD':>8} {'avg risk':>9} {'days flat':>10}")
    print("  " + "-" * 100)
    rows = []
    for lab, kw in [
        ("1 name (SPX only, no constraints)", dict(n_names=1, rho_signal=0.72, rho_pnl=0.47,
                                                   apply_constraints=False)),
        ("13 names, measured corr", dict(n_names=13, rho_signal=0.72, rho_pnl=0.47,
                                         apply_constraints=True)),
        ("13 names, no constraints", dict(n_names=13, rho_signal=0.72, rho_pnl=0.47,
                                          apply_constraints=False)),
        ("13 names, INDEPENDENT (upper bnd)", dict(n_names=13, rho_signal=0.0, rho_pnl=0.0,
                                                   apply_constraints=True)),
        ("13 names, PERFECT corr (lower bnd)", dict(n_names=13, rho_signal=0.95, rho_pnl=0.95,
                                                    apply_constraints=True)),
        ("13 names @ 15% cost-of-credit", dict(n_names=13, rho_signal=0.72, rho_pnl=0.47,
                                               apply_constraints=True, cost_frac_of_credit=0.15)),
        ("13 names @ 20% (SBUX measured)", dict(n_names=13, rho_signal=0.72, rho_pnl=0.47,
                                                apply_constraints=True, cost_frac_of_credit=0.20)),
        ("13 names @ 28% (CAT measured)", dict(n_names=13, rho_signal=0.72, rho_pnl=0.47,
                                               apply_constraints=True, cost_frac_of_credit=0.28)),
        ("13 names, PER-NAME cost + guard", dict(n_names=13, rho_signal=0.72, rho_pnl=0.47,
                                                 apply_constraints=True, per_name_cost=True)),
        ("  ... guard OFF (trade anyway)", dict(n_names=13, rho_signal=0.72, rho_pnl=0.47,
                                                apply_constraints=True, per_name_cost=True,
                                                cost_guard=9.99)),
    ]:
        r = simulate(rng=rng, **common, **kw)
        rows.append((lab, r))
        print(f"  {lab:34} {r['trades_yr']:>10.1f} {r['ann_return']:>+9.2%} {r['ann_vol']:>7.2%} "
              f"{r['sharpe']:>+8.2f} {r['maxdd']:>+8.2%} {r['avg_risk_pct']:>8.1%} "
              f"{r['pct_days_flat']:>9.0%}" + (f"  skipped {r['skip_rate']:.0%}"
                                               if r.get('skip_rate') else ""))

    print("\n  ⚠ Assumptions 1 (single names behave like SPX) and 5 (SPX-level costs) both bias")
    print("    these UPWARD. Read the 13-name rows as an upper bound on the improvement.")
    pn = next((r for l, r in rows if "PER-NAME" in l), None)
    if pn and pn.get("traded_names"):
        tot = sum(pn["traded_names"].values())
        print("\n  Which names actually trade once the guard is applied (share of all fills):")
        for nm, c in sorted(pn["traded_names"].items(), key=lambda kv: -kv[1]):
            star = " (measured)" if nm in MEASURED else ""
            print(f"    {nm:6} {c/tot:>6.1%}   cost {name_cost(nm):>6.1%}{star}")
        never = [n for n in BASKET[:13] if n not in pn["traded_names"]]
        if never:
            print(f"    NEVER TRADES: {', '.join(never)}")
    pd.DataFrame([{**{"variant": l}, **{k: v for k, v in r.items() if k != "traded_names"}}
                  for l, r in rows]).to_csv(
        ROOT / "results" / "spx_vrp" / "basket_mc.csv", index=False)


if __name__ == "__main__":
    main()
