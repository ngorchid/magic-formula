"""What would it actually COST to cap the magic-formula book's loss with index puts?

THE QUESTION. Appendix D records that beta hedging was rejected -- static, price-timed and
vol-timed -- but every one of those hedges was a SHORT FUTURES position, which is symmetric:
it removes the left tail and the right tail together, and that symmetry is most of why it lost.
A protective PUT is the asymmetric alternative, and it had never been costed. Given the
measured beta of 1.10, what does it cost per year to cap losses at 5%, 10%, 20%?

REAL QUOTES, NOT BLACK-SCHOLES. Prices come from the OPRA SPX daily bars already in the repo
(2013-04 to 2026-08, the same source behind the options-vrp backtest). Pricing a hedge off an
assumed implied vol would beg the question entirely: the whole cost of a put IS the implied
vol, and the variance risk premium means the market charges materially more than realised vol
would imply. That premium is exactly what a hedger pays and a model would omit.

METHOD. Roll a long SPX put continuously. At each roll, target strike is set from the BETA, not
the loss directly: a book loss of X% corresponds to a market fall of X/beta, so
K = S x (1 - X/1.10). Hedge notional is beta x NAV = 1.10x, since that is the market exposure
being covered. Cost and payoff are both applied to the book's real daily series
(results/best_magic/best_sp500_pit_all.csv, the authoritative PIT backtest).

⚠ THE HONEST LIMIT, WHICH THIS MEASURES RATHER THAN ASSERTS. An index put hedges only the
BETA component. The book carries 9.14% residual vol that no SPX put touches, so "cap the loss
at X%" is not deliverable by this instrument at all -- it caps the market-driven part. The
script therefore reports the drawdown the hedged book ACTUALLY realises against the nominal
cap, which is the number that matters.

Run: python3 scripts/put_hedge_cost_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats  # noqa: E402

OPRA = Path("/Users/greiner/aktien/trading/data/opra")
BOOK = ROOT / "results" / "best_magic" / "best_sp500_pit_all.csv"
BETA = 1.10           # doc s7 "Beta decomposition", deployed factor set (Graham off)
CAPS = (0.05, 0.10, 0.20, 0.30)
HORIZONS = {"quarterly": 91, "annual": 365}


def load_puts() -> pd.DataFrame:
    b = pd.read_parquet(OPRA / "bars.parquet",
                        columns=["date", "close", "volume", "expiry", "cp", "strike", "dte"])
    b = b[(b.cp == "P") & (b.volume > 0) & (b.close > 0)]
    return b


def spx_spot(dates: pd.DatetimeIndex) -> pd.Series:
    import yfinance as yf
    px = yf.download("^GSPC", start="2012-01-01", end="2026-09-01",
                     auto_adjust=False, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.reindex(dates).ffill()


def run_hedge(puts: pd.DataFrame, spot: pd.Series, book: pd.Series,
              cap: float, horizon: int, anchor: str = "reset",
              marks: pd.Series | None = None) -> dict:
    """Roll a long put, MARKED TO MARKET DAILY. Returns cost, payoff and the hedged series.

    TWO DESIGN CHOICES THAT DOMINATE THE ANSWER, both found by questioning rather than by the
    first implementation.

    1. `anchor` -- what the strike is measured FROM at each roll.
       "reset" prices K off the spot ON THE ROLL DATE, so after an 8% fall the next put is
       struck 9.1% below the NEW, lower level: the floor ratchets down and the book keeps
       eating up to a cap's worth per quarter. That is a rolling one-quarter beta hedge, and
       it is NOT what "limit losses to X%" means.
       "hwm" prices K off the running PEAK, the literal reading of a loss cap. It keeps
       buying puts struck ABOVE spot after a fall, i.e. IN THE MONEY -- which is not insurance
       at all but a partly directional short: you pay intrinsic and lose it if the market
       recovers. A strawman for the idea, kept only for comparison.
       "hwm_atm" is the repaired version and the one to judge the concept on: anchored to the
       peak but NEVER in the money, `K = min(peak x (1 - X/beta), spot)` with a hard filter to
       strikes at or below spot. If a 90 put has already paid out with spot at 80, an 80 put
       captures any further fall just as well and costs far less. Measured at 0% ITM against
       11-17% for raw "hwm".

    2. MARK TO MARKET. An earlier version charged the whole premium on the roll date and
       credited the payoff on the expiry date, with NOTHING in between -- so a put contributed
       no cushion DURING a crash, only a jump afterwards. That is not how the position behaves
       and it corrupted every path statistic: it reported the 10%-cap drawdown as -28.8% when
       the correct figure is -21.0%, and Sharpe as falling at every cap when in fact it is
       roughly unchanged. Cash TOTALS (cost, payoff, net) were right throughout; only the PATH
       was wrong, which is exactly the half that maxDD and Sharpe are computed from.
       The contract is now followed by its own daily close.
       ⚠ Sparse contracts are forward-filled, so a put that stops printing mid-crash carries a
       stale mark. That biases AGAINST the hedge, so the true result is if anything better.
    """
    mkt_fall = cap / BETA
    cal = book.index
    hedge = pd.Series(0.0, index=cal)
    traded = {d: g for d, g in puts.groupby("date", sort=False)}
    rolls, costs, pays = 0, [], []
    peak = float(spot.iloc[0])
    tol = max(30, horizon * 0.15)
    i = 0
    while i < len(cal):
        t = cal[i]
        s0 = spot.get(t, np.nan)
        if np.isfinite(s0):
            peak = max(peak, float(s0))
        day = traded.get(t)
        if not np.isfinite(s0) or day is None or day.empty:
            i += 1
            continue
        err = (day.dte - horizon).abs()
        if err.min() > tol:
            i += 1
            continue
        exp_pick = day.loc[err.idxmin(), "expiry"]
        chain = day[day.expiry == exp_pick]
        if anchor == "reset":
            target, cand = s0 * (1.0 - mkt_fall), chain
        elif anchor == "hwm":
            target, cand = peak * (1.0 - mkt_fall), chain
        else:                                   # hwm_atm — peak-anchored but never ITM
            target = min(peak * (1.0 - mkt_fall), s0)
            cand = chain[chain.strike <= s0]
            if cand.empty:
                cand = chain
        pick = cand.loc[(cand.strike - target).abs().idxmin()]
        prem, K, exp = float(pick.close), float(pick.strike), pd.Timestamp(pick.expiry)
        j = min(int(cal.searchsorted(exp)), len(cal) - 1)
        settle = max(0.0, K - float(spot.iloc[j]))
        win = cal[i:j + 1]
        v = pd.Series(prem, index=win, dtype=float)
        if marks is not None:
            try:
                v = marks.loc[(exp_pick, float(pick.strike))].reindex(win).ffill()
            except KeyError:
                v = pd.Series(prem, index=win, dtype=float)
            v = v.ffill().fillna(prem)
        v.iloc[0] = prem
        v.iloc[-1] = settle
        hedge.loc[win] += v.diff().fillna(0.0) / s0 * BETA
        costs.append(prem / s0 * BETA)
        pays.append(settle / s0 * BETA)
        rolls += 1
        i = j if j > i else i + 1
    years = (cal[-1] - cal[0]).days / 365.25
    return {"cap": cap, "horizon": horizon, "anchor": anchor, "rolls": rolls,
            "cost_yr": float(np.sum(costs) / years),
            "payoff_yr": float(np.sum(pays) / years),
            "net_yr": float((np.sum(pays) - np.sum(costs)) / years),
            "hedged": book + hedge}


def cohort_study(puts: pd.DataFrame, spot: pd.Series, book: pd.Series, cap: float,
                 horizons=(63, 91, 182, 365, 730), step: int = 42) -> pd.DataFrame:
    """Does a LONGER-dated put cost less per year? Measured across overlapping cohorts.

    Premium scales roughly as sqrt(T), so four 3-month puts should cost about twice one
    12-month put -- less theta per year of protection. That is a claim about COST, which is
    observed at purchase and measured precisely.

    PAYOFF is a different matter and is why the single-chain result was misleading. It is
    path-dependent and realised only at expiry: a market that falls 30% mid-year and recovers
    pays a quarterly holder and pays an annual holder NOTHING. With a 13.4-year sample a single
    2-year chain gives ~7 observations, so its payoff is a timing lottery, not an estimate.
    Starting a fresh chain every `step` trading days and averaging across the resulting
    overlapping cohorts removes that dependence on one arbitrary start date. Cohorts overlap,
    so the SPREAD across them describes start-date sensitivity, not independent samples.
    """
    by_date = {d: g for d, g in puts.groupby("date", sort=False)}
    cal = book.index
    rows = []
    for h in horizons:
        tol = max(30, h * 0.15)
        for start in range(0, len(cal) - 252, step):
            costs, payoffs, i, nroll = [], [], start, 0
            while i < len(cal):
                t = cal[i]
                s0 = spot.get(t, np.nan)
                day = by_date.get(t)
                if not np.isfinite(s0) or day is None or day.empty:
                    i += 1
                    continue
                err = (day.dte - h).abs()
                if err.min() > tol:
                    i += 1
                    continue
                exp_pick = day.loc[err.idxmin(), "expiry"]
                chain = day[day.expiry == exp_pick]
                pick = chain.loc[(chain.strike - s0 * (1 - cap / BETA)).abs().idxmin()]
                prem, K, exp = float(pick.close), float(pick.strike), pd.Timestamp(pick.expiry)
                costs.append(prem / s0 * BETA)
                j = min(int(cal.searchsorted(exp)), len(cal) - 1)
                s_t = spot.iloc[j]
                if np.isfinite(s_t):
                    payoffs.append(max(0.0, K - s_t) / s0 * BETA)
                nroll += 1
                i = j if j > i else i + 1
            if nroll < 2:
                continue
            yrs = (cal[-1] - cal[start]).days / 365.25
            rows.append({"horizon": h, "start": cal[start], "rolls": nroll,
                         "cost_yr": sum(costs) / yrs, "payoff_yr": sum(payoffs) / yrs,
                         "net_yr": (sum(payoffs) - sum(costs)) / yrs,
                         "cost_per_roll": float(np.mean(costs))})
    return pd.DataFrame(rows)


def main() -> None:
    book = pd.read_csv(BOOK, index_col=0, parse_dates=True)["net_return"]
    book = book.loc["2013-04-01":]                 # OPRA coverage starts here
    book = book[book.index >= book.replace(0.0, np.nan).first_valid_index()]
    print(f"[load] book {len(book)} days, {book.index[0].date()} -> {book.index[-1].date()}")
    print("[load] OPRA SPX puts …")
    puts = load_puts()
    # Marks come from ALL bars, not only traded ones: a contract that stops printing still has
    # to be valued, and restricting to volume>0 would drop it from the path entirely.
    allp = pd.read_parquet(OPRA / "bars.parquet",
                           columns=["date", "close", "expiry", "cp", "strike"])
    allp = allp[(allp.cp == "P") & (allp.close > 0)]
    marks = allp.set_index(["expiry", "strike", "date"])["close"].sort_index()
    spot = spx_spot(book.index)
    base = summary_stats(book)
    print(f"[ok]   {len(puts):,} traded put bars\n")

    print("=" * 104)
    print(f"COST OF CAPPING THE BOOK'S LOSS WITH SPX PUTS  (beta {BETA}, real OPRA quotes)")
    print("=" * 104)
    print(f"  UNHEDGED: return {base['ann_return']:+.2%}  vol {base['ann_vol']:.2%}  "
          f"Sharpe {base['sharpe']:+.2f}  maxDD {base['max_drawdown']:+.1%}\n")
    print(f"  {'cap':>5s} {'anchor':>9s} {'strike':>8s} {'cost/yr':>9s} {'payoff/yr':>10s} "
          f"{'net/yr':>8s} | {'return':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s}")
    rows = []
    for cap in CAPS:
        for name in ("reset", "hwm", "hwm_atm"):
            h = 91
            r = run_hedge(puts, spot, book, cap, h, anchor=name, marks=marks)
            s = summary_stats(r["hedged"])
            rows.append({**{k: v for k, v in r.items() if k != "hedged"},
                         "return": s["ann_return"], "vol": s["ann_vol"],
                         "sharpe": s["sharpe"], "maxdd": s["max_drawdown"]})
            print(f"  {cap:5.0%} {name:>9s} {1 - cap / BETA:7.1%} {r['cost_yr']:9.2%} "
                  f"{r['payoff_yr']:10.2%} {r['net_yr']:+8.2%} | {s['ann_return']:+8.2%} "
                  f"{s['ann_vol']:7.2%} {s['sharpe']:+7.2f} {s['max_drawdown']:+8.1%}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 104)
    print("DOES THE CAP ACTUALLY HOLD?")
    print("=" * 104)
    print("  An SPX put hedges only the BETA component. The book carries 9.14% residual vol")
    print("  that no index put touches, so the nominal cap is not what you get:\n")
    print(f"  {'nominal cap':>12s} {'realised maxDD (reset anchor)':>34s} {'shortfall':>11s}")
    for cap in CAPS:
        row = df[(df["cap"] == cap) & (df["anchor"] == "reset")].iloc[0]
        print(f"  {cap:12.0%} {row['maxdd']:34.1%} {abs(row['maxdd']) - cap:+11.1%}")

    print("\n" + "=" * 104)
    print("VERDICT vs THE ALPHA IT IS PROTECTING")
    print("=" * 104)
    print(f"  Market-neutral alpha in this book: +3.56%/yr (doc s7, beta-hedged series).")
    for cap in CAPS:
        c = df[(df["cap"] == cap) & (df["anchor"] == "reset")].iloc[0]["cost_yr"]
        print(f"    cap {cap:.0%}: costs {c:.2%}/yr = {c / 0.0356:.1f}x the entire alpha")

    print("\n" + "=" * 104)
    print("DOES A LONGER-DATED PUT COST LESS? (cap 10%, overlapping cohorts every 42 days)")
    print("=" * 104)
    ch = cohort_study(puts, spot, book, cap=0.10)
    g = ch.groupby("horizon")
    print(f"  {'horizon':>8s} {'cohorts':>8s} {'rolls':>6s} {'cost/roll':>10s} {'cost/yr':>9s} "
          f"{'payoff/yr':>10s} {'net/yr':>9s} {'net p10..p90':>18s}")
    for h, gg in g:
        print(f"  {h:6d}d {len(gg):8d} {gg.rolls.mean():6.1f} {gg.cost_per_roll.mean():10.2%} "
              f"{gg.cost_yr.mean():9.2%} {gg.payoff_yr.mean():10.2%} {gg.net_yr.mean():+9.2%} "
              f"{gg.net_yr.quantile(.1):+8.2%}..{gg.net_yr.quantile(.9):+.2%}")
    print("\n  cost/roll rises with maturity (sqrt-of-time); cost/YR is the theta question.")

    out = ROOT / "results" / "put_hedge"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cost.csv", index=False)
    ch.to_csv(out / "cohorts.csv", index=False)
    print(f"\n  wrote {out}/cost.csv")


if __name__ == "__main__":
    main()
