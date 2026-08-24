"""SPX put-spread backtest on real OPRA prices — the first REAL test of options-vrp.

Everything about this strategy was previously either theory or a variance-swap proxy. This
prices actual traded SPX options, 2013-2026, and answers three questions:

  (a) the VRP entry filter — is "IV - RV20 > 2 points" the right threshold, or any threshold?
  (b) the delta distance   — is 16D short / 10D long right?
  (e) the 2x stop          — does it help or hurt on DEFINED-RISK spreads?

(e) is the one we could never test before: it is path-dependent, so it needs daily marks,
which is exactly what no proxy could give us.

METHOD
  entry   on each day with both legs live at 30-45 DTE, if flat and the VRP filter passes,
          sell the strike nearest the short delta and buy nearest the long delta, same expiry.
  mark    each day from the traded close; when a leg does not trade (~24% of days) re-price
          with Black-Scholes using its LAST OBSERVED IV and today's spot/DTE. That is what a
          broker's mark does and it uses no future information.
  manage  take profit at 50% of credit; optional stop at `stop_mult` x credit; time-stop at
          21 DTE; otherwise settle at intrinsic on the expiry.
  cost    ohlcv-1d has NO bid/ask, so the spread must be ASSUMED. Charged as `cost_pts` index
          points per leg per side (4 crossings per round trip) and swept, because it is the
          one input the data cannot supply.

HONEST LIMITS: SPX not SPY and not the 14-name basket; trade prices not quotes, so entry marks
carry timing noise against the 4pm spot; no assignment/pin modelling (SPX is cash-settled
European, so this matters much less than it would for SPY).

Run: python scripts/spx_vrp_lab.py
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from scripts.spx_chain import DIV_YIELD, bs_price  # noqa: E402

OUT = ROOT / "results" / "spx_vrp"


@dataclass
class Config:
    short_delta: float = 0.16       # absolute value; puts are negative-delta
    long_delta: float = 0.10
    vrp_min: float = 0.02           # ATM IV - RV20, in vol POINTS (0.02 = 2 points)
    dte_lo: int = 30
    dte_hi: int = 45
    profit_target: float = 0.50     # close when value <= 50% of credit
    stop_mult: float = 2.0          # close when value >= 2x credit; <=0 disables
    time_stop_dte: int = 21
    cost_pts: float = 0.50          # assumed half-spread per leg per side, index points
    # MARKET-WIDE REGIME GATE — the live strategy's `regime_open(ratio, threshold)`: sell only
    # in contango. None disables it. `regime_continuous` also CLOSES an open spread if the
    # gate shuts mid-hold; entry-only gating cannot protect a position already on, which is
    # why the daily-rebalanced variance-swap proxy looked far more protective than reality.
    regime_thr: float | None = None
    regime_continuous: bool = False
    # EXIT-side regime rule, INDEPENDENT of the entry gate. The entry-only gate was removed
    # 2026-08-15 because the damage demonstrably arrives in positions ALREADY OPEN (Aug 2024: the
    # gate blocked nothing, protected nothing, and an open spread lost $2,186). Testing an exit
    # rule requires it to be separable from entry, which `regime_continuous` was not: it demanded
    # regime_thr, which also gates entry, so the two effects could not be told apart.
    regime_exit_thr: float | None = None
    # Fraction of the position CLOSED when the exit rule fires. 1.0 = close it (default and
    # historical behaviour); 0.5 = halve and keep running. Added 2026-08-24: the "reduce 50%"
    # variant was quoted as Sharpe +0.70 in commit 68aa42e but was an AD-HOC run -- the option
    # was never in the committed code, so its P&L and drawdown could not be reproduced.
    # Partial closes stay ONE P&L observation per POSITION (the legs are summed) so the trade
    # count, and therefore the sqrt(n/yrs) Sharpe annualisation, stays comparable across arms.
    regime_exit_frac: float = 1.0
    cp: str = "P"                   # "P" = bull put spread (default), "C" = bear call spread

    @property
    def label(self) -> str:
        s = "no-stop" if self.stop_mult <= 0 else f"{self.stop_mult:g}x"
        return f"{self.short_delta:.2f}/{self.long_delta:.2f} vrp{self.vrp_min:.2f} {s}"


def regime_ratio() -> pd.Series:
    """VIX/VIX3M, lagged one day (decided on yesterday's close, applied today)."""
    import yfinance as yf
    tk = yf.download(["^VIX", "^VIX3M"], start="2013-01-01", auto_adjust=False,
                     progress=False)["Close"].dropna()
    return (tk["^VIX"] / tk["^VIX3M"]).shift(1)


def load() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ch = pd.read_parquet(OUT / "chain.parquet")
    ch = ch[ch.iv.notna()]
    spot = ch.groupby("date").spot.first()
    ret = spot.pct_change()
    rv20 = ret.rolling(20).std() * np.sqrt(252)
    atm = ch[(ch.dte.between(25, 40)) & (ch.mny.abs() < 0.015)]
    atm_iv = atm.groupby("date").iv.median()
    vrp = (atm_iv - rv20).dropna()
    return ch, spot, vrp


def pick_legs(day: pd.DataFrame, cfg: Config):
    """Nearest-delta short and long leg on a common expiry.

    Puts have negative delta and the long wing sits at a LOWER strike; calls have positive
    delta and the long wing sits HIGHER. Both are credit spreads with capped loss.
    """
    sgn = -1.0 if cfg.cp == "P" else 1.0
    p = day[(day.cp == cfg.cp) & day.dte.between(cfg.dte_lo, cfg.dte_hi)]
    if p.empty:
        return None
    best = None
    for exp, g in p.groupby("expiry"):
        s = g.iloc[(g.delta - sgn * cfg.short_delta).abs().argsort()[:1]]
        l = g.iloc[(g.delta - sgn * cfg.long_delta).abs().argsort()[:1]]
        if s.empty or l.empty:
            continue
        s, l = s.iloc[0], l.iloc[0]
        if (l.strike >= s.strike) if cfg.cp == "P" else (l.strike <= s.strike):
            continue                                   # long wing must be further OTM
        if abs(abs(s.delta) - cfg.short_delta) > 0.05: # refuse a bad delta match
            continue
        if abs(abs(l.delta) - cfg.long_delta) > 0.05:
            continue
        credit = s.close - l.close
        if credit <= 0:
            continue
        cand = (exp, s, l, credit)
        if best is None or s.dte < best[1].dte:
            best = cand
    return best


def mark(chain_by_date: dict, date, contract, strike, expiry, cp, spot, r, last_iv):
    """Traded close if it printed today; else a BS mark from the last observed IV."""
    d = chain_by_date.get(date)
    if d is not None:
        hit = d.get(contract)
        if hit is not None:
            return hit[0], hit[1]      # (price, iv)
    dte = (expiry - date).days
    if dte <= 0 or last_iv is None or not np.isfinite(last_iv):
        return None, last_iv
    px = bs_price(spot, strike, dte / 365.0, r, DIV_YIELD, last_iv, -1.0 if cp == "P" else 1.0)
    return (float(px) if np.isfinite(px) else None), last_iv


def run(cfg: Config, ch: pd.DataFrame, spot: pd.Series, vrp: pd.Series,
        ratio: pd.Series | None = None, daily_out: list | None = None) -> pd.DataFrame:
    """`daily_out`, if given, receives {date, unrealized} each day a position is open, in
    DOLLARS. Needed to build an honest equity curve: attributing a trade's P&L to its exit date
    hides every intra-trade drawdown, which for a 30-45 day short-put hold is most of the
    drawdown there is. Default None leaves behaviour unchanged."""
    # Fetch the ratio if EITHER side needs it. Keying this on regime_thr alone meant an
    # exit-only config left `ratio` as None and the exit rule silently never fired — reporting
    # as "the rule does nothing" rather than "the rule was never evaluated".
    if (cfg.regime_thr is not None or cfg.regime_exit_thr is not None) and ratio is None:
        ratio = regime_ratio()
    dates = np.array(sorted(ch.date.unique()))
    by_date_full = {d: g for d, g in ch.groupby("date")}
    lut = {d: dict(zip(g.contract, zip(g.close, g.iv))) for d, g in ch.groupby("date")}
    rates = ch.groupby("date").r.first()

    trades, open_pos = [], None
    for i, d in enumerate(dates):
        S = spot.get(d)
        if S is None or not np.isfinite(S):
            continue
        r = rates.get(d, 0.02)

        if open_pos is not None:
            ps, iv_s = mark(lut, d, open_pos["cs"], open_pos["ks"], open_pos["exp"], open_pos["cp"], S, r, open_pos["iv_s"])
            pl, iv_l = mark(lut, d, open_pos["cl"], open_pos["kl"], open_pos["exp"], open_pos["cp"], S, r, open_pos["iv_l"])
            open_pos["iv_s"], open_pos["iv_l"] = iv_s, iv_l
            dte = (open_pos["exp"] - d).days
            if ps is not None and pl is not None:
                val = ps - pl
                open_pos["peak"] = max(open_pos["peak"], val)
                if daily_out is not None:
                    daily_out.append({"date": d,
                                      "unrealized": (open_pos["credit"] - val) * 100})
                reason = None
                if val <= cfg.profit_target * open_pos["credit"]:
                    reason = "profit"
                elif cfg.stop_mult > 0 and val >= cfg.stop_mult * open_pos["credit"]:
                    reason = "stop"
                elif dte <= cfg.time_stop_dte:
                    reason = "time"
                _xthr = (cfg.regime_exit_thr if cfg.regime_exit_thr is not None
                         else (cfg.regime_thr if cfg.regime_continuous else None))
                if reason is None and _xthr is not None and ratio is not None:
                    rr = ratio.get(d)
                    if rr is not None and np.isfinite(rr) and rr >= _xthr:
                        reason = "regime"
                if dte <= 0:
                    if open_pos["cp"] == "P":
                        val = max(0.0, open_pos["ks"] - S) - max(0.0, open_pos["kl"] - S)
                    else:
                        val = max(0.0, S - open_pos["ks"]) - max(0.0, S - open_pos["kl"])
                    reason = "expiry"
                # PARTIAL regime reduction: realise `frac`, keep the rest running.
                if (reason == "regime" and 0.0 < cfg.regime_exit_frac < 1.0
                        and open_pos["size"] > cfg.regime_exit_frac):
                    f = cfg.regime_exit_frac
                    open_pos["realised"] += ((open_pos["credit"] - val) * 100 * f
                                             - 4 * cfg.cost_pts * 100 * f)
                    open_pos["size"] -= f
                    open_pos["reduced"] = True
                    reason = None          # position stays open at reduced size

                if reason:
                    sz = open_pos["size"]
                    pnl = (open_pos["realised"]
                           + (open_pos["credit"] - val) * 100 * sz
                           - 4 * cfg.cost_pts * 100 * sz)
                    trades.append({**{k: open_pos[k] for k in
                                      ("entry_date", "exp", "ks", "kl", "credit", "peak",
                                       "ds", "dl", "vrp", "cp")},
                                   "exit_date": d, "exit_val": val, "reason": reason,
                                   "pnl": pnl,
                                   "peak_mult": open_pos["peak"] / open_pos["credit"],
                                   "reduced": open_pos.get("reduced", False)})
                    open_pos = None

        if open_pos is None:
            if cfg.regime_thr is not None and ratio is not None:
                rr = ratio.get(d)
                if rr is None or not np.isfinite(rr) or rr >= cfg.regime_thr:
                    continue
            v = vrp.get(d)
            if v is None or not np.isfinite(v) or v < cfg.vrp_min:
                continue
            day = by_date_full.get(d)
            if day is None:
                continue
            pick = pick_legs(day, cfg)
            if pick is None:
                continue
            exp, s, l, credit = pick
            open_pos = {"entry_date": d, "exp": exp, "ks": s.strike, "kl": l.strike,
                        "cs": s.contract, "cl": l.contract, "credit": credit,
                        "iv_s": s.iv, "iv_l": l.iv, "peak": credit * 0.0,
                        "ds": s.delta, "dl": l.delta, "vrp": v, "cp": cfg.cp,
                        "size": 1.0, "realised": 0.0, "reduced": False}
    return pd.DataFrame(trades)


def stats(t: pd.DataFrame, cfg: Config) -> dict:
    if t.empty:
        return {"label": cfg.label, "n": 0}
    yrs = (t.exit_date.max() - t.entry_date.min()).days / 365.25
    pnl = t.pnl
    wins = pnl > 0
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    return {"label": cfg.label, "n": len(t), "per_yr": len(t) / yrs,
            "total": pnl.sum(), "per_trade": pnl.mean(), "med": pnl.median(),
            "win%": wins.mean(), "sd": pnl.std(),
            "sharpe": pnl.mean() / pnl.std() * np.sqrt(len(t) / yrs) if pnl.std() else np.nan,
            "maxDD": dd, "worst": pnl.min(),
            "profit%": (t.reason == "profit").mean(), "stop%": (t.reason == "stop").mean(),
            "time%": (t.reason == "time").mean(), "exp%": (t.reason == "expiry").mean(),
            "regime%": (t.reason == "regime").mean()}


def table(rows: list[dict], title: str) -> None:
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    print(f"  {'variant':28s} {'n':>4s} {'/yr':>5s} {'total$':>10s} {'$/trade':>9s} "
          f"{'win%':>6s} {'Sharpe':>7s} {'maxDD$':>10s} {'worst$':>9s} | "
          f"{'prof':>5s}{'stop':>5s}{'time':>5s}{'exp':>5s}")
    print("  " + "-" * 114)
    for r in rows:
        if not r.get("n"):
            print(f"  {r['label']:28s}  no trades")
            continue
        print(f"  {r['label']:28s} {r['n']:>4d} {r['per_yr']:>5.1f} {r['total']:>10,.0f} "
              f"{r['per_trade']:>9,.0f} {r['win%']:>6.0%} {r['sharpe']:>+7.2f} "
              f"{r['maxDD']:>10,.0f} {r['worst']:>9,.0f} | "
              f"{r['profit%']:>5.0%}{r['stop%']:>5.0%}{r['time%']:>5.0%}{r['exp%']:>5.0%}")


def main() -> None:
    print("Loading chain...")
    ch, spot, vrp = load()
    print(f"  {len(ch):,} rows, {ch.date.nunique():,} days, VRP series {len(vrp):,} days "
          f"(median {vrp.median()*100:+.2f} pts)")

    base = Config()
    t = run(base, ch, spot, vrp)
    table([stats(t, base)], "BASELINE — live config (16D/10D, VRP>2pts, 2x stop, 50% TP, 21DTE)")
    t.to_parquet(OUT / "trades_baseline.parquet", index=False)

    # (a) VRP threshold
    rows = []
    for v in (-99, 0.0, 0.01, 0.02, 0.03, 0.05):
        c = Config(vrp_min=v)
        c_lab = Config(vrp_min=v)
        r = stats(run(c, ch, spot, vrp), c)
        r["label"] = ("no VRP filter" if v < -1 else f"VRP > {v*100:+.0f} pts")
        rows.append(r)
    table(rows, "(a) VRP ENTRY FILTER — is the 2-point threshold doing anything?")

    # (b) delta distance
    rows = []
    for sd, ld in ((0.30, 0.20), (0.25, 0.15), (0.20, 0.10), (0.16, 0.10),
                   (0.16, 0.08), (0.12, 0.06), (0.10, 0.05)):
        c = Config(short_delta=sd, long_delta=ld)
        r = stats(run(c, ch, spot, vrp), c)
        r["label"] = f"short {sd:.2f}D / long {ld:.2f}D"
        rows.append(r)
    table(rows, "(b) DELTA DISTANCE — where on the curve should we sell?")

    # (e) the stop
    rows = []
    for sm in (0.0, 1.5, 2.0, 3.0):
        c = Config(stop_mult=sm)
        r = stats(run(c, ch, spot, vrp), c)
        r["label"] = "NO stop (wing only)" if sm <= 0 else f"stop at {sm:g}x credit"
        rows.append(r)
    table(rows, "(e) THE 2x STOP — helps or hurts on a defined-risk spread?")

    print(f"\n  wrote {OUT}/trades_baseline.parquet")
    print("\n  NB cost is ASSUMED (0.50 index pts/leg/side) — ohlcv-1d has no bid/ask.")


if __name__ == "__main__":
    main()
