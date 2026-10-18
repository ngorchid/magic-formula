"""Carry as a second trend signal (Appendix H.7): does blending carry into the trend overlay help the BOOK?

CARRY is what a futures position earns if prices do not move. Per market, annualised:
  FX (6E, 6A, 6J)   foreign 3-month rate minus the US 3-month rate (OECD interbank, FRED,
                    monthly; used from the month after it is dated, so no look-ahead)
  bonds (ZN, ZB)    10y / 30y Treasury yield minus the 3-month bill (FRED, daily, lagged 1 day)
  equity (ES)       S&P 500 dividend yield (SPY's trailing-year adjusted minus price return)
                    minus the 3-month bill
  commodities       NOT AVAILABLE: carry needs the futures curve, which is not free. Gold and
                    silver carry is roughly minus the financing rate (uninformative); oil and
                    copper would matter. These four markets keep the pure trend signal.
NB the eurozone rate series on FRED ends 2026-01 and is carried forward.

THE BLEND (fixed before the first run). The live signal is the mean of sign(6m return) and
sign(12m return), in [-1, 1]. The carry signal is sign(carry). Per market with carry:
    signal = (1 - w) * trend + w * carry,    w = 0.3 (MAIN);  0.5 (sensitivity)
and an arm with carry alone (w = 1, commodity markets flat) to see the signal on its own.
Everything else is the LIVE sizer: the lab runs `compute_targets` from the deployed trend-overlay
code with one change, the signal row passed in. The source is patched at run time, not copied,
so any later change to the live sizer is picked up; and at w = 0 the patched sizer must
reproduce the live one exactly -- checked on every day of every run (NULL CHECK).

BOOK TEST. The core is the enhanced magic-formula backtest (PIT S&P 500, 2012-2026). Each overlay
arm is scaled to 10% vol and added to the core, as in tail_conditional_lab.py. Two sizes:
  reference ($100M of notional): the signal's value with no whole-contract rounding
  live ($75k): whether it survives the rounding the live book actually has

PRE-REGISTERED DECISION (before the first run). The blend (w = 0.3) is worth building only if ALL:
  1. reference size: core + blend has a higher Sharpe than core + trend in BOTH halves of the
     common window;
  2. reference size: in the core's worst decile of months, the blend's mean return (vol-matched)
     is NOT LOWER than trend's -- trend is in the book for its crisis payoff, and a blend that
     buys Sharpe by giving it up is the wrong trade here;
  3. live size: the blend's overlay Sharpe is not below trend's.
w = 0.5 and carry alone are descriptive; nothing is chosen from them.

2008. The core starts in 2012, so the book test cannot see the year carry crashed hardest. A
descriptive run on the six financial markets alone (SPY, IEF, TLT, FXE, FXA, FXY have proxies
from 2005-07) reports each arm year by year from 2008.

Costs: 2bp per side on traded notional, as in every trend backtest here. Proxy returns, not
futures (ETF total returns include the bond coupon, which trend_proxy_validation.py showed
over-signals rates; identical across arms).

Usage:  python scripts/trend_carry_lab.py
"""
from __future__ import annotations

import inspect
import io
import logging
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "trend-overlay"))

import trend_overlay.execution as X  # noqa: E402
from trend_live_budget_backtest import COST_BP, WARMUP, load_proxies, stats  # noqa: E402
from trend_overlay.contracts import FUTURES  # noqa: E402
from trend_overlay.execution import TrendPaperConfig, compute_targets  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

W_MAIN, W_SENS = 0.3, 0.5
ARMS = {"trend": 0.0, f"blend_{W_MAIN:g}": W_MAIN, f"blend_{W_SENS:g}": W_SENS, "carry_only": 1.0}
LIVE, REF = 75_000, 100_000_000
TARGET_VOL = 0.10
FIN = ["SPY", "IEF", "TLT", "FXE", "FXA", "FXY"]
OUT = ROOT / "results" / "trend_carry"


# --------------------------------------------------------------------------- the patched sizer
def _patched_compute_targets():
    """compute_targets with an extra `sig_override` (a Series by proxy ETF), via source surgery."""
    src = textwrap.dedent(inspect.getsource(X.compute_targets))
    old_sig = "sig, v_raw, v = signal.iloc[-1], vol.iloc[-1], vol_used.iloc[-1]"
    old_def = "values: dict[str, float] | None = None) -> pd.DataFrame:"
    assert src.count(old_sig) == 1 and src.count(old_def) == 1, "live sizer changed: update the patch"
    src = src.replace(old_def, "values: dict[str, float] | None = None, sig_override=None) -> pd.DataFrame:")
    src = src.replace(old_sig, "sig, v_raw, v = (signal.iloc[-1] if sig_override is None else "
                               "sig_override.reindex(signal.columns)), vol.iloc[-1], vol_used.iloc[-1]")
    src = src.replace("def compute_targets(", "def compute_targets_sig(", 1)
    ns = dict(vars(X))
    exec(compile(src, "<compute_targets_sig>", "exec"), ns)
    return ns["compute_targets_sig"]


compute_targets_sig = _patched_compute_targets()


# --------------------------------------------------------------------------------- carry data
def _fred(sid: str) -> pd.Series:
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    d.columns = ["date", "v"]
    s = pd.Series(pd.to_numeric(d["v"], errors="coerce").values, index=pd.to_datetime(d["date"]))
    return s.dropna() / 100.0


def carry_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Annualised carry per proxy ETF, known at the close of each day (no look-ahead)."""
    def daily(s: pd.Series) -> pd.Series:           # daily FRED series: usable the next day
        return s.reindex(index.union(s.index)).ffill().reindex(index).shift(1)

    def monthly(s: pd.Series) -> pd.Series:         # dated the 1st of month M = M's average:
        s = s.copy()                                 # usable from the 1st of M+1
        s.index = s.index + pd.offsets.MonthBegin(1)
        return s.reindex(index.union(s.index)).ffill().reindex(index)

    bill = daily(_fred("DTB3"))
    us_m = monthly(_fred("IR3TIB01USM156N"))
    c = pd.DataFrame(index=index)
    c["IEF"] = daily(_fred("DGS10")) - bill
    c["TLT"] = daily(_fred("DGS30")) - bill
    c["FXE"] = monthly(_fred("IR3TIB01EZM156N")) - us_m
    c["FXA"] = monthly(_fred("IR3TIB01AUM156N")) - us_m
    c["FXY"] = monthly(_fred("IR3TIB01JPM156N")) - us_m
    spy = yf.download("SPY", start="2003-01-01", auto_adjust=False, progress=False)
    adj, cls = spy["Adj Close"].squeeze(), spy["Close"].squeeze()
    adj.index = cls.index = pd.to_datetime(cls.index).tz_localize(None).normalize()
    dy = (adj.pct_change() - cls.pct_change()).rolling(252, min_periods=240).sum()
    c["SPY"] = dy.reindex(index.union(dy.index)).ffill().reindex(index).shift(1) - bill
    return c


# ------------------------------------------------------------------------------------- the loop
def trend_row(px: pd.DataFrame, cfg: TrendPaperConfig) -> pd.Series:
    return sum(np.sign(px / px.shift(lb) - 1.0) for lb in cfg.lookbacks).iloc[-1] / len(cfg.lookbacks)


def blended(trend: pd.Series, carry: pd.Series, w: float) -> pd.Series:
    cs = np.sign(carry.reindex(trend.index))
    has = cs.notna()
    out = trend.copy()
    if w >= 1.0:
        out[:] = 0.0                                 # carry alone: markets without carry are flat
        out[has] = cs[has]
    else:
        out[has] = (1 - w) * trend[has] + w * cs[has]
    return out


def run(px: pd.DataFrame, carry: pd.DataFrame, budget: float, specs=FUTURES) -> pd.DataFrame:
    cfg = TrendPaperConfig(budget=budget)
    specs = [s for s in specs if s.proxy_etf in px.columns]
    proxy_of = {s.market: s.proxy_etf for s in specs}
    rets = px.pct_change(fill_method=None)
    dates = px.index[WARMUP:]
    held = {a: {} for a in ARMS}
    prev = {a: {} for a in ARMS}
    out = {a: [] for a in ARMS}
    null_checks = 0
    for i, t in enumerate(dates[:-1]):
        win = px.loc[:t]
        tr = trend_row(win[[s.proxy_etf for s in specs]], cfg)
        cr = carry.loc[t] if t in carry.index else pd.Series(dtype=float)
        nxt = dates[i + 1]
        r_next = rets.loc[nxt]
        for a, w in ARMS.items():
            tgt = compute_targets_sig(win, cfg, specs=specs, held=held[a] or None,
                                      sig_override=blended(tr, cr, w))
            if w == 0.0:                              # NULL CHECK against the live sizer
                live = compute_targets(win, cfg, specs=specs, held=held[a] or None)
                assert (live["contracts"] == tgt["contracts"]).all(), f"null check failed {t}"
                null_checks += 1
            pos = {m: int(c) for m, c in tgt["contracts"].items()}
            held[a] = pos
            expo = {m: pos[m] * float(tgt.at[m, "notional"]) for m in pos}
            pnl = sum(e * r_next.get(proxy_of[m], np.nan) for m, e in expo.items()
                      if e and np.isfinite(r_next.get(proxy_of[m], np.nan)))
            turn = sum(abs(expo.get(m, 0.0) - prev[a].get(m, 0.0)) for m in set(expo) | set(prev[a]))
            prev[a] = expo
            out[a].append((nxt, (pnl - turn * COST_BP / 1e4) / budget))
    assert null_checks == len(dates) - 1
    return pd.DataFrame({a: pd.Series(dict(v)) for a, v in out.items()})


# --------------------------------------------------------------------------------- the book
def scale_to_vol(s: pd.Series, target: float = TARGET_VOL) -> pd.Series:
    v = s.std() * np.sqrt(252)
    return s * (target / v) if v > 0 else s


def book_report(core: pd.Series, ov: pd.DataFrame, label: str) -> dict:
    idx = core.index.intersection(ov.dropna().index)
    core, ov = core.loc[idx], ov.loc[idx]
    mid = idx[len(idx) // 2]
    print(f"\n--- {label}: {idx.min():%Y-%m} -> {idx.max():%Y-%m}, overlays scaled to {TARGET_VOL:.0%} vol ---")
    print(f"  {'arm':14}{'overlay Sh':>11}{'book Sh':>9}{'1st half':>10}{'2nd half':>10}{'book maxDD':>12}"
          f"{'worst-decile mean':>19}{'tail corr':>11}")
    core_m = scale_to_vol(core).resample("ME").sum()
    cut = core_m.quantile(0.10)
    res = {}
    for a in ov.columns:
        o = scale_to_vol(ov[a])
        b = core + o
        h1, h2 = stats(b[b.index < mid])["sharpe"], stats(b[b.index >= mid])["sharpe"]
        om = o.resample("ME").sum()
        both = pd.concat([core_m, om], axis=1).dropna()
        tail = both[both.iloc[:, 0] <= cut]
        res[a] = {"ov": stats(ov[a])["sharpe"], "book": stats(b)["sharpe"], "h1": h1, "h2": h2,
                  "dd": stats(b)["maxdd"], "tail": tail.iloc[:, 1].mean(),
                  "tcorr": tail.iloc[:, 0].corr(tail.iloc[:, 1])}
        r = res[a]
        print(f"  {a:14}{r['ov']:>+11.2f}{r['book']:>+9.2f}{h1:>+10.2f}{h2:>+10.2f}{r['dd']:>+12.1%}"
              f"{r['tail']:>+19.2%}{r['tcorr']:>+11.2f}")
    s = stats(core)
    print(f"  {'core alone':14}{'':>11}{s['sharpe']:>+9.2f}{'':>20}{s['maxdd']:>+12.1%}")
    print(f"  overlay correlation with trend: " + ", ".join(
        f"{a} {ov[a].corr(ov['trend']):+.2f}" for a in ov.columns if a != "trend"))
    return res


def main() -> None:
    px = load_proxies()
    carry = carry_frame(px.index)
    print(f"proxy panel {px.shape}, {px.index.min():%Y-%m} -> {px.index.max():%Y-%m}")
    last = carry.dropna(how="all").iloc[-1]
    print("carry today (annualised): " + ", ".join(f"{k} {v:+.2%}" for k, v in last.items()))
    print("share of days with POSITIVE carry: " + ", ".join(
        f"{k} {(carry[k].dropna() > 0).mean():.0%}" for k in carry.columns))

    core = pd.read_csv(ROOT / "results" / "best_magic" / "best_sp500_pit_all.csv",
                       index_col=0, parse_dates=True)["net_return"]
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for budget, label in ((REF, "REFERENCE ($100M, no rounding)"), (LIVE, "LIVE SIZE ($75k notional)")):
        ov = run(px, carry, budget)
        ov.to_csv(OUT / f"overlay_{budget}.csv")
        res[budget] = book_report(core, ov, label)
        print(f"  (null check: patched sizer == live sizer on every day at w=0)")

    # ---- 2008: six financial markets from 2005 -------------------------------------------
    fin = yf.download(FIN, start="2005-01-01", end="2026-09-14", auto_adjust=True,
                      progress=False, group_by="column")["Close"]
    fin.index = pd.to_datetime(fin.index).tz_localize(None).normalize()
    fin = fin.dropna(how="all").ffill()
    ovf = run(fin, carry_frame(fin.index), REF)
    ovf.to_csv(OUT / "financial_markets_2005.csv")
    print("\n--- SIX FINANCIAL MARKETS ONLY (reference size), calendar-year return at 10% vol ---")
    yr = pd.DataFrame({a: scale_to_vol(ovf[a].dropna()).groupby(ovf.dropna().index.year).sum()
                       for a in ovf.columns})
    print(yr.loc[2008:].to_string(float_format=lambda x: f"{x:+.1%}"))
    print("  full-period Sharpe: " + ", ".join(f"{a} {stats(ovf[a].dropna())['sharpe']:+.2f}" for a in ovf))

    # ---- decision ---------------------------------------------------------------------------
    m, t = f"blend_{W_MAIN:g}", "trend"
    R, L = res[REF], res[LIVE]
    c1 = R[m]["h1"] > R[t]["h1"] and R[m]["h2"] > R[t]["h2"]
    c2 = R[m]["tail"] >= R[t]["tail"]
    c3 = L[m]["ov"] >= L[t]["ov"]
    print("\nPRE-REGISTERED DECISION (blend w=0.3 vs trend):")
    print(f"  1. book Sharpe higher in both halves (reference): {R[t]['h1']:+.2f}/{R[t]['h2']:+.2f} -> "
          f"{R[m]['h1']:+.2f}/{R[m]['h2']:+.2f}  {'YES' if c1 else 'NO'}")
    print(f"  2. worst-decile payoff not lower (reference): {R[t]['tail']:+.2%} -> {R[m]['tail']:+.2%}  "
          f"{'YES' if c2 else 'NO'}")
    print(f"  3. overlay Sharpe not lower at live size: {L[t]['ov']:+.2f} -> {L[m]['ov']:+.2f}  "
          f"{'YES' if c3 else 'NO'}")
    print(f"  -> {'WORTH BUILDING' if (c1 and c2 and c3) else 'NOT worth building on this evidence'}")


if __name__ == "__main__":
    main()
