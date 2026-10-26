"""Trend: markets never tested (Appendix H.9) -- does any of them earn a place in the live basket?

WHY NOW. The book's margin cushion (40.6% against a 40% floor) leaves no room for a new sleeve, so
an alpha that can be implemented has to come from inside the existing ones. Adding a market to the
trend overlay does not raise its gross exposure: the sizer splits the same risk budget across more
markets, so margin stays about the same. And trend is the book's only genuine diversifier.

CANDIDATES. Markets with a small enough contract to fit the $30k per-market cap at the live $75k,
and an ETF that holds the futures or the currency itself (so its return is what a futures holder
earns, roll included), with history from 2011 or earlier:
  corn         CORN  ZC  / mini XC   (1,000 bu)
  soybeans     SOYB  ZS  / mini XK   (1,000 bu)
  wheat        WEAT  ZW  / mini XW   (1,000 bu)
  natural gas  UNG   NG  / micro MNG (1,000 MMBtu)
  pound        FXB   6B  / micro M6B
  Canadian $   FXC   6C  / micro MCD
  Swiss franc  FXF   6S  / micro MSF
CONTROL: Russell 2000 (IWM, micro M2K), which mostly repeats US equity (ES). If the control also
"improves" the book, the test is measuring noise.

METHOD. The deployed sizer (`compute_targets` from trend-overlay) walked daily over ETF proxies,
for the live basket and for the live basket plus ONE candidate, at the reference size ($100M, no
rounding) and at the live $75k; 2bp a side on traded notional; contract values as stored
(Appendix E #23 caveat applies equally to all arms). Each arm is scaled to 10% vol and added to
the magic-formula core (PIT S&P 500 backtest), as in H.7.

RULE (fixed before the first run, 2026-10-26). A candidate is worth adding only if ALL hold:
  1. reference size: core + (basket + candidate) has a higher Sharpe than core + basket in BOTH
     halves of the common window;
  2. reference size: in the core's worst decile of months, the arm's mean return (vol-matched) is
     not lower than the basket's (trend is in the book for its crisis payoff);
  3. live size: the arm's overlay Sharpe is not below the basket's.
And the control must FAIL rule 1, else the whole test is void. A combined arm of every passing
candidate is reported afterwards as descriptive only.

Usage: python scripts/trend_new_markets_lab.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "trend-overlay"))

from trend_live_budget_backtest import COST_BP, WARMUP, stats  # noqa: E402
from trend_overlay.contracts import FUTURES, FutureSpec  # noqa: E402
from trend_overlay.execution import TrendPaperConfig, compute_targets  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

LIVE, REF = 75_000, 100_000_000
TARGET_VOL = 0.10
START, END = "2009-01-01", "2026-09-14"
OUT = ROOT / "results" / "trend_new_markets"

CANDIDATES = {
    "corn":     FutureSpec("corn", "CORN", "ZC", "CBOT", "USD", 50, 21_500, "XC", 10, 4_300, notice_buffer_days=30),
    "soybeans": FutureSpec("soybeans", "SOYB", "ZS", "CBOT", "USD", 50, 52_500, "XK", 10, 10_500, notice_buffer_days=30),
    "wheat":    FutureSpec("wheat", "WEAT", "ZW", "CBOT", "USD", 50, 27_500, "XW", 10, 5_500, notice_buffer_days=30),
    "natgas":   FutureSpec("natgas", "UNG", "NG", "NYMEX", "USD", 10000, 35_000, "MNG", 1000, 3_500, notice_buffer_days=14),
    "fx_gbp":   FutureSpec("fx_gbp", "FXB", "6B", "CME", "USD", 62500, 81_000, "M6B", 6250, 8_100, notice_buffer_days=10),
    "fx_cad":   FutureSpec("fx_cad", "FXC", "6C", "CME", "USD", 100000, 73_000, "MCD", 10000, 7_300, notice_buffer_days=10),
    "fx_chf":   FutureSpec("fx_chf", "FXF", "6S", "CME", "USD", 125000, 156_000, "MSF", 12500, 15_600, notice_buffer_days=10),
}
CONTROL = {"russell": FutureSpec("russell", "IWM", "RTY", "CME", "USD", 50, 120_000, "M2K", 5, 12_000, notice_buffer_days=5)}


def load(specs) -> pd.DataFrame:
    cols = sorted({s.proxy_etf for s in specs})
    px = yf.download(cols, start=START, end=END, auto_adjust=True, progress=False, group_by="column")["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px.dropna(how="all").ffill()
    first = max(px[c].first_valid_index() for c in cols)       # common window: every proxy live
    return px.loc[first:]


def run(px: pd.DataFrame, budget: float, specs) -> pd.Series:
    cfg = TrendPaperConfig(budget=budget)
    proxy_of = {s.market: s.proxy_etf for s in specs}
    rets = px.pct_change(fill_method=None)
    dates = px.index[WARMUP:]
    held, prev, out = {}, {}, []
    for i, t in enumerate(dates[:-1]):
        tgt = compute_targets(px.loc[:t], cfg, specs=specs, held=held or None)
        pos = {m: int(c) for m, c in tgt["contracts"].items()}
        held = pos
        expo = {m: pos[m] * float(tgt.at[m, "notional"]) for m in pos}
        r_next = rets.loc[dates[i + 1]]
        pnl = sum(e * r_next.get(proxy_of[m], np.nan) for m, e in expo.items()
                  if e and np.isfinite(r_next.get(proxy_of[m], np.nan)))
        turn = sum(abs(expo.get(m, 0.0) - prev.get(m, 0.0)) for m in set(expo) | set(prev))
        prev = expo
        out.append((dates[i + 1], (pnl - turn * COST_BP / 1e4) / budget))
    return pd.Series(dict(out))


def scale(s: pd.Series) -> pd.Series:
    v = s.std() * np.sqrt(252)
    return s * (TARGET_VOL / v) if v > 0 else s


def judge(core: pd.Series, ov: pd.DataFrame) -> pd.DataFrame:
    idx = core.index.intersection(ov.dropna().index)
    core, ov = core.loc[idx], ov.loc[idx]
    mid = idx[len(idx) // 2]
    cm = scale(core).resample("ME").sum()
    cut = cm.quantile(0.10)
    rows = {}
    for a in ov.columns:
        o = scale(ov[a])
        b = core + o
        om = o.resample("ME").sum()
        both = pd.concat([cm, om], axis=1).dropna()
        rows[a] = {"overlay_sh": stats(ov[a])["sharpe"], "book_sh": stats(b)["sharpe"],
                   "h1": stats(b[b.index < mid])["sharpe"], "h2": stats(b[b.index >= mid])["sharpe"],
                   "book_dd": stats(b)["maxdd"], "worst_decile": both[both.iloc[:, 0] <= cut].iloc[:, 1].mean(),
                   "corr_basket": ov[a].corr(ov["basket"])}
    print(f"  window {idx.min():%Y-%m} -> {idx.max():%Y-%m}; core alone Sharpe {stats(core)['sharpe']:+.2f}")
    return pd.DataFrame(rows).T


def main() -> None:
    allspecs = list(FUTURES) + list(CANDIDATES.values()) + list(CONTROL.values())
    px = load(allspecs)
    print(f"proxy panel {px.shape}, common window {px.index.min():%Y-%m} -> {px.index.max():%Y-%m}")
    core = pd.read_csv(ROOT / "results" / "best_magic" / "best_sp500_pit_all.csv",
                       index_col=0, parse_dates=True)["net_return"]
    arms = {"basket": list(FUTURES)}
    for k, s in {**CANDIDATES, **CONTROL}.items():
        arms[f"+{k}"] = list(FUTURES) + [s]
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for budget, label in ((REF, "REFERENCE ($100M)"), (LIVE, "LIVE ($75k)")):
        ov = pd.DataFrame({a: run(px, budget, sp) for a, sp in arms.items()})
        ov.to_csv(OUT / f"overlay_{budget}.csv")
        print(f"\n--- {label} ---")
        res[budget] = judge(core, ov)
        print(res[budget].to_string(float_format=lambda x: f"{x:+.3f}"))

    R, L = res[REF], res[LIVE]
    b = R.loc["basket"]
    print("\nPRE-REGISTERED DECISION (each vs the live basket):")
    passing = []
    for a in R.index.drop("basket"):
        c1 = R.at[a, "h1"] > b["h1"] and R.at[a, "h2"] > b["h2"]
        c2 = R.at[a, "worst_decile"] >= b["worst_decile"]
        c3 = L.at[a, "overlay_sh"] >= L.at["basket", "overlay_sh"]
        ok = c1 and c2 and c3
        print(f"  {a:10} 1:{'Y' if c1 else 'n'} 2:{'Y' if c2 else 'n'} 3:{'Y' if c3 else 'n'}  "
              f"book {b['book_sh']:+.2f} -> {R.at[a, 'book_sh']:+.2f}, live overlay "
              f"{L.at['basket', 'overlay_sh']:+.2f} -> {L.at[a, 'overlay_sh']:+.2f}  -> {'ADD' if ok else '-'}")
        if ok and a[1:] in CANDIDATES:
            passing.append(a[1:])
    ctl_c1 = R.at["+russell", "h1"] > b["h1"] and R.at["+russell", "h2"] > b["h2"]
    print(f"  control fails rule 1: {'YES (test valid)' if not ctl_c1 else 'NO -> test VOID'}")
    if passing and not ctl_c1:
        specs = list(FUTURES) + [CANDIDATES[k] for k in passing]
        print(f"\n(descriptive) all passing together: {', '.join(passing)}")
        for budget, label in ((REF, "reference"), (LIVE, "live")):
            ov = pd.DataFrame({"basket": run(px, budget, list(FUTURES)), "combined": run(px, budget, specs)})
            r = judge(core, ov)
            print(f"  {label}: " + "; ".join(f"{a} overlay {r.at[a, 'overlay_sh']:+.2f} book {r.at[a, 'book_sh']:+.2f} "
                                             f"halves {r.at[a, 'h1']:+.2f}/{r.at[a, 'h2']:+.2f} worst-decile "
                                             f"{r.at[a, 'worst_decile']:+.2%}" for a in r.index))


if __name__ == "__main__":
    main()
