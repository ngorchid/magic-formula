"""Holistic book assessment — combine the four strategies into one book, honestly.

Streams (with provenance — the book is only as honest as its inputs):
  TREND     : live cross-asset futures overlay net returns (results CSV)      — realistic, 2011-26
  REVERSAL  : mid+small market-neutral reversal champion, net-MOC 1.5bp        — backtest, survivorship-flattered, 2011-26
  VRP       : VIX/VIX3M regime-gated short-vol proxy                           — idealized proxy (level optimistic, corr real), 2011-26
  MAGIC_F   : Greenblatt magic formula stream (SimFin fundamentals)            — real backtest but short window (2022-25) + value-regime-hurt

Method (deliberately NOT mean-variance-optimized — the cvxpy result taught us that overfits):
risk-parity (inverse-vol) weights, the robust default. We report the correlation matrix (the
whole point), the risk-parity book Sharpe vs best single stream, a crisis-window stress, and the
leverage->return dial. Shown two ways: the 3-stream neutral-alpha book over the long window, and
the 4-stream book over the (MF-limited) common window, to see what the beta core adds.

Run: python scripts/book_assess.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from backtest import LinearCostModel, summary_stats
from data import download_ohlcv
from data.universe import sp1500_constituents, sp1500_sectors, sp1500_tickers
from strategies.equity_mn.neutralize import rolling_beta
from reversal_lab import Variant, build_weights


def reversal_series() -> pd.Series:
    import yfinance as yf
    panel = download_ohlcv(sorted(set(sp1500_tickers() + ["SPY"])), "2011-01-01", None)
    pf = panel["adj_close"].dropna(how="all", axis=1); vol = panel["volume"].reindex_like(pf)
    bench = pf["SPY"].pct_change(fill_method=None)
    prices = pf.drop(columns=["SPY"], errors="ignore"); vol = vol.drop(columns=["SPY"], errors="ignore")
    rets = prices.pct_change(fill_method=None)
    betas = rolling_beta(rets, bench, 252); sectors = sp1500_sectors().reindex(prices.columns)
    idio = rets.rolling(20).std()
    vix = (yf.download("^VIX", start="2011-01-01", auto_adjust=True, progress=False)["Close"].squeeze() / 100.0)
    tier = sp1500_constituents().set_index("ticker")["tier"].reindex(prices.columns)
    ms = [c for c in prices.columns if tier.get(c) in ("mid", "small")]
    p, v = prices[ms], vol[ms]
    w = build_weights(Variant("r", horizons=(1, 3, 5, 10), news_filter=True, smooth=2,
                              inv_vol=True, vix_scale=True),
                      p, v, betas.reindex(columns=ms), sectors.reindex(ms), idio.reindex(columns=ms), vix)
    dw = w.diff().abs().fillna(w.abs())
    adv = (p * v).rolling(21).mean().reindex_like(w).ffill(); adv = adv.fillna(adv.median().median())
    gross = (w * p.pct_change(fill_method=None).fillna(0.0)).sum(axis=1)
    return gross - LinearCostModel(1.5, 10.0).charge(dw * 1e6, adv) / 1e6


def vrp_series() -> pd.Series:
    import yfinance as yf
    tk = yf.download(["^VIX", "^VIX3M", "^GSPC"], start="2011-01-01", auto_adjust=True,
                     progress=False)["Close"].dropna()
    vix, vix3m, spx = tk["^VIX"] / 100, tk["^VIX3M"] / 100, tk["^GSPC"]
    ret = spx.pct_change()
    rv20 = ret.rolling(20).std() * np.sqrt(252)
    vrp, ratio = vix - rv20, vix / vix3m
    raw = ((vix.shift(1) ** 2) - 252 * (ret ** 2)).dropna()
    sv = raw * (0.10 / (raw.std() * np.sqrt(252)))           # scale to ~10% vol
    gate = (ratio.shift(1) < 1.00).reindex(sv.index).fillna(False) & (vrp.shift(1) > 0).reindex(sv.index).fillna(False)
    return sv * gate


def magic_series() -> pd.Series:
    from strategies.magic_formula import MagicFormula, MagicFormulaConfig
    return MagicFormula(MagicFormulaConfig()).run().net_returns


def stats_row(name, r, idx):
    s = summary_stats(r.reindex(idx).fillna(0.0))
    return (f"  {name:12s} {s['ann_return']:>+8.1%} {s['ann_vol']:>7.1%} {s['sharpe']:>+7.2f} "
            f"{s['max_drawdown']:>+8.1%} {r.reindex(idx).fillna(0.0).skew():>+6.2f}")


def risk_parity(S: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    vols = S.std().replace(0.0, np.nan)
    w = (1.0 / vols) / (1.0 / vols).sum()
    return (S * w).sum(axis=1), w


def crisis(S, book, windows):
    print("\n  crisis-window P&L (cumulative, book vs streams):")
    print("    " + "window".ljust(22) + "".join(f"{c:>10s}" for c in list(S.columns) + ["BOOK"]))
    for lbl, a, b in windows:
        vals = [(1 + S[c].loc[a:b]).prod() - 1 for c in S.columns] + [(1 + book.loc[a:b]).prod() - 1]
        print("    " + lbl.ljust(22) + "".join(f"{x:>+10.1%}" for x in vals))


def report(S: pd.DataFrame, title: str, crisis_windows):
    common = S.dropna().index
    book, w = risk_parity(S)
    bs = summary_stats(book.reindex(common).fillna(0.0))
    best = max(summary_stats(S[c].reindex(common).fillna(0.0))["sharpe"] for c in S)
    print("\n" + "=" * 78); print(f"{title}  ({common[0].date()}->{common[-1].date()})"); print("=" * 78)
    print(f"  {'stream':12s} {'annRet':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s} {'skew':>6s}")
    for c in S:
        print(stats_row(c, S[c], common))
    print("  " + "-" * 58)
    print(stats_row("BOOK (rp)", book, common) + f"   [best single {best:+.2f}]")
    print(f"\n  risk-parity weights: " + "  ".join(f"{c} {w[c]:.0%}" for c in w.index))
    print("\n  correlation matrix:")
    corr = S.corr()
    print("    " + "".ljust(12) + "".join(f"{c:>10s}" for c in corr.columns))
    for r in corr.index:
        print("    " + f"{r:12s}" + "".join(f"{corr.loc[r, c]:>+10.2f}" for c in corr.columns))
    crisis(S, book, crisis_windows)
    print(f"\n  leverage dial (Sharpe held ex-financing):  2x -> {2*bs['ann_return']:+.1%} @ {2*bs['ann_vol']:.1%}"
          f"   3x -> {3*bs['ann_return']:+.1%} @ {3*bs['ann_vol']:.1%}")
    return book


def main():
    print("building four strategy return series …")
    trend = pd.read_csv(ROOT / "results" / "trend_overlay" / "trend_overlay_net.csv",
                        index_col=0, parse_dates=True)["trend"]
    rev = reversal_series()
    vrp = vrp_series()
    mf = magic_series()
    for s in (trend, rev, vrp, mf):
        s.index = pd.to_datetime(s.index)

    # 3-stream neutral-alpha book (long window)
    S3 = pd.DataFrame({"trend": trend, "reversal": rev, "vrp": vrp}).dropna()
    report(S3, "3-STREAM NEUTRAL BOOK (trend + reversal + vrp)",
           [("2018 vol (Feb)", "2018-02-01", "2018-02-28"),
            ("2020 COVID", "2020-02-15", "2020-03-31"),
            ("2022 bear", "2022-01-01", "2022-12-31")])

    # 4-stream book incl. magic formula (common window is MF-limited)
    S4 = pd.DataFrame({"trend": trend, "reversal": rev, "vrp": vrp, "magic_f": mf}).dropna()
    report(S4, "4-STREAM BOOK (+ magic formula)",
           [("2022 bear", "2022-01-01", "2022-12-31"),
            ("2023-24 megacap rally", "2023-01-01", "2024-12-31")])
    print("\n  PROVENANCE: trend=live-cfg; reversal=backtest (survivorship-flattered);")
    print("  vrp=idealized proxy (level optimistic, corr real); magic_f=real backtest, short 2022-25 window.")


if __name__ == "__main__":
    main()
