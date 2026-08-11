"""Book lab — the stacking thesis: do uncorrelated streams lift book Sharpe more than
squeezing one? Pushing one stream 0.65->0.75 grinds against crowding; combining N
uncorrelated ~0.6 streams compounds (2 -> ~0.85, 3 -> ~1.0). This is the medallion lesson.

Three streams, all on free daily data / existing artifacts:
  - REVERSAL   : short-horizon (1-2d) cross-sectional reversal, mid+small, MOC-cost (this project)
  - MOMENTUM   : residual 12-1 momentum, monthly, market-neutral (the textbook diversifier to
                 short reversal — opposite horizon, ~0 correlation)
  - TREND      : the LIVE cross-asset futures overlay's net returns (results/trend_overlay CSV)

Reports per-stream ann/vol/Sharpe, the correlation matrix (the whole point), and the combined
inverse-vol book vs the best single stream — plus what modest leverage does to book return.

Run: python scripts/book_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import LinearCostModel, summary_stats
from data import download_ohlcv
from data.universe import sp1500_constituents, sp1500_sectors, sp1500_tickers
from signals import residual_momentum
from combination import cs_zscore, winsorize
from strategies.equity_mn.neutralize import neutralize, rolling_beta
from scripts.reversal_lab import Variant, build_weights

CHAMP = Variant("rev", horizons=(1, 3, 5, 10), news_filter=True, smooth=2,
                inv_vol=True, vix_scale=True)


def neutral_book_returns(signal, prices, volume, betas, sectors, rebalance, spread_bps):
    """Generic market-neutral L/S book from a raw signal: winsorize -> z -> beta/sector
    neutralize -> dollar-neutral weights -> rebalance/hold -> net of cost."""
    z = neutralize(cs_zscore(winsorize(signal.reindex_like(prices), 0.01, 0.99)),
                   betas=betas, sectors=sectors)
    z = z.sub(z.mean(axis=1), axis=0)
    w = z.div(z.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    rebal = prices.resample(rebalance).last().index.intersection(prices.index)
    tgt = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    tgt.loc[rebal] = w.loc[rebal]
    w = tgt.ffill().fillna(0.0).shift(1).fillna(0.0)
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    dw = w.diff().abs().fillna(w.abs())
    adv = (prices * volume).rolling(21).mean().reindex_like(w).ffill()
    adv = adv.fillna(adv.median().median())
    cm = LinearCostModel(half_spread_bps=spread_bps, impact_coef_bps=10.0)
    return gross - cm.charge(dw * 1_000_000.0, adv) / 1_000_000.0


def main(start="2011-01-01", end=None):
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"loading S&P1500 + SPY, {start}->{end} …")
    panel = download_ohlcv(sorted(set(sp1500_tickers() + ["SPY"])), start, end)
    pf = panel["adj_close"].dropna(how="all", axis=1)
    volume = panel["volume"].reindex_like(pf).drop(columns=["SPY"], errors="ignore")
    bench = pf["SPY"].pct_change(fill_method=None)
    prices = pf.drop(columns=["SPY"], errors="ignore")
    rets = prices.pct_change(fill_method=None)
    betas = rolling_beta(rets, bench, window=252)
    sectors = sp1500_sectors().reindex(prices.columns)
    idio_vol = rets.rolling(20).std()
    import yfinance as yf
    vix = (yf.download("^VIX", start=start, end=end, auto_adjust=True,
                       progress=False)["Close"].squeeze() / 100.0)
    tier = sp1500_constituents().set_index("ticker")["tier"].reindex(prices.columns)
    midsm = [c for c in prices.columns if tier.get(c) in ("mid", "small")]

    # Stream 1: reversal (mid+small, MOC ~1.5bp) — realistic-cost version of the champion.
    p, v = prices[midsm], volume[midsm]
    w = build_weights(CHAMP, p, v, betas.reindex(columns=midsm), sectors.reindex(midsm),
                      idio_vol.reindex(columns=midsm), vix)
    rr = p.pct_change(fill_method=None).fillna(0.0)
    dw = w.diff().abs().fillna(w.abs())
    adv = (p * v).rolling(21).mean().reindex_like(w).ffill(); adv = adv.fillna(adv.median().median())
    cm = LinearCostModel(half_spread_bps=1.5, impact_coef_bps=10.0)
    rev = (w * rr).sum(axis=1) - cm.charge(dw * 1_000_000.0, adv) / 1_000_000.0

    # Stream 2: residual momentum (all-1500, monthly, ~2.5bp).
    mom = neutral_book_returns(residual_momentum(prices), prices, volume, betas, sectors,
                               rebalance="ME", spread_bps=2.5)

    # Stream 3: live trend overlay net returns.
    tdf = pd.read_csv(ROOT / "results" / "trend_overlay" / "trend_overlay_net.csv",
                      index_col=0, parse_dates=True)
    trend = tdf["trend"]

    streams = {"reversal": rev, "momentum": mom, "trend": trend}
    common = None
    for s in streams.values():
        idx = s.replace(0.0, np.nan).dropna().index
        common = idx if common is None else common.intersection(idx)
    S = pd.DataFrame({k: v.reindex(common).fillna(0.0) for k, v in streams.items()})

    print("\n" + "=" * 74)
    print(f"BOOK LAB  ({common[0].date()}->{common[-1].date()}, per-stream + combined)")
    print("=" * 74)
    print(f"  {'stream':12s} {'annRet':>8s} {'vol':>7s} {'Sharpe':>7s}")
    print("  " + "-" * 40)
    for k in S:
        st = summary_stats(S[k])
        print(f"  {k:12s} {st['ann_return']:>+8.1%} {st['ann_vol']:>7.1%} {st['sharpe']:>+7.2f}")

    print("\n  correlation matrix (the whole point — near-zero = diversifying):")
    corr = S.corr()
    print("  " + "            " + "".join(f"{c:>10s}" for c in corr.columns))
    for r in corr.index:
        print("  " + f"{r:12s}" + "".join(f"{corr.loc[r, c]:>+10.2f}" for c in corr.columns))

    # Inverse-vol combined book (equal risk contribution, ignoring cross-corr = conservative).
    vols = S.std()
    wts = (1.0 / vols) / (1.0 / vols).sum()
    book = (S * wts).sum(axis=1)
    bst = summary_stats(book)
    best = max(summary_stats(S[k])["sharpe"] for k in S)
    print(f"\n  inverse-vol BOOK:  ann {bst['ann_return']:>+.1%}  vol {bst['ann_vol']:.1%}  "
          f"Sharpe {bst['sharpe']:+.2f}   (best single stream {best:+.2f})")
    print(f"  leverage dial (Sharpe held, ex-financing):  2x -> ann {2*bst['ann_return']:+.1%}"
          f"  @ vol {2*bst['ann_vol']:.1%}   3x -> ann {3*bst['ann_return']:+.1%} @ {3*bst['ann_vol']:.1%}")
    print("\n  NB reversal/momentum here are backtest (survivorship-flattered); trend is live-cfg.")
    print("  The DIRECTION — near-zero corr => book Sharpe > any single stream — is the robust part.")


if __name__ == "__main__":
    main()
