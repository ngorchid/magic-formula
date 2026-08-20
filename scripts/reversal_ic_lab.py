"""Does reversal alpha vary with anything observable AT TRADE TIME? The pre-ML test.

WHY THIS BEFORE ANY MODEL. The proposal is meta-labelling: keep the signal for DIRECTION, add a
model to decide which names to actually trade. That only pays if the signal's information
content genuinely VARIES across names in a way you can see beforehand. If the information
coefficient is flat across every observable, there is no structure to learn and a model would be
fitting noise — expensively, and with a large new silent-failure surface.

So: measure the IC conditioned on candidate features. Flat means stop. Sloped means you have a
feature set, and quite possibly a simple conditional filter that captures most of it with no
model at all.

THE BAR, from the breadth x frequency grid: the best cell is gross Sharpe +0.78 and net +0.06,
so costs drag ~0.72. To clear a 0.5 net Sharpe at that cost, selection must lift gross to ~1.22
— a 56% improvement. Published meta-labelling gains are typically 10-30% in precision, which
usually translates to less in Sharpe. That is the number any result here has to be judged
against, not "is the slope statistically significant".

IC is measured on the DEPLOYED signal (post-neutralisation, post-news-filter, post-inverse-vol),
because that is what selection would actually rank, and against the 5-day forward return, which
is the holding period the grid selected.

⚠ ONE FEATURE IS NOT A FREE PARAMETER: `inv_vol` is already ON in the champion, i.e. the signal
DIVIDES by idiosyncratic vol and so deliberately down-weights the names that move most. If IC
per unit risk is actually HIGHER in high-vol names, that flag is working against the book and
flipping it is a one-line change, not an ML project.

Run: python3 scripts/reversal_ic_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from data import download_ohlcv                           # noqa: E402
from data.universe import (sp1500_constituents, sp1500_sectors,  # noqa: E402
                           sp1500_tickers)
from strategies.equity_mn.neutralize import rolling_beta   # noqa: E402
from reversal_lab import Variant, build_weights            # noqa: E402

FWD = 5          # forward horizon, matching the weekly rebalance the grid selected
NQ = 5           # quintiles


def daily_ic(sig: pd.DataFrame, fwd: pd.DataFrame, mask=None) -> pd.Series:
    """Cross-sectional Spearman IC per day. Spearman, not Pearson: reversal forward returns are
    fat-tailed and a handful of huge moves would otherwise set the number."""
    s = sig.where(mask) if mask is not None else sig
    f = fwd.where(mask) if mask is not None else fwd
    out = {}
    for d in s.index:
        a, b = s.loc[d], f.loc[d]
        ok = a.notna() & b.notna() & (a != 0)
        if ok.sum() >= 20:
            out[d] = a[ok].corr(b[ok], method="spearman")
    return pd.Series(out).dropna()


def report(name: str, ic: pd.Series) -> tuple[float, float]:
    if ic.empty:
        return float("nan"), float("nan")
    t = ic.mean() / ic.std() * np.sqrt(len(ic))
    return float(ic.mean()), float(t)


def main() -> None:
    print("loading mid+small …")
    panel = download_ohlcv(sorted(set(sp1500_tickers() + ["SPY"])), "2011-01-01", None)
    pf = panel["adj_close"].dropna(how="all", axis=1)
    vol = panel["volume"].reindex_like(pf)
    bench = pf["SPY"].pct_change(fill_method=None)
    prices = pf.drop(columns=["SPY"], errors="ignore")
    vol = vol.drop(columns=["SPY"], errors="ignore")
    r = prices.pct_change(fill_method=None)
    betas = rolling_beta(r, bench, 252)
    sectors = sp1500_sectors().reindex(prices.columns)
    idio = r.rolling(20).std()
    import yfinance as yf
    vix = (yf.download("^VIX", start="2011-01-01", auto_adjust=True,
                       progress=False)["Close"].squeeze() / 100.0)
    tier = sp1500_constituents().set_index("ticker")["tier"].reindex(prices.columns)
    ms = [c for c in prices.columns if tier.get(c) in ("mid", "small")]
    prices, vol, idio = prices[ms], vol[ms], idio[ms]

    sig = build_weights(
        Variant("champion", horizons=(1, 3, 5, 10), news_filter=True, smooth=2,
                inv_vol=True, vix_scale=True),
        prices, vol, betas.reindex(columns=ms), sectors.reindex(ms), idio, vix)
    common = sig.replace(0.0, np.nan).dropna(how="all").index
    sig = sig.reindex(common).fillna(0.0)
    px = prices.reindex(common)
    fwd = (px.shift(-FWD) / px - 1.0)

    base_mu, base_t = report("all", daily_ic(sig, fwd))
    print("\n" + "=" * 92)
    print(f"CONDITIONAL IC — deployed signal vs {FWD}-day forward return, mid+small, "
          f"{common[0].date()}..{common[-1].date()}")
    print("=" * 92)
    print(f"  baseline IC (all names, all days): {base_mu:+.4f}   t = {base_t:+.1f}")
    print("\n  If IC is FLAT across a feature, selection on it cannot help. The question is not")
    print(f"  significance -- with ~{len(common):,} days almost anything is significant -- but SPREAD.\n")

    dollar_vol = (px * vol.reindex(common)).rolling(21).mean()
    hl_range = ((panel["high"][ms].reindex(common) - panel["low"][ms].reindex(common))
                / px).rolling(21).mean()
    feats = {
        "idio vol (20d)": idio.reindex(common),
        "dollar volume (21d)": dollar_vol,
        "price level": px,
        "high-low range (21d)": hl_range,
        "|signal| strength": sig.abs().replace(0.0, np.nan),
    }
    print(f"  {'feature':24}" + "".join(f"{'Q' + str(i + 1):>10}" for i in range(NQ))
          + f"{'Q5-Q1':>10}{'spread':>9}")
    print("  " + "-" * 88)
    rows = []
    for fname, fdf in feats.items():
        f = fdf.reindex_like(sig)
        ranks = f.rank(axis=1, pct=True)
        mus = []
        for i in range(NQ):
            lo, hi = i / NQ, (i + 1) / NQ
            m = (ranks > lo) & (ranks <= hi) & (sig != 0)
            mu, _ = report(fname, daily_ic(sig, fwd, m))
            mus.append(mu)
        spread = mus[-1] - mus[0]
        rel = abs(spread) / abs(base_mu) if base_mu else float("nan")
        rows.append({"feature": fname, **{f"Q{i+1}": mus[i] for i in range(NQ)},
                     "Q5_Q1": spread, "rel_to_base": rel})
        print(f"  {fname:24}" + "".join(f"{m:>+10.4f}" for m in mus)
              + f"{spread:>+10.4f}{rel:>8.0%}")

    print("\n  'spread' = |Q5-Q1| as a fraction of the baseline IC. A model can only exploit what")
    print("  varies; a spread well under ~50% of baseline leaves nothing worth a pipeline.")
    print("\n  ⚠ THE BAR: the best grid cell is gross +0.78 / net +0.06. Clearing net 0.5 needs")
    print("  gross ~1.22 -- a 56% lift from selection alone. Judge any slope above against that.")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "reversal_conditional_ic.csv", index=False)
    print(f"\n  wrote {ROOT}/results/reversal_conditional_ic.csv")


if __name__ == "__main__":
    main()
