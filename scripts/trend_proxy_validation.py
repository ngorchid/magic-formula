"""Do the trend overlay's ETF proxies actually describe the futures it trades?

Known limitation #1 of the trend sleeve is that signals come from ETF proxies, not the
futures themselves. That is the same class of gap that just cost the treasury sleeve its
confidence: there the research ran on a fitted CMT curve, and when the position was finally
priced on real contracts the futures delivered HALF the modelled P&L on the 5y and nothing on
the 10y. Nobody has run the equivalent check here, and the sleeve is a candidate for real
money.

TWO QUESTIONS, AND THEY ARE NOT THE SAME.

  SIGNAL fidelity   TSMOM trades the SIGN of a 126/252-day return. A proxy can track poorly
                    in level and still produce an identical signal, because only the sign
                    matters. This is the question that decides whether the strategy TRADES
                    THE RIGHT DIRECTION, and it is the one that matters most.
  RETURN fidelity   the backtested P&L is the proxy's return; the realised P&L is the
                    future's. A persistent drift between them (USO's roll decay against CL,
                    say) means the backtest is measuring a different asset.

A proxy can pass one and fail the other. GLD tracks bullion almost exactly (good on both);
USO holds rolling futures and bleeds in contango, so it can share CL's direction while
earning a materially different return.

WHY THIS MATTERS MORE THAN IT LOOKS: at a $50k budget only copper, AUD and JPY size at all,
and at $75k oil joins them. The markets that actually get a position first are NOT the ones
whose proxies are most trustworthy -- so an average across ten markets would hide the problem.
Everything below is reported per market.

CAVEAT ON THE FUTURES SIDE. yfinance continuous futures differ by instrument: Treasury series
proved BACK-ADJUSTED (no quarterly splices), while CL=F showed roll-window days running ~1.85x
normal volatility. A splice is not a market move, so each series is screened for outliers
beyond OUTLIER_SD trailing deviations and the metrics are reported with those days excluded.

Usage:  python scripts/trend_proxy_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "trend-overlay"))

from trend_overlay.contracts import FUTURES  # noqa: E402

START = "2011-01-01"
END = "2026-09-14"
FUT_YF = {"ES": "ES=F", "ZN": "ZN=F", "GC": "GC=F", "HG": "HG=F", "CL": "CL=F",
          "6E": "6E=F", "6A": "6A=F", "ZB": "ZB=F", "SI": "SI=F", "6J": "6J=F"}
LOOKBACKS = (126, 252)     # the deployed variant D blend
OUTLIER_SD = 8.0
OUT = ROOT / "results" / "trend_proxy"


def clean_returns(px: pd.Series) -> pd.Series:
    """Daily returns with invalid and splice days removed.

    NON-POSITIVE PRICES ARE FATAL AND MUST GO FIRST. CL=F printed -$37.63 on 2020-04-20 when
    the May contract went negative on storage; a percentage return across that is meaningless
    (-306% then +127%) and it corrupts correlation, beta and drift for the whole sample. The
    rolling-sigma outlier screen does NOT catch it, because the event inflates the very
    standard deviation it is measured against -- so the sign test has to be explicit.
    """
    px = px.where(px > 0)
    r = px.pct_change()
    r = r.mask(px.isna() | px.shift(1).isna())
    z = (r.abs() / r.rolling(60).std())
    return r.mask(z > OUTLIER_SD)


def main() -> None:
    proxies = sorted({f.proxy_etf for f in FUTURES})
    futs = sorted({FUT_YF[f.symbol] for f in FUTURES if f.symbol in FUT_YF})
    raw = yf.download(proxies + futs, start=START, end=END, auto_adjust=True,
                      progress=False, group_by="column")["Close"].sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()

    rows = []
    for f in FUTURES:
        if f.symbol not in FUT_YF:
            continue
        pcol, fcol = f.proxy_etf, FUT_YF[f.symbol]
        if pcol not in raw or fcol not in raw:
            continue
        px_p, px_f = raw[pcol].dropna(), raw[fcol].dropna()
        idx = px_p.index.intersection(px_f.index)
        if len(idx) < 750:
            continue
        px_p, px_f = px_p.reindex(idx), px_f.reindex(idx)
        rp, rf = clean_returns(px_p), clean_returns(px_f)
        both = pd.concat([rp.rename("p"), rf.rename("f")], axis=1).dropna()

        # Staleness depresses correlation mechanically: CPER prints an unchanged close on
        # ~10% of days against HG=F's 0.6%, so its low corr is a LIQUIDITY artifact, not a
        # tracking failure. Reported rather than silently corrected.
        stale = float((both["p"] == 0).mean())
        corr = float(both["p"].corr(both["f"]))
        beta = float(np.polyfit(both["f"], both["p"], 1)[0])
        drift = float((both["p"].mean() - both["f"].mean()) * 252)
        volr = float(both["p"].std() / both["f"].std())

        # SIGNAL agreement: does the proxy give the same TSMOM sign as the future?
        agree = {}
        for lb in LOOKBACKS:
            sp = np.sign(px_p.pct_change(lb))
            sf = np.sign(px_f.pct_change(lb))
            c = pd.concat([sp.rename("p"), sf.rename("f")], axis=1).dropna()
            c = c[(c["p"] != 0) & (c["f"] != 0)]
            agree[lb] = float((c["p"] == c["f"]).mean()) if len(c) else np.nan

        rows.append({"market": f.market, "proxy": pcol, "fut": f.symbol, "n": len(both),
                     "corr": corr, "stale": stale, "beta": beta, "vol_ratio": volr,
                     "drift_pa": drift, "sig126": agree[126], "sig252": agree[252]})

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "proxy_fidelity.csv", index=False)

    print("=" * 100)
    print("PROXY vs FUTURE, per market  (2011-2026, splice/outlier days removed)")
    print("=" * 100)
    print(f"  {'market':12}{'proxy':7}{'fut':5}{'corr':>7}{'stale':>7}{'beta':>7}{'vol rat':>9}"
          f"{'drift p.a.':>12}{'sign agree 126d':>17}{'252d':>8}")
    for _, r in d.sort_values("sig252").iterrows():
        flag = "  <-- CHECK" if (r["sig252"] < 0.90 or abs(r["drift_pa"]) > 0.05) else ""
        print(f"  {r['market']:12}{r['proxy']:7}{r['fut']:5}{r['corr']:>7.2f}"
              f"{r['stale']:>6.0%}{r['beta']:>7.2f}{r['vol_ratio']:>9.2f}{r['drift_pa']:>+11.1%}"
              f"{r['sig126']:>16.0%}{r['sig252']:>8.0%}{flag}")

    print("\n  corr/beta/vol_ratio: does the proxy MOVE like the future (return fidelity)?")
    print("  drift p.a.: proxy annual return minus future's. Non-zero = a different asset.")
    print("  sign agree: share of days the 126/252d TSMOM SIGN matches. This is the one that")
    print("  decides whether the strategy trades the right DIRECTION, and 252d is the")
    print("  dominant leg of the deployed 126+252 blend.")

    print("\n" + "=" * 100)
    print("WHAT THIS MEANS AT YOUR BUDGET")
    print("=" * 100)
    order = {"copper": "$50k", "fx_aud": "$50k", "fx_jpy": "$50k",
             "equity_us": "$75k", "oil": "$75k", "rates_10y": "$300k"}
    print("  Markets that actually size first, with their proxy's 252d signal agreement:")
    for m, b in order.items():
        row = d[d.market == m]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"    {b:6} {m:12} {r['proxy']:6} sign agree {r['sig252']:.0%}, "
              f"drift {r['drift_pa']:+.1%} p.a.")
    print("\n  A market that only appears at $300k is one the backtest leans on and the live")
    print("  book does not hold -- worth knowing which side of that line the good proxies sit.")
    print(f"\nWrote {OUT / 'proxy_fidelity.csv'}")


if __name__ == "__main__":
    main()
