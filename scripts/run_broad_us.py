"""Broad US universe test — the enhanced Magic Formula on ALL SEC filers, incl. real small caps.

The S&P 1500 can't reach genuine small/micro caps; EDGAR's ~10k-filer list can. Pipeline:
  1. chunked yfinance download of all filers (fast),
  2. liquidity + history filter on prices alone (drops micro-junk before the slow EDGAR pull),
  3. EDGAR fundamentals for survivors (non-operating filers — ETFs/funds — have no revenue and
     drop out via the rank's valid mask),
  4. market-cap floor, then a cap-ceiling scan of the enhanced strategy.

Reports GROSS and REALISTIC-COST Sharpe per size ceiling, to answer two questions: does the
strategy work in real small caps, and does the edge survive small-cap trading frictions?

CAVEATS: current filers only => survivorship-biased (worst for small caps, so return levels are
overstated); no free sector map for the broad universe, so financials/utilities are NOT excluded
here (a deviation from Greenblatt). Recent-ish window (fundamentals reliable ~2013+).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from data import broad_us_tickers, download_ohlcv, load_fundamentals
from strategies.magic_formula import ENHANCED_ITEMS, EnhancedMagicConfig, enhanced_rank
from strategies.magic_formula.construct import pnl, weights_banded

CEILINGS = {"<$100M": 1e8, "<$300M": 3e8, "<$1B": 1e9, "<$3B": 3e9, "<$10B": 1e10, "none": np.inf}
REAL_COST = dict(half_spread_bps=20.0, impact_coef_bps=30.0)   # small-cap realistic
LOW_COST = dict(half_spread_bps=0.0, impact_coef_bps=0.0)      # gross (frictionless)


def _download_chunked(tickers, start, end, chunk=120, pause=6.0, retries=3):
    """Gentle, throttle-aware chunked download. Yahoo rate-limits bulk yfinance use, so we
    use small batches, pause between them, and retry an empty/failed chunk with exponential
    backoff (an all-empty chunk is the signature of throttling, not of bad tickers)."""
    adj_p, close_p, vol_p = [], [], []
    n_chunks = (len(tickers) - 1) // chunk + 1
    for i in range(0, len(tickers), chunk):
        grp = tickers[i:i + chunk]
        got, p = 0, None
        for attempt in range(retries):
            try:
                p = download_ohlcv(grp, start, end, use_cache=(attempt == 0))
                got = p["adj_close"].dropna(how="all", axis=1).shape[1]
                if got > 0:
                    break
            except Exception:  # noqa: BLE001 - empty batch (throttled); back off and retry
                p = None
            time.sleep(pause * (2 ** attempt))
        if p is not None and got > 0:
            adj_p.append(p["adj_close"]); close_p.append(p["close"]); vol_p.append(p["volume"])
        flag = "" if got > 0 else "  <-- empty (throttled?)"
        print(f"    chunk {i//chunk+1}/{n_chunks} ({len(grp)} tickers, {got} with data){flag}")
        time.sleep(pause)
    if not adj_p:
        raise RuntimeError("no data for any chunk — yfinance is rate-limiting; retry later")
    adj = pd.concat(adj_p, axis=1); close = pd.concat(close_p, axis=1); vol = pd.concat(vol_p, axis=1)
    adj = adj.loc[:, ~adj.columns.duplicated()].sort_index()
    close = close.loc[:, ~close.columns.duplicated()].reindex(adj.index)
    vol = vol.loc[:, ~vol.columns.duplicated()].reindex(adj.index)
    return adj, close, vol


def main(start: str = "2015-01-01", end: str | None = None,
         min_dollar_vol: float = 2e6, min_mcap: float = 50e6, use_graham: bool = True) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg = EnhancedMagicConfig(use_graham=use_graham)

    tickers = broad_us_tickers() + ["SPY"]
    print(f"[1/5] downloading prices for {len(tickers)} filers {start}→{end} (chunked) …")
    adj, close, volume = _download_chunked(sorted(set(tickers)), start, end)
    adj = adj.dropna(how="all", axis=1)
    close = close.reindex(columns=adj.columns)
    volume = volume.reindex(columns=adj.columns)
    spy = adj["SPY"].pct_change(fill_method=None) if "SPY" in adj else None
    print(f"      {adj.shape[1]} tickers returned price data")

    # 2. liquidity + history filter on prices alone (cheap; shrinks the EDGAR pull)
    dollar_vol = (close * volume).median(axis=0)
    enough_hist = adj.notna().sum(axis=0) >= 500  # ~2yr of daily bars
    liquid = adj.columns[(dollar_vol > min_dollar_vol) & enough_hist]
    liquid = [t for t in liquid if t != "SPY"]
    print(f"[2/5] liquid (median $vol>${min_dollar_vol/1e6:.0f}M, >=2yr): {len(liquid)} names")

    adj, close, volume = adj[liquid], close.reindex(columns=liquid), volume.reindex(columns=liquid)

    print(f"[3/5] EDGAR fundamentals for {len(liquid)} names (slow first run) …")
    f = load_fundamentals(liquid, start, end, items=ENHANCED_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)

    # 4. operating-company + market-cap floor. base eligibility.
    base = mcap.notna() & (mcap >= min_mcap)
    print(f"[4/5] with EDGAR mcap>=${min_mcap/1e6:.0f}M: ~{int(base.sum(axis=1).median())} names/day (median)")

    print("[5/5] cap-ceiling scan (gross vs realistic small-cap costs) …\n")
    spy_ret = spy
    rows = []
    for label, ceiling in CEILINGS.items():
        elig = base & (mcap <= ceiling)
        if int(elig.sum(axis=1).median()) < cfg.top_n:
            rows.append((label, int(elig.sum(axis=1).median()), None, None, None, None, None))
            continue
        rank = enhanced_rank(f, mcap, adj, elig, cfg)
        w = weights_banded(rank.where(elig), adj, cfg.rebalance, cfg.top_n, cfg.hold_n)
        gross, _ = pnl(w, adj, volume, close, **LOW_COST)
        net, turn = pnl(w, adj, volume, close, **REAL_COST)
        n_elig = int(elig.sum(axis=1).median())
        med_held = float(mcap.where(w > 0).median(axis=1).median() / 1e9)
        rows.append((label, n_elig, gross, net, turn, med_held, True))

    common = None
    for r in rows:
        if r[6]:
            idx = r[3].replace(0.0, np.nan).dropna().index
            common = idx if common is None else common.intersection(idx)

    print("=" * 92)
    print(f"BROAD US UNIVERSE — enhanced Magic Formula, all SEC filers [survivorship-biased]")
    print(f"  window {start}→{end}, graham={use_graham}, real costs {REAL_COST['half_spread_bps']:.0f}/"
          f"{REAL_COST['impact_coef_bps']:.0f}bps + ADV impact")
    print("=" * 92)
    print(f"  {'max cap':8s} {'#elig':>6s} {'med_held':>9s} {'gross_shp':>10s} {'NET_shp':>8s} "
          f"{'net_ret':>8s} {'turn':>6s}")
    for label, n_elig, gross, net, turn, med_held, ok in rows:
        if not ok:
            print(f"  {label:8s} {n_elig:>6d}  — too few names (<{cfg.top_n})")
            continue
        gs = summary_stats(gross.reindex(common).fillna(0.0))["sharpe"]
        ns = summary_stats(net.reindex(common).fillna(0.0))
        print(f"  {label:8s} {n_elig:>6d} {med_held:>7.2f}B {gs:>+10.2f} {ns['sharpe']:>+8.2f} "
              f"{ns['ann_return']:>+8.2%} {turn:>5.1f}x")
    if spy_ret is not None:
        s = summary_stats(spy_ret.reindex(common).fillna(0.0))
        print(f"  {'SPY':8s} {'—':>6s} {'—':>9s} {'—':>10s} {s['sharpe']:>+8.2f} {s['ann_return']:>+8.2%}")

    out = ROOT / "results" / "broad_us"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({r[0]: r[3] for r in rows if r[6]}).to_csv(out / "broad_us_net.csv")
    print(f"\n  wrote {out}/broad_us_net.csv")


if __name__ == "__main__":
    main(use_graham="--no-graham" not in sys.argv)
