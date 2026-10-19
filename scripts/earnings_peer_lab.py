"""Peer read-through: do earlier reporters' reactions predict a later reporter's own earnings reaction?

THE QUESTION (pre-event direction, track 1). When peers in the same GICS sub-industry report first,
their results say something about shared demand, costs and pricing. If the market does not fully
price that into the later reporter before it reports, the peers' reactions predict the sign of
the later reporter's own reaction.

THE OBVIOUS OBJECTION, measured rather than assumed: a peer's good news may already lift the later
reporter (and its sector) before it reports. So:
  * every return is measured against the SECTOR ETF, which removes sector-wide moves;
  * the lab reports how much of the signal the later reporter has ALREADY absorbed between the
    peers' reports and the eve of its own (pre-move), next to what is left on the reaction day.

EVENTS. Earnings reaction days R from EDGAR 8-K Item 2.02 (timed by acceptance, as in
earnings_gap_lab.py). Using the actual date assumes it was known in advance; for scheduled
releases it is (confirmed weeks ahead), and unscheduled pre-announcements are a small minority.

SIGNAL (fixed before the first run). For a later reporter i on day R: peers = members of the same
GICS sub-industry (current labels; former members have none and drop out) whose own reaction day
lies in the 10 trading days before R. Peer reaction = the peer's return minus the sector ETF from
the close before its reaction day to the close of it. Signal = the mean over at least 2 peers.

TRADE. sign(signal) x i's reaction return minus the sector ETF, close R-1 -> close R: entered and
exited in the closing auction. Cost: 1bp per side auction slippage + IB commission on $5k + 1bp per
side on the ETF leg (~6bp a round trip).

PRIMARY TEST (fixed before the first run): S&P 500 point in time, net, 2005-2022. PASS = mean > 0
with date-clustered t >= 2 AND > 0 in 2020-2022. Descriptive: the top half by |signal|, the hit
rate, and the pre-move. 2023 onwards is held out unless --holdout is passed.

Usage:  python scripts/earnings_peer_lab.py [--holdout]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data.loaders import download_ohlcv  # noqa: E402
from data.universe import sp500_constituents, sp500_pit_eligible, sp500_pit_universe  # noqa: E402
from earnings_gap_lab import END, SECTOR_ETF, START, cik_map, events  # noqa: E402

LOOKBACK, MIN_PEERS = 10, 2
ORDER, COMM_SH, COMM_MIN, SLIP_BP, ETF_BP = 5_000.0, 0.0035, 0.35, 1.0, 1.0
MIN_PX = 5.0
DESIGN_END = "2022-12-31"
PERIODS = {"2005-2012": ("2005-01-01", "2012-12-31"), "2013-2019": ("2013-01-01", "2019-12-31"),
           "2020-2022": ("2020-01-01", "2022-12-31")}
HOLDOUT = {"2023+ (holdout)": ("2023-01-01", END)}
OUT = ROOT / "results" / "earnings_peer"


def t_clustered(x: pd.Series, day: pd.Series) -> float:
    dev = (x - x.mean()).groupby(day.values).sum()
    se = np.sqrt((dev ** 2).sum()) / len(x)
    return x.mean() / se if se > 0 else np.nan


def main() -> None:
    holdout = "--holdout" in sys.argv
    tickers = sp500_pit_universe(START, END)
    panel = download_ohlcv(tickers, START, END)
    raw = panel["close"].sort_index()
    adj = panel["adj_close"].reindex_like(raw)
    keep = adj.notna().sum() > 300
    adj, raw = adj.loc[:, keep], raw.loc[:, keep]
    cal = adj.index
    print(f"  {adj.shape[1]} stocks with prices (expect ~660)")
    labels = sp500_constituents().set_index("ticker")
    sub = labels["sub_industry"].to_dict()
    sec = labels["sector"].to_dict()
    member = sp500_pit_eligible(cal, list(adj.columns)).astype(bool)
    ep = download_ohlcv(sorted(set(SECTOR_ETF.values()) | {"SPY"}), START, END)["adj_close"].reindex(cal)

    react, _ = events(list(adj.columns), cik_map(), cal)
    react = react[react["ticker"].isin(sub)]
    react["i"] = cal.get_indexer(react["day"])
    A, Rw, M = adj.to_numpy(), raw.to_numpy(), member.to_numpy()
    col = {t: j for j, t in enumerate(adj.columns)}
    E = {e: ep[e].to_numpy() for e in ep.columns}

    def hedged(t, i0, i1):
        e = E[SECTOR_ETF.get(sec[t], "SPY")]
        v = [A[i0, col[t]], A[i1, col[t]], e[i0], e[i1]]
        return (v[1] / v[0] - 1) - (v[3] / v[2] - 1) if np.all(np.isfinite(v)) else np.nan

    react["reaction"] = [hedged(t, i - 1, i) if i >= 1 else np.nan
                         for t, i in zip(react["ticker"], react["i"])]
    react["sub"] = react["ticker"].map(sub)
    by_sub = {s: g for s, g in react.dropna(subset=["reaction"]).groupby("sub")}

    rows = []
    for t, i, s, r in react[["ticker", "i", "sub", "reaction"]].itertuples(index=False):
        if i < LOOKBACK + 1 or not np.isfinite(r) or not M[i - 1, col[t]] or Rw[i - 1, col[t]] < MIN_PX:
            continue
        g = by_sub[s]
        peers = g[(g["i"] >= i - LOOKBACK) & (g["i"] < i) & (g["ticker"] != t)]
        if len(peers) < MIN_PEERS:
            continue
        sig = peers["reaction"].mean()
        first = int(peers["i"].min())
        shares = ORDER / Rw[i - 1, col[t]]
        cost = 2 * max(COMM_MIN, COMM_SH * shares) / ORDER + 2 * SLIP_BP / 1e4 + 2 * ETF_BP / 1e4
        rows.append({"day": cal[i], "ticker": t, "signal": sig, "n_peers": len(peers),
                     "reaction": r, "pre_move": hedged(t, first, i - 1),
                     "gross": np.sign(sig) * r, "net": np.sign(sig) * r - cost})
    tr = pd.DataFrame(rows)
    if not holdout:
        tr = tr[tr.day <= DESIGN_END]
    OUT.mkdir(parents=True, exist_ok=True)
    tr.to_csv(OUT / "trades.csv", index=False)
    print(f"  events with >= {MIN_PEERS} sub-industry peers in the prior {LOOKBACK} sessions: {len(tr)} "
          f"({len(tr) / tr.day.dt.year.nunique():.0f} a year), mean peers {tr.n_peers.mean():.1f}")

    periods = {"design 2005-2022": ("2005-01-01", DESIGN_END), **PERIODS, **(HOLDOUT if holdout else {})}
    big = tr["signal"].abs() >= tr["signal"].abs().median()
    print(f"\n{'subset':16}{'period':>18}{'n':>7}{'hit':>6}{'gross bp':>10}{'net bp':>9}{'t(net)':>8}"
          f"{'corr sig/reaction':>19}{'corr sig/pre-move':>19}")
    res = {}
    for lab, sel in (("all", tr.index == tr.index), ("top half |sig|", big)):
        for p, (a, b) in periods.items():
            x = tr[sel & (tr.day >= a) & (tr.day <= b)]
            if len(x) < 20:
                continue
            t = t_clustered(x["net"], x["day"])
            res[(lab, p)] = (x["net"].mean(), t)
            print(f"{lab:16}{p:>18}{len(x):>7}{(x.gross > 0).mean():>6.0%}{x.gross.mean() * 1e4:>+10.1f}"
                  f"{x.net.mean() * 1e4:>+9.1f}{t:>+8.2f}{x.signal.corr(x.reaction):>+19.3f}"
                  f"{x.signal.corr(x.pre_move):>+19.3f}")
        print()

    m, t = res[("all", "design 2005-2022")]
    m20 = res[("all", "2020-2022")][0]
    ok = m > 0 and t >= 2 and m20 > 0
    print(f"PRIMARY (all, net, 2005-2022): mean {m * 1e4:+.1f}bp, t {t:+.2f}; 2020-2022 {m20 * 1e4:+.1f}bp"
          f"  ->  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
