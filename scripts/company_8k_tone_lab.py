"""Track 2 (H.19): does the tone of a company's own 8-Ks between earnings predict its next reaction?

THE QUESTION (the user's: an educated guess at the direction BEFORE the release). Between two
earnings releases a company files 8-Ks on contracts, guidance updates, management changes,
financing, investor days. If the market under-weights that scattered, public material, the tone of
what a company filed during the quarter predicts the sign of its next earnings reaction.
H.17 found the peers' read-through priced in before the report; this checks the company's own
filings, and measures the same pre-move.

EVENTS. S&P 500 members (point in time, at the close before the reaction day), price >= $5.
Earnings release = 8-K Item 2.02; reaction day R from its acceptance time (before 09:30 New York
-> that day, after 16:00 -> the next session, during the session -> dropped); several 2.02 filings
within 7 days: the first counts.

WINDOW. The company's non-earnings 8-Ks (company_8k_fetch.py: Item 2.02 filings and those with
only 5.07/9.01 already excluded) accepted from 7 days after the previous earnings release (the
first week is dominated by release follow-ups) until 16:00 New York on R-1 (so a trade at the
close of R-1 could have used them). At least one filing in the window.

TONE (fixed before the first run). Loughran-McDonald word lists, as in H.18. Window tone = pooled
(positive - negative) / (positive + negative) word counts over all filings in the window; at least
10 tone words. SIGNAL = window tone minus the company's own pooled 8-K tone over the two years
before the window starts (at least 3 filings and 30 tone words), since legal and boilerplate
language makes raw 8-K tone differ widely between companies.

TRADE. Ranked into quintiles against the previous calendar quarter's events (no look-ahead).
Top quintile long, bottom quintile short, each against its sector ETF, close R-1 -> close R
(the reaction). Costs as in H.17: IB commission on $5k a side + 1bp auction slippage a side +
1bp a side on the ETF leg (~6bp a round trip).

PRIMARY TEST (fixed before the first run): net per-event mean, 2005-2022, date-clustered t >= 2
AND mean > 0 in 2020-2022. 2023 onwards is held out unless --holdout is passed.
Descriptive: the reaction and the PRE-MOVE (window start -> close R-1, sector-hedged) by quintile,
i.e. whether the tone was already priced before the release; the number of filings in the window.

Usage:  python scripts/company_8k_tone_lab.py [--holdout]
"""
from __future__ import annotations

import gzip
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data.loaders import download_ohlcv  # noqa: E402
from data.universe import sp500_constituents, sp500_pit_eligible, sp500_pit_universe  # noqa: E402
from earnings_gap_lab import END, SECTOR_ETF, START, _to_et, cik_map  # noqa: E402

EARN_DIR = ROOT / "data" / "cache" / "earnings_8k_full"
LIST_DIR = ROOT / "data" / "cache" / "8k_all"
TEXT_DIR = ROOT / "data" / "cache" / "8k_text"
LM_PATH = ROOT / "data" / "cache" / "lm_dictionary.csv"
TONE_CACHE = ROOT / "data" / "cache" / "8k_tone.parquet"
OUT = ROOT / "results" / "company_8k_tone"

SKIP_DAYS, MIN_WIN_WORDS, BASE_YEARS, BASE_MIN_FILINGS, BASE_MIN_WORDS = 7, 10, 2, 3, 30
ORDER, COMM_SH, COMM_MIN, SLIP_BP, ETF_BP = 5_000.0, 0.0035, 0.35, 1.0, 1.0
MIN_PX = 5.0
DESIGN_END = "2022-12-31"
PERIODS = {"2005-2012": ("2005-01-01", "2012-12-31"), "2013-2019": ("2013-01-01", "2019-12-31"),
           "2020-2022": ("2020-01-01", "2022-12-31")}
HOLDOUT = {"2023+ (holdout)": ("2023-01-01", END)}
WORD = re.compile(r"[A-Z]+")


def t_clustered(x: pd.Series, day: pd.Series) -> float:
    dev = (x - x.mean()).groupby(day.values).sum()
    se = np.sqrt((dev ** 2).sum()) / len(x)
    return x.mean() / se if se > 0 else np.nan


def tones() -> pd.DataFrame:
    """One row per non-earnings 8-K with text: cik, acceptance (ET), positive and negative counts."""
    if TONE_CACHE.exists():
        return pd.read_parquet(TONE_CACHE)
    lm = pd.read_csv(LM_PATH)
    pos = set(lm.loc[lm["Positive"] > 0, "Word"].str.upper())
    neg = set(lm.loc[lm["Negative"] > 0, "Word"].str.upper())
    rows = []
    for k, f in enumerate(sorted(LIST_DIR.glob("*.parquet")), 1):
        cik = int(f.stem)
        for acc, stamp, items in pd.read_parquet(f)[["accessionNumber", "acceptanceDateTime", "items"]].itertuples(index=False):
            p = TEXT_DIR / str(cik) / f"{acc}.txt.gz"
            if not p.exists():
                continue
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                c = Counter(WORD.findall(fh.read().upper()))
            rows.append((cik, _to_et(stamp), str(items), sum(v for w, v in c.items() if w in pos),
                         sum(v for w, v in c.items() if w in neg), sum(c.values())))
        if k % 100 == 0:
            print(f"  tone: {k} companies", flush=True)
    df = pd.DataFrame(rows, columns=["cik", "accepted", "items", "n_pos", "n_neg", "n_words"])
    df.to_parquet(TONE_CACHE)
    return df


def earnings(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Earnings events per cik: acceptance (ET), reaction-day index i (-1 = during the session)."""
    rows = []
    for f in EARN_DIR.glob("*.parquet"):
        cik = int(f.stem)
        for stamp in pd.read_parquet(f)["acceptanceDateTime"]:
            rows.append((cik, _to_et(stamp)))
    ev = pd.DataFrame(rows, columns=["cik", "accepted"]).sort_values(["cik", "accepted"])
    ev = ev[ev.groupby("cik")["accepted"].diff().dt.days.fillna(99) > 7].copy()
    hm = ev["accepted"].dt.hour * 60 + ev["accepted"].dt.minute
    day = ev["accepted"].dt.normalize()
    ev["i"] = np.where(hm < 570, cal.searchsorted(day, side="left"),
                       np.where(hm >= 960, cal.searchsorted(day, side="right"), -1))
    ev["prev"] = ev.groupby("cik")["accepted"].shift(1)
    return ev.dropna(subset=["prev"])


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
    sec = sp500_constituents().set_index("ticker")["sector"].to_dict()
    member = sp500_pit_eligible(cal, list(adj.columns)).astype(bool)
    ep = download_ohlcv(sorted(set(SECTOR_ETF.values()) | {"SPY"}), START, END)["adj_close"].reindex(cal)
    cmap = cik_map()
    tick_of: dict[int, str] = {}
    for t in adj.columns:
        if t in cmap:
            tick_of.setdefault(cmap[t], t)            # dual share classes: one line

    tn = tones()
    print(f"  non-earnings 8-Ks with text: {len(tn)}")
    by_cik = {c: g.sort_values("accepted").reset_index(drop=True) for c, g in tn.groupby("cik")}
    ev = earnings(cal)
    ev = ev[(ev["i"] >= 1) & (ev["i"] < len(cal)) & ev["cik"].isin(tick_of)]

    A, Rw, M = adj.to_numpy(), raw.to_numpy(), member.to_numpy()
    col = {t: j for j, t in enumerate(adj.columns)}
    E = {e: ep[e].to_numpy() for e in ep.columns}

    def hedged(t: str, i0: int, i1: int) -> float:
        e = E[SECTOR_ETF.get(sec.get(t, ""), "SPY")]
        v = [A[i0, col[t]], A[i1, col[t]], e[i0], e[i1]]
        return (v[1] / v[0] - 1) - (v[3] / v[2] - 1) if np.all(np.isfinite(v)) else np.nan

    rows = []
    for cik, prev, i in ev[["cik", "prev", "i"]].itertuples(index=False):
        g = by_cik.get(cik)
        if g is None:
            continue
        t = tick_of[cik]
        j = col[t]
        if not M[i - 1, j] or not np.isfinite(Rw[i - 1, j]) or Rw[i - 1, j] < MIN_PX:
            continue
        w0 = prev + pd.Timedelta(days=SKIP_DAYS)
        w1 = cal[i - 1] + pd.Timedelta(hours=16)
        win = g[(g.accepted >= w0) & (g.accepted < w1)]
        base = g[(g.accepted >= w0 - pd.DateOffset(years=BASE_YEARS)) & (g.accepted < w0)]
        wp, wn = win.n_pos.sum(), win.n_neg.sum()
        bp, bn = base.n_pos.sum(), base.n_neg.sum()
        if len(win) < 1 or wp + wn < MIN_WIN_WORDS or len(base) < BASE_MIN_FILINGS or bp + bn < BASE_MIN_WORDS:
            continue
        i0 = int(cal.searchsorted(w0.normalize(), side="left"))
        rows.append({"day": cal[i], "ticker": t, "n_filings": len(win),
                     "signal": (wp - wn) / (wp + wn) - (bp - bn) / (bp + bn),
                     "reaction": hedged(t, i - 1, i),
                     "pre_move": hedged(t, i0, i - 1) if i0 < i - 1 else np.nan,
                     "px": Rw[i - 1, j]})
    ev2 = pd.DataFrame(rows).dropna(subset=["reaction"]).sort_values("day")

    q = ev2["day"].dt.to_period("Q")
    ev2["quint"] = np.nan
    for per in q.unique():
        prev = ev2.loc[q == per - 1, "signal"]
        if len(prev) < 100:
            continue
        cuts = prev.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        ev2.loc[q == per, "quint"] = np.searchsorted(cuts, ev2.loc[q == per, "signal"]) + 1
    ev2 = ev2.dropna(subset=["quint"])
    if not holdout:
        ev2 = ev2[ev2.day <= DESIGN_END]
    OUT.mkdir(parents=True, exist_ok=True)
    ev2.to_csv(OUT / "events.csv", index=False)
    print(f"  events with a window and a baseline: {len(ev2)} ({len(ev2) / ev2.day.dt.year.nunique():.0f} a year); "
          f"median filings in the window {ev2.n_filings.median():.0f}")
    print(f"  corr(signal, reaction) {ev2.signal.corr(ev2.reaction):+.3f}; "
          f"corr(signal, pre-move) {ev2.signal.corr(ev2.pre_move):+.3f}")

    print(f"\n{'quintile':>9}{'n':>7}{'pre-move bp':>13}{'reaction bp':>13}")
    for k, g in ev2.groupby("quint"):
        print(f"{int(k):>9}{len(g):>7}{g.pre_move.mean() * 1e4:>+13.1f}{g.reaction.mean() * 1e4:>+13.1f}")

    tr = ev2[ev2.quint.isin([1, 5])].copy()
    tr["side"] = np.where(tr.quint == 5, 1.0, -1.0)
    cost = 2 * np.maximum(COMM_MIN, COMM_SH * ORDER / tr["px"]) / ORDER + 2 * SLIP_BP / 1e4 + 2 * ETF_BP / 1e4
    tr["gross"] = tr.side * tr.reaction
    tr["net"] = tr.gross - cost
    tr.to_csv(OUT / "trades.csv", index=False)

    periods = {"design 2005-2022": ("2005-01-01", DESIGN_END), **PERIODS, **(HOLDOUT if holdout else {})}
    print(f"\ntrades = quintile 5 long + quintile 1 short, sector-ETF hedged, close R-1 -> close R")
    print(f"{'period':>18}{'n':>7}{'hit':>6}{'gross bp':>10}{'net bp':>9}{'t(net)':>8}")
    res = {}
    for p, (a, b) in periods.items():
        x = tr[(tr.day >= a) & (tr.day <= b)]
        if len(x) < 30:
            continue
        t = t_clustered(x["net"], x["day"])
        res[p] = (x["net"].mean(), t)
        print(f"{p:>18}{len(x):>7}{(x.gross > 0).mean():>6.0%}{x.gross.mean() * 1e4:>+10.1f}"
              f"{x.net.mean() * 1e4:>+9.1f}{t:>+8.2f}")

    m, t = res["design 2005-2022"]
    m20 = res["2020-2022"][0]
    ok = m > 0 and t >= 2 and m20 > 0
    print(f"\nPRIMARY (net, 2005-2022): mean {m * 1e4:+.1f}bp, t {t:+.2f}; 2020-2022 {m20 * 1e4:+.1f}bp"
          f"  ->  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
