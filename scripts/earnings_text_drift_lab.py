"""Stage A: does the tone of the earnings press release predict drift AFTER the first reaction?

THE QUESTION. On the reaction day the market prices the headline numbers. The press release also
carries softer content: how management words the quarter, how confident the outlook sounds.
If investors under-react to that wording, its tone predicts the drift over the following days,
and a trade entered in the closing auction of the reaction day (after the first move is in the
price) still earns it. Published evidence (Demers & Vega 2008; Huang, Teoh & Zhang 2014) found
this in pre-2010 samples; the question is whether it survived.

EVENTS. S&P 500 members (point in time, at the close before the reaction day), price >= $5.
Earnings release = 8-K Item 2.02 on EDGAR; text = its exhibit 99.1 (earnings_text_fetch.py).
Reaction day R from the acceptance time, as in earnings_gap_lab.py (before 09:30 New York -> that
day, after 16:00 -> the next session, during the session -> dropped). Several 2.02 filings within
7 days: the first is kept.

TONE (fixed before the first run). Loughran-McDonald word lists (the finance dictionary; from the
pysentiment2 package, cached in data/cache/lm_dictionary.csv). tone = (positive - negative) /
(positive + negative) word counts. Releases with fewer than 20 tone words are dropped.
SIGNAL = tone SURPRISE: tone minus the mean of the company's own previous (up to 4, at least 2)
releases. Raw tone is mostly a company and industry style; the surprise is the news.

TRADE. Ranked into quintiles against the previous calendar quarter's events (no look-ahead in
the breakpoints). Top quintile long, bottom quintile short, each against its sector ETF: entered at
the close of R, exited at the close of R+5 (PRIMARY), with R+1, R+3, R+10 reported. Costs: IB
commission on $5k a side + 1bp auction slippage a side + 1bp a side on the ETF leg; the short
side pays 0.3%/yr borrow.

PRIMARY TEST (fixed before the first run): net per-event mean of the long and short legs,
2005-2022, date-clustered t >= 2 AND mean > 0 in 2020-2022. 2023 onwards is held out unless
--holdout is passed.
Descriptive: the reaction-day return by quintile (is the tone already priced on R?), the
correlation of tone surprise with the reaction, and the "disagreement" subset (tone and
reaction of opposite sign), where an under-reaction to the text should show up most clearly.

Usage:  python scripts/earnings_text_drift_lab.py [--holdout]
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

LIST_DIR = ROOT / "data" / "cache" / "earnings_8k_full"
TEXT_DIR = ROOT / "data" / "cache" / "earnings_text"
LM_PATH = ROOT / "data" / "cache" / "lm_dictionary.csv"
TONE_CACHE = ROOT / "data" / "cache" / "earnings_tone.parquet"
OUT = ROOT / "results" / "earnings_text_drift"

MIN_TONE_WORDS, HIST_MIN, HIST_MAX = 20, 2, 4
HORIZONS, PRIMARY_H = (1, 3, 5, 10), 5
ORDER, COMM_SH, COMM_MIN, SLIP_BP, ETF_BP, BORROW = 5_000.0, 0.0035, 0.35, 1.0, 1.0, 0.003
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
    """One row per filing with EX-99.1 text: cik, accession, acceptance (ET), word counts, tone."""
    if TONE_CACHE.exists():
        return pd.read_parquet(TONE_CACHE)
    lm = pd.read_csv(LM_PATH)
    pos = set(lm.loc[lm["Positive"] > 0, "Word"].str.upper())
    neg = set(lm.loc[lm["Negative"] > 0, "Word"].str.upper())
    rows = []
    for k, f in enumerate(sorted(LIST_DIR.glob("*.parquet")), 1):
        cik = int(f.stem)
        for acc, stamp in pd.read_parquet(f)[["accessionNumber", "acceptanceDateTime"]].itertuples(index=False):
            p = TEXT_DIR / str(cik) / f"{acc}.txt.gz"
            if not p.exists():
                continue
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                c = Counter(WORD.findall(fh.read().upper()))
            n_pos = sum(v for w, v in c.items() if w in pos)
            n_neg = sum(v for w, v in c.items() if w in neg)
            rows.append((cik, acc, _to_et(stamp), n_pos, n_neg, sum(c.values())))
        if k % 100 == 0:
            print(f"  tone: {k} companies", flush=True)
    df = pd.DataFrame(rows, columns=["cik", "acc", "accepted", "n_pos", "n_neg", "n_words"])
    df["tone"] = (df.n_pos - df.n_neg) / (df.n_pos + df.n_neg).replace(0, np.nan)
    df.to_parquet(TONE_CACHE)
    return df


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

    # ---- events: tone per filing -> ticker, reaction day, tone surprise
    cmap = cik_map()
    tick_of: dict[int, list[str]] = {}
    for t in adj.columns:
        if t in cmap:
            tick_of.setdefault(cmap[t], []).append(t)
    tn = tones()
    print(f"  releases with text: {len(tn)}; with >= {MIN_TONE_WORDS} tone words: "
          f"{int(((tn.n_pos + tn.n_neg) >= MIN_TONE_WORDS).sum())}")
    tn = tn[(tn.n_pos + tn.n_neg) >= MIN_TONE_WORDS].sort_values(["cik", "accepted"])
    # several 2.02 filings within 7 days (amendments, pre-announcements): the first counts
    gap = tn.groupby("cik")["accepted"].diff().dt.days.fillna(99)
    tn = tn[gap > 7].copy()
    roll = tn.groupby("cik")["tone"].transform(lambda s: s.shift(1).rolling(HIST_MAX, min_periods=HIST_MIN).mean())
    tn["surprise"] = tn["tone"] - roll
    tn = tn.dropna(subset=["surprise"])

    hm = tn["accepted"].dt.hour * 60 + tn["accepted"].dt.minute
    day = tn["accepted"].dt.normalize()
    i_pre = cal.searchsorted(day, side="left")
    i_post = cal.searchsorted(day, side="right")
    tn["i"] = np.where(hm < 570, i_pre, np.where(hm >= 960, i_post, -1))
    n_intraday = int((tn["i"] == -1).sum())
    tn = tn[(tn["i"] >= 1) & (tn["i"] < len(cal) - max(HORIZONS))]
    print(f"  events with a tone surprise: {len(tn)} (dropped {n_intraday} released during the session)")

    A, Rw, M = adj.to_numpy(), raw.to_numpy(), member.to_numpy()
    col = {t: j for j, t in enumerate(adj.columns)}
    E = {e: ep[e].to_numpy() for e in ep.columns}

    def hedged(t: str, i0: int, i1: int) -> float:
        e = E[SECTOR_ETF.get(sec.get(t, ""), "SPY")]
        v = [A[i0, col[t]], A[i1, col[t]], e[i0], e[i1]]
        return (v[1] / v[0] - 1) - (v[3] / v[2] - 1) if np.all(np.isfinite(v)) else np.nan

    rows = []
    for cik, i, s, tone in tn[["cik", "i", "surprise", "tone"]].itertuples(index=False):
        for t in tick_of.get(cik, [])[:1]:                 # dual share classes: one line
            j = col[t]
            if not M[i - 1, j] or not np.isfinite(Rw[i - 1, j]) or Rw[i - 1, j] < MIN_PX:
                continue
            r = {"day": cal[i], "ticker": t, "surprise": s, "tone": tone,
                 "reaction": hedged(t, i - 1, i), "px": Rw[i, j]}
            for h in HORIZONS:
                r[f"drift{h}"] = hedged(t, i, i + h)
            rows.append(r)
    ev = pd.DataFrame(rows).dropna(subset=["reaction", f"drift{PRIMARY_H}"]).sort_values("day")

    # ---- quintiles against the previous calendar quarter's surprises
    q = ev["day"].dt.to_period("Q")
    ev["quint"] = np.nan
    for per in q.unique():
        prev = ev.loc[q == per - 1, "surprise"]
        if len(prev) < 100:
            continue
        cuts = prev.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        ev.loc[q == per, "quint"] = np.searchsorted(cuts, ev.loc[q == per, "surprise"]) + 1
    ev = ev.dropna(subset=["quint"])
    if not holdout:
        ev = ev[ev.day <= DESIGN_END]
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT / "events.csv", index=False)
    print(f"  ranked events: {len(ev)} ({len(ev) / ev.day.dt.year.nunique():.0f} a year); "
          f"corr(surprise, reaction) {ev.surprise.corr(ev.reaction):+.3f}")

    print(f"\n{'quintile':>9}{'n':>7}{'reaction bp':>13}" + "".join(f"{f'drift R+{h} bp':>15}" for h in HORIZONS))
    for k, g in ev.groupby("quint"):
        print(f"{int(k):>9}{len(g):>7}{g.reaction.mean() * 1e4:>+13.1f}"
              + "".join(f"{g[f'drift{h}'].mean() * 1e4:>+15.1f}" for h in HORIZONS))

    # ---- trades: top quintile long, bottom quintile short
    tr = ev[ev.quint.isin([1, 5])].copy()
    tr["side"] = np.where(tr.quint == 5, 1.0, -1.0)
    shares = ORDER / tr["px"]
    comm = 2 * np.maximum(COMM_MIN, COMM_SH * shares) / ORDER

    def net(h: int) -> pd.Series:
        borrow = np.where(tr.side < 0, BORROW * h / 252, 0.0)
        return tr.side * tr[f"drift{h}"] - comm - 2 * SLIP_BP / 1e4 - 2 * ETF_BP / 1e4 - borrow

    for h in HORIZONS:
        tr[f"net{h}"] = net(h)
        tr[f"gross{h}"] = tr.side * tr[f"drift{h}"]
    tr.to_csv(OUT / "trades.csv", index=False)

    periods = {"design 2005-2022": ("2005-01-01", DESIGN_END), **PERIODS, **(HOLDOUT if holdout else {})}
    disagree = np.sign(tr.surprise) != np.sign(tr.reaction)
    res = {}
    for lab, sel in (("all", tr.index == tr.index), ("disagreement", disagree)):
        print(f"\n{lab}: trades = quintile 5 long + quintile 1 short, sector-ETF hedged")
        print(f"{'period':>18}{'n':>7}{'hit':>6}" + "".join(f"{f'R+{h} gross/net bp':>22}" for h in HORIZONS)
              + f"{'t(net R+' + str(PRIMARY_H) + ')':>13}")
        for p, (a, b) in periods.items():
            x = tr[sel & (tr.day >= a) & (tr.day <= b)]
            if len(x) < 30:
                continue
            t = t_clustered(x[f"net{PRIMARY_H}"], x["day"])
            res[(lab, p)] = (x[f"net{PRIMARY_H}"].mean(), t)
            print(f"{p:>18}{len(x):>7}{(x[f'gross{PRIMARY_H}'] > 0).mean():>6.0%}"
                  + "".join(f"{x[f'gross{h}'].mean() * 1e4:>+11.1f}/{x[f'net{h}'].mean() * 1e4:>+9.1f}" for h in HORIZONS)
                  + f"{t:>+13.2f}")

    m, t = res[("all", "design 2005-2022")]
    m20 = res[("all", "2020-2022")][0]
    ok = m > 0 and t >= 2 and m20 > 0
    print(f"\nPRIMARY (all, net R+{PRIMARY_H}, 2005-2022): mean {m * 1e4:+.1f}bp, t {t:+.2f}; "
          f"2020-2022 {m20 * 1e4:+.1f}bp  ->  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
