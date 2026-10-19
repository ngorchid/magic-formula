"""Score the pre-earnings LLM forecasts (docs/llm_earnings_forecast_prompt.md) against what happened.

For every forecast row (status OK, p_up given) in ~/aktien/trading/llm-forecasts/predictions.csv
(a clone of the GitHub repo the cloud routine appends to; pulled first):

  1. MATCH the release: the company's 8-K Item 2.02 on EDGAR within 3 days of the scheduled date,
     timed by its acceptance time (EDGAR's feed reports it shifted by twice the UTC offset;
     converted as in earnings_gap_lab.py).
  2. VALID only if the forecast was made BEFORE that acceptance time. A forecast made after the
     release is dropped and counted: it could have seen the result.
  3. REACTION session: release before 09:30 New York -> that day; after 16:00 -> the next trading
     day; during the session -> excluded (the trade could not be entered before it).
  4. REACTION = the stock's return minus its sector ETF's, from the close before the reaction
     session to its close (the trade a forecast would drive, entered and exited in the closing
     auction).

Scored per prompt version: hit rate of the forecast direction (p_up above or below 0.50; exactly
0.50 counts as no view), a binomial test against 50%, the Brier score against always saying 0.50
(0.25), and the return from trading the sign, gross and net of ~6bp a round trip, with a t-statistic
clustered by reaction day. Also by confidence. Only the earliest forecast per company and release
counts. Rows still waiting for a release or a price are
reported as PENDING, not scored.

Usage:
  python scripts/llm_forecast_eval.py [path/to/predictions.csv]
  python scripts/llm_forecast_eval.py --selftest     # checks matching/validity on known filings
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data.universe import sp500_constituents  # noqa: E402
from earnings_gap_lab import SECTOR_ETF, _get, _to_et, cik_map  # noqa: E402

DEFAULT = Path.home() / "aktien" / "trading" / "llm-forecasts" / "predictions.csv"   # git clone of ngorchid/llm-forecasts
OUT = ROOT / "results" / "llm_forecast"
MATCH_DAYS, COST = 3, 6e-4


def releases(cik: int) -> list[pd.Timestamp]:
    """Acceptance times (naive ET) of the company's recent 8-K Item 2.02 filings, fetched fresh."""
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    r = sub.get("filings", {}).get("recent", {})
    return sorted(_to_et(a) for f, it, a in zip(r.get("form", []), r.get("items", []),
                                                  r.get("acceptanceDateTime", []))
                  if f == "8-K" and "2.02" in str(it))


def classify(run_utc: pd.Timestamp, release_date: pd.Timestamp, accepted: list[pd.Timestamp]) -> tuple:
    """-> (status, acceptance ET or None). Pure: the self-test exercises it directly."""
    near = [a for a in accepted if abs((a.normalize() - release_date).days) <= MATCH_DAYS]
    if not near:
        return "PENDING" if release_date >= pd.Timestamp.now().normalize() - pd.Timedelta(days=5) \
            else "UNMATCHED", None
    acc = near[0]
    run_et = run_utc.tz_convert("America/New_York").tz_localize(None)
    if run_et >= acc:
        return "AFTER_RELEASE", acc
    hm = acc.hour * 60 + acc.minute
    if 9 * 60 + 30 <= hm < 16 * 60:
        return "INTRADAY_RELEASE", acc
    return "VALID", acc


def reaction_day(acc: pd.Timestamp, cal: pd.DatetimeIndex) -> int | None:
    day = acc.normalize()
    i = cal.searchsorted(day, side="left" if acc.hour * 60 + acc.minute < 9 * 60 + 30 else "right")
    return int(i) if i < len(cal) else None


def t_clustered(x: pd.Series, day: pd.Series) -> float:
    dev = (x - x.mean()).groupby(day.values).sum()
    se = math.sqrt((dev ** 2).sum()) / len(x)
    return x.mean() / se if se > 0 else float("nan")


def binom_p(k: int, n: int) -> float:
    """Two-sided p-value for k hits in n against 50% (normal approximation, continuity-corrected)."""
    if n == 0:
        return float("nan")
    z = (abs(k - n / 2) - 0.5) / math.sqrt(n / 4)
    return math.erfc(max(z, 0) / math.sqrt(2))


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["status"].astype(str).str.upper() == "OK")].copy()
    df["p_up"] = pd.to_numeric(df["p_up"], errors="coerce")
    df = df.dropna(subset=["p_up"])
    if df.empty:
        print("no forecasts with status OK yet")
        return df
    df["run_utc"] = pd.to_datetime(df["run_utc"], utc=True, errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["yf"] = df["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    # One forecast per company and release: a retried push or a company forecast on two runs
    # would otherwise be counted twice. The EARLIEST is kept (furthest from the release).
    n0 = len(df)
    df = df.sort_values("run_utc").drop_duplicates(["prompt_version", "yf", "release_date"], keep="first")
    if len(df) < n0:
        print(f"duplicates dropped (same company and release, later forecast): {n0 - len(df)}")
    cmap = cik_map()
    sectors = sp500_constituents().set_index("ticker")["sector"]
    acc_cache: dict[str, list] = {}
    st, accs = [], []
    for t, run, rel in zip(df["yf"], df["run_utc"], df["release_date"]):
        if t not in cmap or pd.isna(run) or pd.isna(rel):
            st.append("NO_CIK" if t not in cmap else "BAD_ROW")
            accs.append(None)
            continue
        if t not in acc_cache:
            acc_cache[t] = releases(cmap[t])
        s, a = classify(run, rel, acc_cache[t])
        st.append(s)
        accs.append(a)
    df["eval_status"], df["accepted_et"] = st, accs

    v = df[df["eval_status"] == "VALID"]
    if not v.empty:
        etf_of = {t: SECTOR_ETF.get(str(sectors.get(t, "")), "SPY") for t in v["yf"]}
        start = (v["accepted_et"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        px = yf.download(sorted(set(v["yf"]) | set(etf_of.values())), start=start, auto_adjust=True,
                         progress=False)["Close"]
        px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
        cal = px.index
        rr, rd = [], []
        for t, a in zip(v["yf"], v["accepted_et"]):
            i = reaction_day(a, cal)
            ok = i is not None and i >= 1 and t in px and np.isfinite(px[t].iat[i]) and np.isfinite(px[t].iat[i - 1])
            if not ok:
                rr.append(np.nan)
                rd.append(pd.NaT)
                continue
            e = etf_of[t]
            rr.append((px[t].iat[i] / px[t].iat[i - 1] - 1) - (px[e].iat[i] / px[e].iat[i - 1] - 1))
            rd.append(cal[i])
        df.loc[v.index, "reaction"] = rr
        df.loc[v.index, "reaction_day"] = rd
        df.loc[v.index[pd.isna(rr)], "eval_status"] = "PENDING"
    return df


def report(df: pd.DataFrame) -> None:
    print("evaluation status:", df["eval_status"].value_counts().to_dict())
    s = df[(df["eval_status"] == "VALID") & df["reaction"].notna()].copy()
    if s.empty:
        print("nothing scorable yet")
        return
    s["dir"] = np.sign(s["p_up"] - 0.5)
    for ver, g in s.groupby("prompt_version"):
        view = g[g["dir"] != 0]
        hits = int((np.sign(view["reaction"]) == view["dir"]).sum())
        brier = ((g["p_up"] - (g["reaction"] > 0).astype(float)) ** 2).mean()
        ret = view["dir"] * view["reaction"]
        net = ret - COST
        print(f"\nprompt version {ver}: {len(g)} scored forecasts, {len(view)} with a view, "
              f"{g['reaction_day'].nunique()} reaction days")
        print(f"  hit rate {hits / max(len(view), 1):.1%} ({hits}/{len(view)}), binomial p {binom_p(hits, len(view)):.3f}")
        print(f"  Brier {brier:.4f} (always-0.50 baseline 0.2500; lower is better)")
        if len(view) >= 3:
            print(f"  trading the sign: gross {ret.mean() * 1e4:+.1f}bp, net {net.mean() * 1e4:+.1f}bp per event, "
                  f"t {t_clustered(net, view['reaction_day']):+.2f}")
        for c, gc in view.groupby("confidence"):
            h = (np.sign(gc["reaction"]) == gc["dir"]).mean()
            print(f"    confidence {c:7}: n {len(gc):4d}  hit {h:.1%}  net {(gc['dir'] * gc['reaction'] - COST).mean() * 1e4:+.1f}bp")


def selftest() -> None:
    """Classification on real filings: Apple released after the close on 2026-07-30 (16:30 ET)."""
    acc = releases(cik_map()["AAPL"])
    rel = pd.Timestamp("2026-07-30")
    cases = [("forecast at 14:00 ET the same day", "2026-07-30T18:00:00Z", "VALID"),
             ("forecast after the release (21:00 ET)", "2026-07-31T01:00:00Z", "AFTER_RELEASE")]
    ok = True
    for lab, run, want in cases:
        got, a = classify(pd.Timestamp(run), rel, acc)
        print(f"  [{'ok ' if got == want else 'FAIL'}] {lab}: {got} (accepted {a})")
        ok &= got == want
    cal = pd.bdate_range("2026-07-27", "2026-08-07")
    i = reaction_day(pd.Timestamp("2026-07-30 16:30"), cal)
    good = cal[i] == pd.Timestamp("2026-07-31")
    print(f"  [{'ok ' if good else 'FAIL'}] after-close release reacts the next session: {cal[i].date()}")
    i2 = reaction_day(pd.Timestamp("2026-10-13 06:30"), pd.bdate_range("2026-10-12", "2026-10-16"))
    good2 = pd.bdate_range("2026-10-12", "2026-10-16")[i2] == pd.Timestamp("2026-10-13")
    print(f"  [{'ok ' if good2 else 'FAIL'}] pre-market release reacts the same session")
    far, _ = classify(pd.Timestamp("2026-07-30T18:00:00Z"), pd.Timestamp("2026-03-01"), acc)
    print(f"  [{'ok ' if far == 'UNMATCHED' else 'FAIL'}] no filing near an old scheduled date: {far}")
    sys.exit(0 if ok and good and good2 and far == "UNMATCHED" else 1)


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
    path = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), DEFAULT))
    if path == DEFAULT and (path.parent / ".git").exists():
        import subprocess
        subprocess.run(["git", "-C", str(path.parent), "pull", "-q", "--ff-only"], check=False)
    if not path.exists():
        print(f"no predictions file at {path}")
        return
    df = score(pd.read_csv(path))
    if df.empty:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "scored.csv", index=False)
    report(df)
    print(f"\nWrote {OUT / 'scored.csv'}")


if __name__ == "__main__":
    main()
