"""Download the earnings press release (8-K exhibit 99.1) for every S&P 500 earnings filing since 2004.

For the earnings-text labs (Stage A: does the release's content predict drift after the first
reaction?). Point in time by construction: each file is tied to its EDGAR acceptance time.

  1. per company, the 8-K Item 2.02 filings with accession number and acceptance time
     (data/cache/earnings_8k_full/{cik}.parquet; the older earnings_8k cache lacks accessions)
  2. per filing, the index page -> the EX-99.1 document (else the first EX-99.x) -> plain text,
     gzipped to data/cache/earnings_text/{cik}/{accession}.txt.gz

Resumable: anything already on disk is skipped, so rerun after an interruption. Respects SEC's
limit (~8 requests/s) and its User-Agent rule (SEC_USER_AGENT from .env, never printed).

Usage:  python scripts/earnings_text_fetch.py
"""
from __future__ import annotations

import gzip
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data.universe import sp500_pit_universe  # noqa: E402
from earnings_gap_lab import END, START, _ua, cik_map  # noqa: E402

LIST_DIR = ROOT / "data" / "cache" / "earnings_8k_full"
TEXT_DIR = ROOT / "data" / "cache" / "earnings_text"
S = requests.Session()


def get(url: str) -> requests.Response | None:
    for attempt in range(5):
        time.sleep(0.12)
        try:
            r = S.get(url, headers={"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"}, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        time.sleep(2 ** attempt)
    return None


def filings(cik: int) -> pd.DataFrame:
    path = LIST_DIR / f"{cik}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    keys = ("accessionNumber", "form", "filingDate", "acceptanceDateTime", "items")
    r = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if r is None:
        return pd.DataFrame(columns=keys)
    sub = r.json()
    blocks = [sub.get("filings", {}).get("recent", {})]
    for f in sub.get("filings", {}).get("files", []):
        if f.get("filingTo", "9999") >= "2004-08-01":
            rr = get(f"https://data.sec.gov/submissions/{f['name']}")
            if rr is not None:
                blocks.append(rr.json())
    frames = [pd.DataFrame({k: b.get(k, [""] * len(b.get("form", []))) for k in keys})
              for b in blocks if b.get("form")]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keys)
    df = df[(df["form"] == "8-K") & df["items"].astype(str).str.contains("2.02", regex=False)
            & (df["filingDate"] >= "2004-08-01")].reset_index(drop=True)
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def exhibit_text(cik: int, acc: str) -> str | None:
    nod = acc.replace("-", "")
    r = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{acc}-index.htm")
    if r is None:
        return None
    try:
        tables = pd.read_html(io.StringIO(r.text))
    except ValueError:
        return None
    docs = pd.concat([t for t in tables if {"Type", "Document"} <= set(t.columns)], ignore_index=True) \
        if any({"Type", "Document"} <= set(t.columns) for t in tables) else pd.DataFrame()
    if docs.empty:
        return None
    typ = docs["Type"].astype(str).str.upper()
    pick = docs[typ.str.startswith("EX-99.1")]
    if pick.empty:
        pick = docs[typ.str.startswith("EX-99")]
    if pick.empty:
        return None
    name = str(pick.iloc[0]["Document"]).split()[0]
    d = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{name}")
    if d is None:
        return None
    if name.lower().endswith((".htm", ".html")):
        text = BeautifulSoup(d.content, "lxml").get_text(" ")
    else:
        text = d.text
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    cmap = cik_map()
    tickers = [t for t in sp500_pit_universe(START, END) if t in cmap]
    ciks = sorted({cmap[t] for t in tickers})
    print(f"{len(ciks)} S&P 500 filers; step 1: filing lists", flush=True)
    todo = []
    for k, cik in enumerate(ciks, 1):
        for acc in filings(cik)["accessionNumber"]:
            todo.append((cik, acc))
        if k % 100 == 0:
            print(f"  lists {k}/{len(ciks)}", flush=True)
    todo = [(c, a) for c, a in todo if not (TEXT_DIR / str(c) / f"{a}.txt.gz").exists()
            and not (TEXT_DIR / str(c) / f"{a}.none").exists()]
    print(f"step 2: {len(todo)} press releases to fetch", flush=True)
    for k, (cik, acc) in enumerate(todo, 1):
        out = TEXT_DIR / str(cik)
        out.mkdir(parents=True, exist_ok=True)
        text = exhibit_text(cik, acc)
        if text:
            with gzip.open(out / f"{acc}.txt.gz", "wt", encoding="utf-8") as f:
                f.write(text)
        else:
            (out / f"{acc}.none").touch()       # no exhibit found; do not retry every run
        if k % 1000 == 0:
            print(f"  releases {k}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
