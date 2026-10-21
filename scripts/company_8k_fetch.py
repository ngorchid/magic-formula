"""Download the text of every NON-earnings 8-K filed by S&P 500 companies since 2004 (track 2).

For the between-earnings lab: does what a company itself files between two earnings releases
(contracts, guidance updates, management changes, investor-day material, financing) predict the
direction of its next earnings reaction? Point in time by construction: each file is tied to its
EDGAR acceptance time.

  1. per company, all 8-K / 8-K/A filings with accession number, acceptance time and items
     (data/cache/8k_all/{cik}.parquet). Dropped: filings carrying Item 2.02 (the earnings releases
     themselves; earnings_text_fetch.py has them) and filings whose only items are 5.07
     (shareholder vote results) and/or 9.01 (exhibit list).
  2. per filing, the index page -> the EX-99.1 document (else the first EX-99.x, else the
     primary 8-K document) -> plain text, gzipped to data/cache/8k_text/{cik}/{accession}.txt.gz

Resumable: anything on disk is skipped. Three worker threads share one rate limiter that keeps the
whole download under SEC's limit (8 requests/s against SEC's 10); User-Agent from .env.

Usage:  python scripts/company_8k_fetch.py
"""
from __future__ import annotations

import gzip
import io
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data.universe import sp500_pit_universe  # noqa: E402
from earnings_gap_lab import END, START, cik_map  # noqa: E402
import earnings_text_fetch  # noqa: E402

_LOCK = threading.Lock()
_NEXT = [0.0]
RATE = 8.0                                   # requests per second, all threads together
WORKERS = 3


def get(url: str):
    """earnings_text_fetch.get behind a shared rate limiter (its own 0.12s sleep stays too)."""
    with _LOCK:
        now = time.monotonic()
        wait = _NEXT[0] - now
        _NEXT[0] = max(now, _NEXT[0]) + 1.0 / RATE
    if wait > 0:
        time.sleep(wait)
    return earnings_text_fetch.get(url)

LIST_DIR = ROOT / "data" / "cache" / "8k_all"
TEXT_DIR = ROOT / "data" / "cache" / "8k_text"
ROUTINE_ONLY = {"5.07", "9.01"}


def filings(cik: int) -> pd.DataFrame:
    path = LIST_DIR / f"{cik}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    keys = ("accessionNumber", "form", "filingDate", "acceptanceDateTime", "items", "primaryDocument")
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
    items = df["items"].astype(str).str.split(",").apply(lambda xs: {x.strip() for x in xs if x.strip()})
    df = df[df["form"].isin(["8-K", "8-K/A"]) & (df["filingDate"] >= "2004-08-01")
            & ~items.apply(lambda s: "2.02" in s) & ~items.apply(lambda s: bool(s) and s <= ROUTINE_ONLY)]
    df = df.drop_duplicates("accessionNumber").reset_index(drop=True)
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def filing_text(cik: int, acc: str, primary: str) -> str | None:
    nod = acc.replace("-", "")
    name = None
    r = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{acc}-index.htm")
    if r is not None:
        try:
            tables = [t for t in pd.read_html(io.StringIO(r.text)) if {"Type", "Document"} <= set(t.columns)]
        except ValueError:
            tables = []
        if tables:
            docs = pd.concat(tables, ignore_index=True)
            typ = docs["Type"].astype(str).str.upper()
            pick = docs[typ.str.startswith("EX-99.1")]
            if pick.empty:
                pick = docs[typ.str.startswith("EX-99")]
            if not pick.empty:
                name = str(pick.iloc[0]["Document"]).split()[0]
    name = name or (primary or None)
    if not name:
        return None
    d = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{name}")
    if d is None:
        return None
    text = BeautifulSoup(d.content, "lxml").get_text(" ") if name.lower().endswith((".htm", ".html")) else d.text
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    cmap = cik_map()
    ciks = sorted({cmap[t] for t in sp500_pit_universe(START, END) if t in cmap})
    print(f"{len(ciks)} S&P 500 filers; step 1: filing lists", flush=True)
    todo = []
    for k, cik in enumerate(ciks, 1):
        for acc, prim in filings(cik)[["accessionNumber", "primaryDocument"]].itertuples(index=False):
            todo.append((cik, acc, prim))
        if k % 100 == 0:
            print(f"  lists {k}/{len(ciks)}", flush=True)
    todo = [x for x in todo if not (TEXT_DIR / str(x[0]) / f"{x[1]}.txt.gz").exists()
            and not (TEXT_DIR / str(x[0]) / f"{x[1]}.none").exists()]
    print(f"step 2: {len(todo)} filings to fetch", flush=True)
    def one(job):
        cik, acc, prim = job
        out = TEXT_DIR / str(cik)
        out.mkdir(parents=True, exist_ok=True)
        text = filing_text(cik, acc, prim)
        if text:
            tmp = out / f"{acc}.txt.gz.part"          # renamed when complete: a kill never leaves
            with gzip.open(tmp, "wt", encoding="utf-8") as f:   # a truncated file that a rerun would skip
                f.write(text)
            tmp.rename(out / f"{acc}.txt.gz")
        else:
            (out / f"{acc}.none").touch()

    with ThreadPoolExecutor(WORKERS) as pool:
        for k, _ in enumerate(pool.map(one, todo), 1):
            if k % 1000 == 0:
                print(f"  filings {k}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
