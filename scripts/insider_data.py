"""Open-market insider trades from the SEC's Form 3/4/5 data sets -> one parquet.

SOURCE. https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets, one zip
per quarter since 2006 (downloaded to data/cache/form345/). Each zip holds the filings
(SUBMISSION), their reporting owners (REPORTINGOWNER) and the non-derivative transactions
(NONDERIV_TRANS), joined by accession number.

KEPT: Form 4 (and 4/A) non-derivative transactions with code P (open-market or private purchase)
or S (open-market or private sale), a positive share count and price. Codes M (option exercise),
A (grant), F (tax withholding), G (gift) and the rest are not discretionary trades and are dropped.

OWNER ROLE: from RPTOWNER_RELATIONSHIP (Officer, Director, TenPercentOwner, Other, combined with
commas); a filing with several owners is kept once per owner. `ceo_cfo` marks an officer title
containing CEO, chief executive, CFO or chief financial.

DATE: FILING_DATE is the day the Form 4 was filed. The data sets carry no acceptance time, so a
user of the data must assume the filing could have come after the close (enter next session).

Run: python scripts/insider_data.py      -> data/cache/insider_trades.parquet
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "cache" / "form345"
OUT = ROOT / "data" / "cache" / "insider_trades.parquet"
CEO_CFO = re.compile(r"\bCEO\b|\bCFO\b|CHIEF EXECUTIVE|CHIEF FINANCIAL", re.I)


def _tsv(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    with z.open(name) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), sep="\t", usecols=cols,
                           dtype=str, quoting=3, on_bad_lines="skip")


def one_quarter(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        sub = _tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                                         "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
        own = _tsv(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK",
                                             "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
        tr = _tsv(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                                            "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                            "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS"])
    sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"])].copy()
    for c in ("TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    tr = tr[(tr["TRANS_SHARES"] > 0) & (tr["TRANS_PRICEPERSHARE"] > 0)]
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
    rel = df["RPTOWNER_RELATIONSHIP"].fillna("")
    out = pd.DataFrame({
        "accession": df["ACCESSION_NUMBER"],
        "filing_date": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
        "trans_date": pd.to_datetime(df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"),
        "issuer_cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce").astype("Int64"),
        "symbol": df["ISSUERTRADINGSYMBOL"].fillna("").str.upper().str.strip(),
        "owner_cik": pd.to_numeric(df["RPTOWNERCIK"], errors="coerce").astype("Int64"),
        "code": df["TRANS_CODE"],
        "shares": df["TRANS_SHARES"],
        "price": df["TRANS_PRICEPERSHARE"],
        "owned_after": df["SHRS_OWND_FOLWNG_TRANS"],
        "officer": rel.str.contains("Officer"),
        "director": rel.str.contains("Director"),
        "ten_pct": rel.str.contains("TenPercentOwner"),
        "ceo_cfo": df["RPTOWNER_TITLE"].fillna("").str.contains(CEO_CFO),
        "amendment": df["DOCUMENT_TYPE"].eq("4/A"),
    })
    out["value"] = out["shares"] * out["price"]
    return out.dropna(subset=["filing_date", "issuer_cik"])


def main() -> None:
    files = sorted(SRC.glob("*_form345.zip"))
    if not files:
        sys.exit(f"no zips in {SRC}")
    parts = []
    for p in files:
        q = one_quarter(p)
        print(f"{p.name}: {len(q):,} P/S rows ({(q['code'] == 'P').sum():,} purchases)")
        parts.append(q)
    df = pd.concat(parts, ignore_index=True)
    # The same transaction can appear in an original and a later quarter's amendment; keep the
    # original (earliest filing) per owner, issuer, date, code, shares and price.
    df = (df.sort_values("filing_date")
            .drop_duplicates(["owner_cik", "issuer_cik", "trans_date", "code", "shares", "price"]))
    df.to_parquet(OUT, index=False)
    print(f"\n{len(df):,} rows, {df['filing_date'].min().date()} .. {df['filing_date'].max().date()} -> {OUT}")


if __name__ == "__main__":
    main()
