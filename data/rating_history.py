"""Credit rating action history from NRSRO Rule 17g-7(b) XBRL disclosures.

SEC Rule 17g-7(b) obliges every NRSRO to publish its full rating-action history in a
common XBRL format (the "ROCR" R15 taxonomy). That makes downgrade dates available for
free, which no commercial ratings feed does. Coverage starts 2012-06-15; actions are
published on a 12-month delay for issuer-paid ratings (24 months otherwise), so this is
a *backtest* source only — live use needs the agencies' press releases instead.

Because the schema is mandated, one parser handles every agency. Fitch is the only one
of the big three whose file is reachable without an account:

    https://www.fitchratings.com/ratings-history-disclosure   -> accept terms
    https://assets.fitchratings.com/17g7/file                 -> ~180 MB zip, monthly

We read *obligor* ratings (``<OD>``), not instrument ratings (``<ISD>``/``<IND>``): one
record per company is the right granularity for an equity screen, and it sidesteps the
CUSIP redistribution licence attached to the instrument identifiers.

Identifiers are agency-dependent. The schema offers CIK, LEI and CUSIP, but Fitch
populates only ``<LEI>`` (~81% of rows) and leaves ``<CIK>`` empty on ~97% — so the
ticker join runs on normalised company names, with CIK honoured where an agency does
supply it. See ``map_to_tickers`` for why the matching is exact rather than fuzzy.

Fitch rates roughly 360 of the ~825 names that passed through the S&P 500 since 2011;
the remainder genuinely carry no Fitch IDR. Use ``rated_tickers`` to restrict a study
universe, or a "no downgrade" control group silently absorbs every uncovered company.
"""
from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import pandas as pd

# <RAC> rating action classification, per the SEC publication guide.
ACTION_LABELS = {
    "HS": "history-start",   # rating already outstanding when disclosure began
    "NW": "new",
    "UP": "upgrade",
    "DG": "downgrade",
    "WD": "withdrawn-default",
    "WE": "withdrawn-extinguished",
    "WO": "withdrawn-other",
}

# <OSC> SEC category values worth keeping for a corporate-equity screen. The bulk of the
# file by volume is structured finance (RMBS/CMBS/CLO/ABS); dropping it early is what
# makes a 180 MB archive tractable.
CORPORATE_CATEGORIES = ("Corporate", "Financial", "Insurance")

# Canonical notch scale, higher = better. Both the S&P/Fitch and Moody's alphabets map
# onto it so files from different agencies are directly comparable.
_SCALE: list[tuple[int, tuple[str, ...]]] = [
    (21, ("AAA", "AAA")), (20, ("AA+", "AA1")), (19, ("AA", "AA2")), (18, ("AA-", "AA3")),
    (17, ("A+", "A1")), (16, ("A", "A2")), (15, ("A-", "A3")),
    (14, ("BBB+", "BAA1")), (13, ("BBB", "BAA2")), (12, ("BBB-", "BAA3")),
    (11, ("BB+", "BA1")), (10, ("BB", "BA2")), (9, ("BB-", "BA3")),
    (8, ("B+", "B1")), (7, ("B", "B2")), (6, ("B-", "B3")),
    (5, ("CCC+", "CAA1")), (4, ("CCC", "CAA2")), (3, ("CCC-", "CAA3")),
    (2, ("CC", "CA")), (1, ("C", "C")),
    (0, ("D", "RD")), (0, ("SD", "DD")),
]
RATING_NOTCH: dict[str, int] = {sym: n for n, syms in _SCALE for sym in syms}

IG_FLOOR = 12  # BBB- / Baa3: the lowest investment-grade notch

# Suffixes agencies bolt onto a rating that carry no notch information.
_MODIFIER_RE = re.compile(
    r"\s*(\((EXP|P|PI|SF|CW|CWN|CWP)\)|\*[-+]?|/[A-Z0-9-]+|[UEP]$|SF$)+\s*$"
)
_NOT_A_RATING = {"", "NR", "WD", "WR", "NAV", "N/A", "NONE", "UNSOLICITED"}


def normalise_rating(raw: str | None) -> str | None:
    """Strip agency modifiers and upper-case: ``'BBB-(EXP)'`` -> ``'BBB-'``."""
    if raw is None:
        return None
    r = str(raw).strip().upper().replace(" ", "")
    for _ in range(3):  # modifiers stack, e.g. "BB+*-(EXP)"
        r = _MODIFIER_RE.sub("", r)
    return None if r in _NOT_A_RATING else r


def rating_notch(raw: str | None) -> int | None:
    """Map a rating string to the canonical notch scale, or None if unrecognised."""
    r = normalise_rating(raw)
    return RATING_NOTCH.get(r) if r else None


def is_investment_grade(raw: str | None) -> bool | None:
    n = rating_notch(raw)
    return None if n is None else n >= IG_FLOOR


def _local(tag: str) -> str:
    """Strip the XML namespace: ``'{http://...}RAD'`` -> ``'RAD'``."""
    return tag.rpartition("}")[2]


def _text(elem: ET.Element, name: str) -> str | None:
    for child in elem:
        if _local(child.tag) == name:
            # Fitch wraps some obligor names in literal double quotes.
            return (child.text or "").strip().strip('"').strip() or None
    return None


def _obligor_rows(od: ET.Element, agency: str, source: str) -> Iterator[dict]:
    """Flatten one ``<OD>`` obligor tuple into one row per rating action."""
    category = _text(od, "OSC")
    name = _text(od, "OBNAME")
    cik = _text(od, "CIK")
    lei = _text(od, "LEI")
    industry = _text(od, "OIG")

    for ord_elem in od:
        if _local(ord_elem.tag) != "ORD":
            continue
        rating = _text(ord_elem, "R")
        yield {
            "agency": agency,
            "obligor": name,
            "cik": cik.zfill(10) if cik and cik.strip("0") else None,
            "lei": lei,
            "category": category,
            "industry": industry,
            "action_date": _text(ord_elem, "RAD"),
            "action": _text(ord_elem, "RAC"),
            "rating_raw": rating,
            "notch": rating_notch(rating),
            "outlook": _text(ord_elem, "ROL"),
            "rating_type": _text(ord_elem, "RT"),
            "rating_subtype": _text(ord_elem, "RST"),
            "issuer_paid": _text(ord_elem, "IP"),
            "source_file": source,
        }


def parse_instance(fh, agency_hint: str = "", source: str = "") -> list[dict]:
    """Stream one ROCR XBRL instance into obligor rating-action rows.

    Uses iterparse and drops each ``<OD>`` after reading it, so memory stays flat
    regardless of instance size (some agency instances are several hundred MB).
    """
    rows: list[dict] = []
    agency = agency_hint
    rocra: ET.Element | None = None

    for event, elem in ET.iterparse(fh, events=("start", "end")):
        tag = _local(elem.tag)
        if event == "start":
            if tag == "ROCRA":
                rocra = elem
            continue
        if tag == "RAN" and not agency_hint:
            agency = (elem.text or "").strip()
        elif tag == "OD":
            rows.extend(_obligor_rows(elem, agency, source))
            if rocra is not None:
                # Detaches every processed sibling; the parser keeps appending to ROCRA.
                rocra.clear()
    return rows


def parse_rocr_zip(
    path: str | Path,
    categories: tuple[str, ...] | None = CORPORATE_CATEGORIES,
    agency_hint: str = "",
) -> pd.DataFrame:
    """Parse an NRSRO 17g-7(b) zip into a tidy rating-action frame.

    ``categories`` filters on ``<OSC>``; pass None to keep structured finance too.
    """
    rows: list[dict] = []
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        for name in members:
            # The publication guide mirrors the category hierarchy in folder names, so
            # most structured-finance instances can be skipped without being parsed.
            if categories and _skippable(name, categories):
                continue
            with zf.open(name) as fh:
                rows.extend(parse_instance(fh, agency_hint, source=name))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")
    df["action_label"] = df["action"].map(ACTION_LABELS)
    df["is_ig"] = df["notch"].ge(IG_FLOOR).where(df["notch"].notna())
    if categories:
        df = df[df["category"].isin(categories)]
    return df.sort_values(["obligor", "action_date"]).reset_index(drop=True)


ALL_CATEGORIES = ("Financial", "Insurance", "Corporate", "RMBS", "CMBS", "CLO", "CDO",
                  "ABCP", "Other ABS", "Other SFP", "Sovereign", "US Public", "INT Public")

# Agencies encode the identifier in the file name for instrument-level instances
# (the guide's own example is ``ACME-CIK-9876543210-2010-12-31.xml``). Those hold
# <ISD>/<IND> tuples only, so skipping them loses no obligor data.
_INSTRUMENT_RE = re.compile(r"[-/](CUSIP|ISIN|SEDOL|VALOR|WKN|SICC)[-/]", re.I)


def _member_category(member_name: str) -> str | None:
    """SEC category token embedded in the file name, if the agency encodes one.

    Fitch names members ``Fitch_Ratings-<Category>-...``; underscores and spaces are
    treated alike so ``US_Public`` and ``US Public`` both resolve.
    """
    norm = re.sub(r"[_\s]+", " ", member_name).lower()
    for c in sorted(ALL_CATEGORIES, key=len, reverse=True):
        if re.search(rf"[-/ ]{re.escape(c.lower())}[-/ ]", norm):
            return c
    return None


def _skippable(member_name: str, categories: tuple[str, ...]) -> bool:
    """True when the file name alone proves the member holds nothing we want.

    Falls through to False whenever the name is uninformative, so ``<OSC>`` remains
    the authority and an unrecognised naming scheme costs speed, never correctness.
    """
    if _INSTRUMENT_RE.search(member_name):
        return True
    cat = _member_category(member_name)
    return cat is not None and cat not in categories


# Labels that together form ONE economic series: the long-term entity credit opinion.
#
# Moody's splits it across two labels by credit quality — investment-grade issuers
# carry an "LT Issuer Rating" (measured 84.0% IG here), speculative-grade issuers a
# "LT Corporate Family Rating" (0.5% IG). A company falling to junk therefore SWITCHES
# LABEL, and Moody's books that as a withdrawal plus a new rating rather than a
# downgrade. Occidental 2020-03-18 is the canonical case:
#
#     2019-08-01  LT Issuer Rating             DG -> Baa3
#     2020-03-18  LT Issuer Rating             WO -> WR
#     2020-03-18  LT Corporate Family Ratings  NW -> Ba1
#
# There is no DG action on the fall to junk. Transitions must therefore be computed
# across the union of labels, and derived from the notch rather than from the
# agency-reported action code.
ENTITY_SERIES = {
    "Long Term Issuer Default Rating": "LT_ENTITY",  # Fitch
    "LT Issuer Rating": "LT_ENTITY",                 # Moody's, investment grade
    "LT Corporate Family Ratings": "LT_ENTITY",      # Moody's, speculative grade
}

# CAVEAT on Moody's: its *obligor* file is dominated by speculative-grade CFRs (6,558
# entities carry only a CFR against 2,727 carrying only an Issuer Rating), because
# Moody's investment-grade opinions are senior-unsecured ratings held at INSTRUMENT
# level, in the issuer file. Measured against the S&P 500: 92% of Fitch-matched names
# have been IG at some point, but only 53% of Moody's-matched ones. So a Moody's-only
# name often has a history that BEGINS at its fall to junk, and "no downgrade in the
# trailing year" is then an absence of data rather than an observation. Prefer Fitch as
# the primary source; adding Moody's IG side needs <ISD>/<IND> parsing, which this
# module does not do.


def entity_ratings(df: pd.DataFrame, series_map: dict[str, str] = ENTITY_SERIES) -> pd.DataFrame:
    """Keep only long-term entity credit ratings, tagged with a canonical ``series``.

    Everything else is a parallel series on a different scale and would corrupt a
    transition: short-term ratings use their own alphabet (F1/F2/F3), and bank
    deposit, counterparty-risk, insurance-strength and probability-of-default
    ratings ("Ba3-PD") measure different things entirely.
    """
    out = df.copy()
    out["series"] = out["rating_type"].map(series_map)
    return out[out["series"].notna()].copy()


def rating_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive *rated* observations per entity, with the prior rating attached.

    Unmappable ratings are dropped before pairing, which is what lets a withdrawal
    followed by a re-rating under another label read as a single clean transition.
    That silently excludes national-scale ratings ("A3.br", "Aa3.mx") and provisional
    ones ("(P)Baa3") — they are not comparable to the global scale, so dropping them
    is correct rather than merely convenient.
    """
    d = df if "series" in df.columns else entity_ratings(df)
    d = d.dropna(subset=["notch", "action_date"]).copy()
    d["entity"] = d["lei"].fillna(d["obligor"])
    # notch breaks same-date ties so a withdrawal-and-reassign lands in a stable order
    d = d.sort_values(["entity", "series", "action_date", "notch"])
    grp = d.groupby(["entity", "series"], dropna=False)
    d["prev_notch"] = grp["notch"].shift(1)
    d["prev_rating"] = grp["rating_raw"].shift(1)
    d["prev_date"] = grp["action_date"].shift(1)
    return d[d["prev_notch"].notna()].reset_index(drop=True)


def downgrade_events(df: pd.DataFrame) -> pd.DataFrame:
    """Every transition to a lower notch, whatever the agency called the action."""
    t = rating_transitions(df)
    return t[t["notch"].lt(t["prev_notch"])].reset_index(drop=True)


def ig_to_hy_crossings(df: pd.DataFrame) -> pd.DataFrame:
    """Fallen angels: transitions that carry an entity from IG into high yield."""
    t = rating_transitions(df)
    return t[t["prev_notch"].ge(IG_FLOOR) & t["notch"].lt(IG_FLOOR)].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Obligor -> ticker.  Fitch identifies obligors by LEI, and there is no free LEI->ticker
# map, so the join runs on normalised company names against EDGAR's ticker->title list.
# Matching is deliberately EXACT on the normalised key: fuzzy and token-subset variants
# were measured at roughly 50% false positives (American Water Works -> American
# International Group, Cooper Companies -> Cooper Industries), and a false positive here
# stamps a downgrade onto the wrong company, which is worse than no match at all.
# --------------------------------------------------------------------------------------

_LEGAL_SUFFIX = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|COMPANIES|LLC|LP|LLP|PLC|LTD"
    r"|LIMITED|NV|SA|AG|SE|AB|OYJ|THE)\b"
)


def normalise_company(name: str) -> str:
    """Company name -> match key: drop parentheticals, EDGAR state tags, legal suffixes.

    Discriminative words (Group, Holdings, Capital, Financial, Industries, ...) are
    deliberately *kept* — stripping them collapses distinct issuers onto one key.
    """
    s = str(name).upper().replace("&", " AND ")
    s = re.sub(r"\([^)]*\)", " ", s)             # "(The)", "(AIMCO)"
    # EDGAR appends registrant tags: "/DE/", "/NEW", "INC/RI", bare "INC/".
    s = re.sub(r"\s*/[A-Z]{0,4}/?\s*$", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    for _ in range(4):                       # suffixes stack: "Holdings, Inc."
        s = _LEGAL_SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def edgar_ticker_names(tickers: list[str] | None = None) -> dict[str, str]:
    """Ticker -> registrant name, from EDGAR's cached ``company_tickers.json``.

    This is a *current* snapshot: names for companies that have since delisted are
    absent, which is why ``rated_tickers`` should gate any universe built from it.
    """
    from data.fundamentals import _edgar_cik_map

    _edgar_cik_map()  # ensures the cache file exists
    path = Path(__file__).resolve().parent / "edgar_cache" / "company_tickers.json"
    raw = json.loads(path.read_text())
    records = raw.values() if isinstance(raw, dict) else raw
    out = {str(r["ticker"]).upper().replace(".", "-"): r["title"] for r in records}
    return {t: out[t] for t in tickers if t in out} if tickers else out


def map_to_tickers(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Add a ``ticker`` column by exact normalised-name match against ``tickers``.

    Ambiguous keys (several tickers sharing one normalised name) are dropped rather
    than guessed. Rows that match nothing keep ``ticker = NaN``.
    """
    key_to_ticker: dict[str, list[str]] = {}
    for ticker, name in edgar_ticker_names(tickers).items():
        key_to_ticker.setdefault(normalise_company(name), []).append(ticker)
    unique = {k: v[0] for k, v in key_to_ticker.items() if len(v) == 1 and k}

    out = df.copy()
    out["ticker"] = out["obligor"].map(lambda o: unique.get(normalise_company(o)))
    if "cik" in out.columns:  # honour CIK where an agency does populate it
        from data.fundamentals import _edgar_cik_map

        wanted = set(tickers)
        cik_to_ticker: dict[str, str] = {}
        for t, c in _edgar_cik_map().items():
            if t in wanted:  # never resolve outside the requested universe
                cik_to_ticker.setdefault(c, t)
        by_cik = out["cik"].map(cik_to_ticker)
        out["ticker"] = out["ticker"].fillna(by_cik)
    return out


def rated_tickers(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    """Universe members this agency actually rates.

    Essential for an unbiased screen: without it a "no downgrade" control group
    silently absorbs every company the agency never covered. Fitch rates roughly
    360 of the ~825 names that passed through the S&P 500 since 2011 — the rest
    (Apple, Adobe, Applied Materials, ...) carry no Fitch IDR at all.
    """
    mapped = map_to_tickers(entity_ratings(df), tickers)
    return sorted(mapped["ticker"].dropna().unique())


def downgrade_flags(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    tickers: list[str],
    window_months: int = 12,
    fallen_angels_only: bool = False,
) -> pd.DataFrame:
    """Boolean (date x ticker) frame: was this name downgraded in the trailing window?

    Parent and financing subsidiaries are deduplicated (Ford Motor Company and Ford
    Motor Credit both map to F and act on the same dates), so one corporate event
    counts once.
    """
    events = ig_to_hy_crossings(df) if fallen_angels_only else downgrade_events(df)
    ev = map_to_tickers(events, tickers).dropna(subset=["ticker", "action_date"])
    ev = ev[["ticker", "action_date"]].drop_duplicates()

    flags = pd.DataFrame(False, index=calendar, columns=sorted(set(tickers)))
    window = pd.DateOffset(months=window_months)
    for ticker, date in ev.itertuples(index=False):
        if ticker not in flags.columns:
            continue
        mask = (calendar >= date) & (calendar < date + window)
        flags.loc[mask, ticker] = True
    return flags


# --------------------------------------------------------------------------------------
# Self-test: exercises the parser on a synthetic instance so it can be validated before
# the 180 MB download lands.  python -m data.rating_history
# --------------------------------------------------------------------------------------

_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns="http://xbrl.sec.gov/ratings/2015-03-31"
            xmlns:xbrli="http://www.xbrl.org/2003/instance">
  <ROCRA>
    <RAN>Test Ratings Inc</RAN>
    <FCD>2026-08-01</FCD>
    <OD>
      <OSC>Corporate</OSC><OIG>Automotive</OIG>
      <OBNAME>Fallen Motors Inc</OBNAME><CIK>0000037996</CIK>
      <ORD><IP>true</IP><R>BBB-</R><RAD>2019-01-15</RAD><RAC>HS</RAC>
           <ROL>Negative</ROL><RT>Long Term Issuer Default Rating</RT></ORD>
      <ORD><IP>true</IP><R>BB+ (EXP)</R><RAD>2020-03-25</RAD><RAC>DG</RAC>
           <RT>Long Term Issuer Default Rating</RT></ORD>
      <ORD><IP>true</IP><R>BB</R><RAD>2020-09-01</RAD><RAC>DG</RAC>
           <RT>Long Term Issuer Default Rating</RT></ORD>
      <ORD><IP>true</IP><R>BB+</R><RAD>2021-06-01</RAD><RAC>UP</RAC>
           <RT>Long Term Issuer Default Rating</RT></ORD>
      <ORD><IP>true</IP><R>F2</R><RAD>2020-03-25</RAD><RAC>DG</RAC>
           <RT>Short Term Issuer Default Rating</RT></ORD>
    </OD>
    <OD>
      <OSC>Corporate</OSC><OIG>Technology</OIG>
      <OBNAME>Steady Systems Corp</OBNAME><CIK>0000320193</CIK>
      <ORD><IP>true</IP><R>Aa1</R><RAD>2015-02-01</RAD><RAC>HS</RAC>
           <RT>Long Term Issuer Default Rating</RT></ORD>
      <ORD><IP>true</IP><R>Aa2</R><RAD>2022-08-11</RAD><RAC>DG</RAC>
           <RT>Long Term Issuer Default Rating</RT></ORD>
    </OD>
    <OD>
      <OSC>Corporate</OSC><OIG>Energy</OIG>
      <OBNAME>Series Switch Petroleum Corporation</OBNAME><LEI>5493001KJTIIGC8Y1R12</LEI>
      <ORD><IP>true</IP><R>Baa3</R><RAD>2019-08-01</RAD><RAC>DG</RAC>
           <RT>LT Issuer Rating</RT></ORD>
      <ORD><IP>true</IP><R>WR</R><RAD>2020-03-18</RAD><RAC>WO</RAC>
           <RT>LT Issuer Rating</RT></ORD>
      <ORD><IP>true</IP><R>Ba1</R><RAD>2020-03-18</RAD><RAC>NW</RAC>
           <RT>LT Corporate Family Ratings</RT></ORD>
    </OD>
    <OD>
      <OSC>RMBS</OSC><OBNAME>Some Trust 2015-1</OBNAME>
      <ORD><IP>false</IP><R>AAA</R><RAD>2015-01-01</RAD><RAC>NW</RAC></ORD>
    </OD>
  </ROCRA>
</xbrli:xbrl>
"""


def _selftest() -> None:
    import io
    import tempfile

    assert rating_notch("BBB-") == 12 and rating_notch("Baa3") == 12
    assert rating_notch("BB+ (EXP)") == 11 and rating_notch("Ba1") == 11
    assert rating_notch("AA+") == rating_notch("Aa1") == 20
    assert rating_notch("NR") is None and rating_notch("WD") is None
    assert is_investment_grade("BBB-") is True
    assert is_investment_grade("BB+") is False
    print("rating scale            OK")

    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "test-2026-08-01.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("Corporate/Auto/test-2026-08-01.xml", _FIXTURE)

        df = parse_rocr_zip(zpath)
        assert len(df) == 10, f"expected 10 corporate actions, got {len(df)}"
        assert set(df["agency"]) == {"Test Ratings Inc"}
        assert "RMBS" not in set(df["category"])
        assert df.loc[df["obligor"].eq("Fallen Motors Inc"), "cik"].iloc[0] == "0000037996"
        print(f"parse_rocr_zip          OK  ({len(df)} rows, SF filtered)")

        lt = entity_ratings(df)
        assert len(lt) == 9, f"series filter should drop the short-term row, got {len(lt)}"
        assert not lt["rating_raw"].str.startswith("F").any()
        print(f"entity_ratings          OK  ({len(df) - len(lt)} short-term row dropped)")

        dg = downgrade_events(lt)
        assert len(dg) == 4, f"expected 4 notch downgrades, got {len(dg)}"
        # The short-term F2 downgrade must not leak in and corrupt prev_rating.
        assert dg["prev_notch"].notna().all()
        print(f"downgrade_events        OK  ({len(dg)} downgrades)")

        fa = ig_to_hy_crossings(lt)
        # Two: one plain DG, and one recorded as withdrawal + new label with NO DG
        # action anywhere (the Moody's IssuerRating -> CorporateFamilyRating switch).
        assert len(fa) == 2, f"expected 2 fallen angels, got {len(fa)}"
        switch = fa[fa["obligor"].str.startswith("Series Switch")]
        assert len(switch) == 1, "series-switch fallen angel was missed"
        assert switch.iloc[0]["prev_rating"] == "Baa3" and switch.iloc[0]["notch"] == 11
        assert switch.iloc[0]["action"] == "NW", "caught via notch, not the action code"
        fa = fa[fa["obligor"].eq("Fallen Motors Inc")]
        row = fa.iloc[0]
        assert row["obligor"] == "Fallen Motors Inc"
        assert row["prev_rating"] == "BBB-" and row["notch"] == 11
        assert str(row["action_date"].date()) == "2020-03-25"
        print(f"ig_to_hy_crossings      OK  ({row['obligor']} "
              f"{row['prev_rating']}->{normalise_rating(row['rating_raw'])} "
              f"on {row['action_date'].date()})")

    # Name normalisation must keep discriminative words while dropping legal ones.
    assert normalise_company("Kimco Realty OP, LLC") == "KIMCO REALTY OP"
    assert normalise_company("Clorox Company (The)") == normalise_company("CLOROX CO /DE/")
    assert normalise_company("American Water Works Company, Inc.") != \
           normalise_company("American International Group, Inc.")
    assert normalise_company("Cooper Companies, Inc.") != normalise_company("Cooper Industries plc")
    print("normalise_company       OK  (discriminative words preserved)")

    # Streaming parse must not depend on zip packaging.
    rows = parse_instance(io.BytesIO(_FIXTURE.encode()), source="inline")
    assert len(rows) == 11  # 10 corporate + 1 RMBS (no category filter at this level)
    print("parse_instance          OK  (streaming, unfiltered)")
    print("\nall self-tests passed")


if __name__ == "__main__":
    _selftest()
