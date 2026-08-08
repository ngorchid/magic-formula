"""OPRA DBN -> Parquet, with a DATE-AWARE instrument_id mapping.

Databento recycles instrument_ids: 80.8% of ids here map to more than one contract over
time (up to 221). The mapping is therefore (instrument_id, [start_date, end_date)) ->
(expiry, cp, strike), and bars must be joined on id AND date. Collapsing to one row per id
assigns recycled ids the wrong contract -- which shows up as negative days-to-expiry.

OSI symbol layout: 'SPX   180420P01700000' -> expiry 2018-04-20, Put, strike 1700.000
"""
import sys, time
import numpy as np, pandas as pd, databento as db
from pathlib import Path

SP = Path(sys.argv[1])
store = db.DBNStore.from_file(SP / "opra/opra-pillar-20130401-20260806.ohlcv-1d.dbn.zst")

spans = []
for osi, sp_list in store.metadata.mappings.items():
    body = osi[6:]
    if len(body) < 15:
        continue
    try:
        exp = pd.Timestamp(f"20{body[0:2]}-{body[2:4]}-{body[4:6]}")
        cp, strike = body[6], int(body[7:15]) / 1000.0
    except Exception:
        continue
    for s in sp_list:
        sym = s.get("symbol")
        if sym and sym.isdigit():
            spans.append((int(sym), pd.Timestamp(s["start_date"]), pd.Timestamp(s["end_date"]),
                          exp, cp, strike))
sm = pd.DataFrame(spans, columns=["instrument_id", "start", "end", "expiry", "cp", "strike"])
sm.to_parquet(SP / "opra/instrument_spans.parquet", index=False)
print(f"[map] {len(sm):,} spans over {sm.instrument_id.nunique():,} ids", flush=True)

arr = store.to_ndarray()
bars = pd.DataFrame({
    "date": pd.to_datetime(arr["ts_event"], utc=True).tz_localize(None).normalize(),  # already the session date
    "instrument_id": arr["instrument_id"].astype("int64"),
    "open": arr["open"]/1e9, "high": arr["high"]/1e9, "low": arr["low"]/1e9,
    "close": arr["close"]/1e9, "volume": arr["volume"].astype("int64"),
})
print(f"[read] {len(bars):,} bars", flush=True)

t0 = time.time()
m = bars.merge(sm, on="instrument_id", how="left")
m = m[(m.date >= m.start) & (m.date < m.end)]
m = m.drop(columns=["start", "end"]).drop_duplicates(["date", "instrument_id"])
print(f"[join] {len(m):,} rows kept in {time.time()-t0:.0f}s "
      f"(dropped {len(bars)-len(m):,} unmatched)", flush=True)

m["dte"] = (m.expiry - m.date).dt.days
bad = (m.dte < 0).sum()
print(f"[check] negative DTE rows: {bad:,}  ({bad/len(m):.4%})", flush=True)
m = m[m.dte >= 0]
m.to_parquet(SP / "opra/bars.parquet", index=False)
print(f"[done] {len(m):,} rows -> bars.parquet  {m.date.min().date()}..{m.date.max().date()}", flush=True)
