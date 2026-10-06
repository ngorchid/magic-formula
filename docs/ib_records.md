# IB records archive — setup

`scripts/download_ib_records.py` downloads IB's own records for the live account (U27760647)
every day via the Flex Web Service and archives them unchanged. Together with the
`orderRef = "<strategy>:<run id>"` tag on every order (since 2026-10-06), this gives a full audit
trail: every execution, fee and cash movement from IB, attributable to a strategy and to the run
(and its `run.log` section) that placed it.

## 1. Create the Activity Flex Query (Client Portal, once)

Performance & Reports → Flex Queries → **Activity Flex Query → +**

**Delivery configuration**

| Setting | Value | Why |
|---|---|---|
| Accounts | U27760647 | the live account |
| Format | **XML** | the script parses XML |
| Period | **Last 7 Calendar Days** | rolling overlap: a missed day or a late correction is still captured |
| Date format / Time format | `yyyyMMdd` / `HHmmss` | sortable; the script's checks assume `yyyyMMdd` |
| Date/time separator | `;` | |
| Profit and Loss | Default | |
| Include Canceled Trades | **Yes** | busted/cancelled executions are part of the trail |
| Include Currency Rates | **Yes** | the daily FX-rates table (this account has no separate section for it) |
| **Include Audit Trail Fields** | **Yes** | **required** — without it IB omits `orderReference`, and trades cannot be attributed to a strategy |
| Display Account Alias in Place of Account ID | No | |
| Breakout by Day | **Yes** | one NAV/summary row per day instead of one per period |

**Sections — tick ONLY these (names exactly as IB lists them), and "Select All" fields inside each**

| Section | What it gives the audit trail |
|---|---|
| Account Information | account identity, base currency, capabilities |
| **Trades** — tick *Executions*, *Orders* and *Closed Lots* | every fill: price, qty, commission, fees, `orderReference`, IB order/exec ids, time |
| Commission Details | exchange/clearing/regulatory fee breakdown per execution |
| **Transaction Fees** | transaction TAXES (UK stamp duty, Italian/French FTT, …) — not in the commission columns |
| **Cash Transactions** | dividends, withholding tax, interest, fees, deposits/withdrawals |
| **Statement of Funds** | complete cash ledger with running balance — ties every cash change to its cause |
| Cash Report | cash by currency (start/end) |
| **Open Positions** | end-of-day positions and cost basis |
| Net Asset Value (NAV) Summary in Base | daily NAV (with Breakout by Day) |
| Change in NAV | NAV bridge: trading, fees, dividends, interest, deposits |
| Realized and Unrealized Performance Summary in Base | P&L per instrument |
| Corporate Actions | splits, mergers, spin-offs |
| Transfers (ACATS, Internal) | security and cash transfers in/out |
| Incoming/Outgoing Trade Transfers | trades moved between accounts/brokers |
| Options, Exercises, Assignments and Expirations | options-vrp lifecycle |
| Interest Accruals | interest earned/charged by currency |
| Change in Dividend Accruals | dividends declared, accrued, reversed |
| Open Dividend Accruals | dividends declared but not yet paid |
| Financial Instrument Information | contract details (conid, multiplier, expiry) for every symbol |
| Currency Conversion Rate | the FX rates IB used, versus base currency |

Recommended additions (reviewed against the first real statement, 2026-10-06):

| Section | Why |
|---|---|
| Prior Period Positions | positions for EVERY day — otherwise a missed run leaves a gap in daily positions |
| Mark-to-Market Performance Summary in Base | IB's daily P&L per instrument = per sleeve (instruments don't overlap) |
| Forex Balances, Forex P/L Details | foreign cash balances and realised/unrealised FX P&L (FX sweeps; tax) |
| Interest Details (Tiers) | how margin interest per currency/tier was computed |
| Complex Position Summary (optional) | options-vrp spreads as combined positions |

This account's Flex UI offers no *Currency Conversion Rate* section and no *Codes* section (a
static legend — the codes are in IB's reference guide). Daily FX rates come from the general
configuration question **"Include currency rates?" → Yes**: every row already carries
`fxRateToBase`, but only the rates table lets a foreign balance be valued on a day with no
transaction in that currency.

**Timestamps are US/Eastern** (e.g. `20261005;143014` = the 20:30 CET trend run). The XML does not
state the zone — convert before comparing with the CET run logs.

All other sections (securities lending, borrow fees, soft dollars, debit card, models, …) do not
apply to this account. Ticking one is harmless — it just comes back empty — so when unsure,
include it: too much is safe, too little is not.

Save it and note the **Query ID** (shown in the query list).

## 2. Create the Flex Web Service token

Performance & Reports → Flex Queries → **Flex Web Service Configuration** (gear icon) → enable →
**Generate New Token**. Choose the longest validity offered and, if you like, restrict it to this
PC's public IP. Tokens **expire**: the script emails `1012 Token has expired` when it does — then
generate a new one and update `.env`.

## 3. Configure `.env` (live checkout, never committed)

```
FLEX_TOKEN=<token>
FLEX_QUERY_ID=<query id>
EXPECT_ACCOUNT=U27760647
```

## 4. Run / schedule

`scripts\download_ib_records.bat` (logs to `logs\ib_records.log`); scheduled task
`IBRecordsDownload` runs it daily. Activity statements cover the previous business day once IB's
overnight processing is done, so the run is in the morning; codes 1005–1008 ("processing
pending") are transient and the next run's 7-day window picks the data up anyway.

## Storage — `C:\Users\Nicolas\IB-records` (override with `IB_RECORDS_DIR`)

```
raw/YYYY/flex_<query>_<from>_<to>_<downloaded>.xml   immutable, exactly as IB sent it (write-once)
manifest.jsonl                                      append-only: file, sha256, bytes, period, account
tables/*.csv                                        derived from ALL raw files each run; disposable
```

- Raw files are never modified; verify any of them against its `sha256` in the manifest.
- `tables/trades.csv` adds `strategy` and `run_id` (parsed from `orderReference`) and keeps the
  latest version of each IB trade id, so IB corrections replace the earlier record.
- `--rebuild` regenerates the tables offline from raw.
- The folder lives outside every checkout so a re-clone can't touch it. **It is not yet backed
  up** — it should be (it is the books of record).

## Attribution of events without an order (owner's rules, 2026-10-06)

Every cash-type row (cash transactions, statement of funds, corporate actions, option
exercises/expiries, transaction taxes, FX transactions) and every trade gets a `sleeve` and a
`category` column; `tables/attribution_summary.csv` totals the base-currency cash ledger by
date × sleeve × category.

| Event | Sleeve |
|---|---|
| Dividends, payment in lieu, withholding tax, corporate actions, option exercise/expiry, futures variation margin, transaction taxes | the sleeve holding the instrument — by the orderReference of the tagged trade in that conid, else by asset class (STK/CASH → magic-formula, FUT → trend-overlay, OPT → options-vrp) |
| SYEP securities-lending income | magic-formula |
| Interest in a foreign currency; FX translation of foreign cash | magic-formula |
| Interest in the base currency (USD) | **book** |
| Market-data subscription fees | **options-vrp** |
| Other account fees | book |
| Deposits / withdrawals | capital (not P&L) |
| Anything else | unassigned → audit alert |

Daily checks (alert email): any unassigned row; IB's NAV bridge (Change in NAV: dividends,
withholding, interest, fees, commissions, transaction taxes, deposits) must equal the sum of the
rows it summarises in the latest statement; corporate actions and option assignments/exercises
are flagged once each (`review_alerted.json`) because they may need a correction in the sleeve's
own ledger.

The account is enrolled in **SYEP** (IB lends out shares), so also tick *Securities
Borrowed/Lent*, *Securities Borrowed/Lent Activity* and *Securities Borrowed/Lent Fee Details*.

## Alerts (email, failure only)

- download failed (with the IB error code; transient ones say so)
- statement for an unexpected account
- audit trail incomplete: no `orderReference` column (audit trail fields off), a trade since
  2026-10-06 without a strategy tag (manual trade?), or a required section missing
