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
# the three sleeve ledgers for the two-way execution check (read-only). magic-formula defaults to
# this checkout's results/paper/state.json; the other two default to the -live checkouts.
TREND_STATE=C:\Users\Nicolas\PycharmProjects\trend-overlay-live\results\paper\state.json
OPTIONS_STATE=C:\Users\Nicolas\PycharmProjects\options-vrp-live\results\paper\state.json
# optional dead-man's-switch URL, pinged after every completed run (never commit it)
HEARTBEAT_URL=
```

options-vrp reads `IB_RECORDS_DIR` (same default) to apply the execution-id backfill.

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
| **Stock delivered by an options-vrp assignment/exercise** (2026-10-07) | **options-vrp** — when a stock row on the option's underlying lands within `DELIVERY_WINDOW_DAYS` = 3 trading days of the event; that row and every later row on the stock (dividends, the sale) follow. Each delivery row records how it was identified: `delivery_match` = **fingerprint** (exactly multiplier × contracts, at exactly the strike) or **window** (the fallback). If magic-formula has a tagged trade in the same conid → **conflict** (alert). Rows from before the event keep the normal rule |
| Anything else | unassigned → audit alert |

Daily checks (alert email): any unassigned row; IB's NAV bridge (Change in NAV: dividends,
withholding, interest, fees, commissions, transaction taxes, deposits) must equal the sum of the
rows it summarises in the latest statement; corporate actions and option assignments/exercises
are flagged once each (`review_alerted.json`) because they may need a correction in the sleeve's
own ledger.

The account is enrolled in **SYEP** (IB lends out shares), so also tick *Securities
Borrowed/Lent*, *Securities Borrowed/Lent Activity* and *Securities Borrowed/Lent Fee Details*.

## Unseen fee/interest descriptions (2026-10-07)

Every fee/interest/unknown-type description is normalised (month names, dates, amounts, symbols and
ISINs stripped) and kept in `description_registry.json`. The first time an UNSEEN one appears it is
reported once, naming the sleeve it was given — so a reworded market-data fee that slips to "book"
is no longer silent. The very first seed (no `.registry_seeded` marker yet) is silent; a registry
that goes missing **after** that is re-seeded **with an alert**, because descriptions first seen
since the loss can no longer be told apart.

## Two-way execution check (2026-10-07)

Every night, Flex executions are reconciled against the three sleeve ledgers on the IB execution
id (`ibExecID`); Flex is the authority. `tables/exec_reconciliation.csv` lists one row per class:

| Class | Meaning | Alerts |
|---|---|---|
| matched | every id of a ledger fill is in Flex, contract / side / quantity / price agree | no |
| field_mismatch | ids found, facts differ (detail column says which) | **yes** |
| ledger_not_in_flex | a ledger id Flex lacks, on a date some statement covers | **yes** |
| not_yet_covered | the same, but no statement covers that date yet | no |
| not_covered_aged | still not covered after `AGEING_DAYS` = 3 business days (download stuck? window too short?) | **yes** |
| assignment_linked | IB's side of an assignment options-vrp recorded: the delivered stock (100 × contracts at the strike) or the assigned short leg, matched to the sleeve's `ASSIGNED` row | no |
| flex_tagged_unbooked | tagged execution no ledger references — a sleeve failed to book | **yes** |
| flex_untagged | untagged execution — manual trade or tagging failure; assignment/exercise deliveries labelled | **yes** |
| linked_by_tag_only | ledger row with a tag but no ids (VRP self-heal) matched on tag + contract + side + quantity | no |
| fx_sweep_unledgered | magic-formula FX sweeps: tagged, deliberately not in its ledger | no |

Notes:

- **conid, commission, currency** are recorded by all three sleeves on fills since 2026-10-07
  (additive fields; older rows keep working). Where a ledger row has conids they REPLACE the
  description match (no spelling workarounds); commission is compared to Flex `ibCommission`
  (±$0.02) and currency to Flex currency. Older rows fall back to the description (stock symbol with
  yfinance/IB spellings reconciled; future root + expiry; option underlying + expiry + strike +
  right).
- **options-vrp assignment rows**: `ASSIGNED` (the event, not a fill), `ASSIGNED_STOCK_SOLD` and
  `ASSIGNED_LONG_SOLD` (the SAFETY unwind's fills, compared like any fill), and
  `ASSIGNED_CLOSED_OUTSIDE` (unwound by hand; not a fill — the hand-placed orders show as untagged,
  which is expected).
- **Combos at leg level.** PROVISIONAL until verified on real VRP fills: the API returns a
  combo-level execution id besides the legs', which Flex may not report, so ONE missing id on a
  spread whose two legs both matched is treated as that combo-level id.
- **Backfill.** For a VRP tag-only row the real leg ids are written to `tables/exec_backfill.csv`;
  options-vrp applies them to its own ledger at the start of its next run, keeping the superseded
  state on the row. This job never writes a ledger.
- **Earlier days from the IB API.** IB documents `reqExecutions` as returning the current day's
  executions; nothing here relies on it — Flex is the only source for past days.
- A missing or unreadable ledger is reported ("two-way check incomplete"), never treated as empty.

## Realised commission vs the cost model (2026-10-07)

`tables/commission_by_sleeve.csv`: per sleeve and asset class, executions, units (shares /
contracts), total commission and cost per unit, from IB's own records (tagged executions since
2026-10-06; combo rows excluded — their cost is on the legs). For options-vrp the per-contract cost
is compared with the cost guard's assumption (`VRP_ASSUMED_COMMISSION`, default $0.65 = the sleeve's
`COMMISSION_PER_CONTRACT`); above 1.25× over at least 10 contracts it alerts — the cost guard, and so
the edge estimate, would be under-pricing trades.

## Sanitised VRP fills for calibration (`scripts/export_vrp_fills.py`)

Run on the machine holding the archive: writes every options-vrp-tagged, option/combo or
assignment-coded Trade / Order / OptionEAE row with only the agreed fields (ids, tag, conid, side,
quantity, price, commission, timestamps, option fields, codes), the **account id masked everywhere**,
and prints how each IB order appears (record / asset class / level of detail). Send the CSV back to
calibrate the combo-level-id rule; expect the first nightly check to flag VRP rows — use the flags
to calibrate, not to silence it.

## Heartbeat (2026-10-07)

On every completed run: a one-line **success email** (`[IB records] OK <date>: N execution(s)
matched, no issues`, also on days with no activity) or the failure email instead; and, if
`HEARTBEAT_URL` is set, a plain GET to it — no query, no body, no account data — that fails quietly
and never blocks the download. Not sent: on a failed download (so a dead-man's switch fires) and on
`--rebuild`.

Dead-man's switch setup (owner): pick the service (e.g. Healthchecks.io), set its expected schedule
to the `IBRecordsDownload` task's (weekdays 08:30) plus a grace period, put the ping URL in the live
`.env` as `HEARTBEAT_URL` — never in the repo, since anyone holding it can fake a success — and test
it once by skipping a run.

## Alerts (email, failure only)

- download failed (with the IB error code; transient ones say so)
- statement for an unexpected account
- audit trail incomplete: no `orderReference` column (audit trail fields off), a trade since
  2026-10-06 without a strategy tag (manual trade?), or a required section missing
- a failing two-way execution class, a missing ledger, an unseen fee/interest description
