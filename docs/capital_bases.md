# Capital bases per sleeve (return denominators)

Each live sleeve's percentage returns and drawdowns are measured against one documented **return
base**. The values live in `config/capital_bases.json` and are read by `scripts/book_summary.py`.

| Sleeve | Base | Effective | What it represents |
|---|---|---|---|
| magic-formula | **inception NAV** ($50,000) | 2026-08-17 | The account's funded size when magic-formula went live. Read from the sleeve's own `state.json` (`inception_nav`); the config entry documents it |
| trend-overlay | **$75,000** | 2026-09-14 | Notional sizing budget: `BUDGET` 50,000 × `OVERLAY_MULT` 1.5. This is the exposure the sleeve is sized to, not cash set aside |
| options-vrp | **$50,000** | 2026-10-04 | `BUDGET`: the risk base that sizes each spread (max loss ≤ 3% of it per position) |

## What these are NOT

They are **not allocations of the account**. All three sleeves share one account's collateral, so
the bases overlap: they sum to $175,000 against a NAV of about $50,000. That is by design (see the
trading-book documentation, Capital allocation). A "bases + unallocated cash = NAV" identity
therefore does not exist for these numbers.

What does tie is P&L, and the book summary shows it as its own line:

```
NAV = inception capital + magic ledger P&L + trend ledger P&L + options ledger P&L + unattributed
```

The identity holds by definition; the **check** is the size of the unattributed term (FX financing,
FX-sweep commissions, yfinance-vs-IB marking, and anything a ledger failed to book). Above 1% of
NAV (`UNATTRIBUTED_WARN_FRAC`, provisional until its normal size is measured) the email flags it.
A deposit or withdrawal that is not reflected in `BOOK_INCEPTION_CAPITAL` shows up there too.

## What to show outside

For anything outside-facing, the headline is **IBKR's own account-level time-weighted return** (the
`twr` field of the Flex ChangeInNAV section; the book summary shows it as "Account TWR"). Sleeves are
shown as **dollar P&L**; any sleeve percentage is stated against its disclosed base above, and the
email prints that base next to the percentage. NB the daily Flex query covers a rolling 7-day
window, so its TWR is a 7-day figure; an inception-to-date TWR needs a second Flex query over that
period.

## Rule for changes

A base changes only when the thing it represents changes: a new `BUDGET` or `OVERLAY_MULT`, a
capital flow, or a sleeve going live or being retired.

1. Edit `config/capital_bases.json`: the new amount and the effective date.
2. Add a row to the change log below: date, sleeve, old → new, reason.
3. Commit those two together, on their own, like any config change (no analysis in the same
   commit).

An environment override (`TREND_BASE`, `OPTIONS_BASE`) is still possible for a one-off. The book
summary email then prints a "return base OVERRIDDEN" warning until the config and this log match.

Changing a base rescales that sleeve's percentages from the effective date. History is not
restated: compare across a change only in dollars.

## Change log

| Date | Sleeve | Old → New | Reason |
|---|---|---|---|
| 2026-08-17 | magic-formula | — → $50,000 (inception NAV) | Live on real capital |
| 2026-09-14 | trend-overlay | — → $75,000 | Live on real capital; `BUDGET` 50,000 × `OVERLAY_MULT` 1.5 |
| 2026-10-04 | options-vrp | — → $50,000 | Live on real capital; `BUDGET` 50,000 |
| 2026-10-07 | all | (unchanged) | Bases documented and moved from code defaults into `config/capital_bases.json`; owner accepted the NAV bridge (overlapping bases cannot be allocations) |
