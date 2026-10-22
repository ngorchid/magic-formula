# Pre-earnings direction forecast — scheduled task prompt (version 3, 2026-10-22)

Runs as a scheduled **cloud routine** (Claude Code on the web) with the GitHub repo
`ngorchid/llm-forecasts` checked out; the text between the two lines below is its prompt,
**unchanged**. Model: Opus 5.5 (`claude-opus-5-5`) from version 2; version 1 ran on Sonnet 5.5.
Environment `forecasts-full-net` (full network access; the default environment blocks the
earnings calendars). Schedule: weekdays, **20:00 Zurich**: 14:00 New York in a normal week and
15:00 in the daylight-saving gap weeks (late March, late October), so always before the close and
before after-close releases. The evaluation drops any forecast made after its company's release
anyway. If the prompt is ever changed, change the version number in its first line: forecasts from
different versions are scored as separate tests.

Evaluation: `python scripts/llm_forecast_eval.py` in `algo_trading` (Appendix H, H.20).

**Why version 3 (2026-10-22).** Only the count of sources was recorded, so no forecast could
be checked against what it was based on (for example where a quoted consensus figure came
from). Version 3 adds a `sources` column with the URLs read per company; nothing else changes.
The column was added to the header of `predictions.csv` on 2026-10-22; earlier rows leave it
empty.

**Why version 2 (2026-10-20).** Version 1's runs were too shallow to test anything: the
20:00 run on 2026-10-19 took 86 seconds, reused the company list from earlier rows in the file
instead of the calendar (missing KO, PM, VZ, LMT, NDAQ, DGX), searched several companies per
query, and returned 6 forecasts between 0.48 and 0.54 and 8 "no information". Version 2 builds
the calendar fresh from two sources, skips companies already forecast, requires three separate
searches per company before "no information", and runs on Opus.

---

PROMPT VERSION 3. You are making pre-earnings forecasts for a research test. The quality of each
forecast matters more than speed: take the time each company needs. Follow these steps exactly and
do not improvise on the format.

**Step 0 — record the time.** Note the current time in UTC (ISO format, e.g.
2026-10-20T18:00:00Z) as RUN_UTC, and the name of the model you are running as MODEL.

**Step 1 — who reports. Build the list fresh; do not take it from predictions.csv.** From at
least TWO public earnings calendars (for example https://www.nasdaq.com/market-activity/earnings,
https://www.digrin.com/stocks/earnings_calendar/YYYY-MM-DD, Yahoo Finance, Zacks, company investor
relations pages), list the US companies scheduled to report: (a) TODAY after the market close, and
(b) on the NEXT US trading day before the market open. Merge the calendars; where they disagree on
date or timing, check the company's own announcement. Keep only current S&P 500 members (check
every candidate against https://en.wikipedia.org/wiki/List_of_S%26P_500_companies). Call the
count N_ELIGIBLE.

**Step 2 — remove companies already forecast.** Read predictions.csv. Drop every eligible
company that already has a row with the same ticker and release_date (any status). Write no row for
them. Call the remaining list NEW.

**Step 3 — sample.** If NEW has 30 or fewer companies, take all of them. Otherwise sort NEW
alphabetically, set k = ceiling(len(NEW) / 30), and take every k-th ticker starting at position
(day of the month of RUN_UTC, modulo k), counting from 0. Do not choose companies any other way.
Record the rule as SAMPLE_RULE (for example "all" or "k=2, start=1").

**Step 4 — research each sampled company SEPARATELY, one at a time.** For every company run at
least three separate searches or page reads of its own (never one query covering several
companies), one for each of:
  (a) the company's own news, press releases, guidance and SEC filings in the last 30 days;
  (b) analyst estimate revisions, rating changes and previews for this quarter;
  (c) results and commentary from peers, suppliers or customers that have already reported this
      season, and relevant industry data.
Read the pages, not only the search snippets, where a snippet is not enough. Keep a list of
the URL of every page you actually read for this company (not search-result pages): these are
its SOURCES. Use only information public before RUN_UTC; do NOT use the stock's price or
after-hours trading after RUN_UTC.
**Hard rule:** if the company has ALREADY published its results for this quarter (the release is
out before RUN_UTC), give no forecast: STATUS = ALREADY_RELEASED, P_UP empty.
STATUS = NO_INFO is allowed only after all three searches have been done and found nothing usable.

**Step 5 — forecast.** For each company with STATUS = OK, estimate P_UP: the probability that the
stock's return from the last close before the release to the close of the first full trading
session after it is HIGHER than the return of its sector ETF over the same period. In other words,
will the reaction beat the sector? Think about what the market already expects (estimates,
recent stock move, peers' reactions) and what could surprise it, not only whether the company is
doing well. Use 0.50 only when the evidence is truly balanced; be calibrated (a forecast of 0.70
should come true 7 times in 10). Also give CONFIDENCE (low, medium, high) and a REASON of at most
25 words naming the main evidence. N_SOURCES = the number of distinct sources you actually read
for that company, which must equal the number of URLs in its SOURCES.

**Step 6 — write the results.** The repository `llm-forecasts` is checked out in your working
directory. Append one row per sampled company to its file `predictions.csv` (the header row is
already there). Never edit, reorder or delete existing rows. Columns, in this order:

`prompt_version,run_utc,model,ticker,company,release_date,release_timing,p_up,confidence,status,reason,n_sources,n_eligible,sample_rule,sources`

- `prompt_version`: 3.
- `release_date`: the scheduled date (YYYY-MM-DD); `release_timing`: BMO (before open) or AMC
  (after close).
- `status`: OK, ALREADY_RELEASED, or NO_INFO (P_UP empty).
- `n_eligible`: N_ELIGIBLE from step 1 (before removing companies already forecast).
- `sources`: the full URLs from SOURCES, separated by single spaces, the whole field in double
  quotes. Also for NO_INFO (the pages read that had nothing usable) and ALREADY_RELEASED (the
  page showing the release). Empty only on a NO_CALENDAR row.
- Quote any field containing a comma. Tickers as the calendar shows them (BRK.B, not BRK-B).
- If no calendar could be read, write a single row with ticker NONE and status NO_CALENDAR. If
  every eligible company was already forecast, write nothing and do not commit.

Then commit only `predictions.csv` with the message `forecasts <RUN_UTC> v3 (<number> rows)` and
push to `main`. If the push is rejected, pull with rebase and push again; never force-push.

**Step 7 — finish.** Reply with: the number of rows written, N_ELIGIBLE, and the list of eligible
tickers with their timing, marking those skipped as already forecast.

---

## Version 2 (2026-10-20 to 2026-10-21, Opus 5.5), kept for reference

PROMPT VERSION 2. You are making pre-earnings forecasts for a research test. The quality of each
forecast matters more than speed: take the time each company needs. Follow these steps exactly and
do not improvise on the format.

**Step 0 — record the time.** Note the current time in UTC (ISO format, e.g.
2026-10-20T18:00:00Z) as RUN_UTC, and the name of the model you are running as MODEL.

**Step 1 — who reports. Build the list fresh; do not take it from predictions.csv.** From at
least TWO public earnings calendars (for example https://www.nasdaq.com/market-activity/earnings,
https://www.digrin.com/stocks/earnings_calendar/YYYY-MM-DD, Yahoo Finance, Zacks, company investor
relations pages), list the US companies scheduled to report: (a) TODAY after the market close, and
(b) on the NEXT US trading day before the market open. Merge the calendars; where they disagree on
date or timing, check the company's own announcement. Keep only current S&P 500 members (check
every candidate against https://en.wikipedia.org/wiki/List_of_S%26P_500_companies). Call the
count N_ELIGIBLE.

**Step 2 — remove companies already forecast.** Read predictions.csv. Drop every eligible
company that already has a row with the same ticker and release_date (any status). Write no row for
them. Call the remaining list NEW.

**Step 3 — sample.** If NEW has 30 or fewer companies, take all of them. Otherwise sort NEW
alphabetically, set k = ceiling(len(NEW) / 30), and take every k-th ticker starting at position
(day of the month of RUN_UTC, modulo k), counting from 0. Do not choose companies any other way.
Record the rule as SAMPLE_RULE (for example "all" or "k=2, start=1").

**Step 4 — research each sampled company SEPARATELY, one at a time.** For every company run at
least three separate searches or page reads of its own (never one query covering several
companies), one for each of:
  (a) the company's own news, press releases, guidance and SEC filings in the last 30 days;
  (b) analyst estimate revisions, rating changes and previews for this quarter;
  (c) results and commentary from peers, suppliers or customers that have already reported this
      season, and relevant industry data.
Read the pages, not only the search snippets, where a snippet is not enough. Use only information
public before RUN_UTC; do NOT use the stock's price or after-hours trading after RUN_UTC.
**Hard rule:** if the company has ALREADY published its results for this quarter (the release is
out before RUN_UTC), give no forecast: STATUS = ALREADY_RELEASED, P_UP empty.
STATUS = NO_INFO is allowed only after all three searches have been done and found nothing usable.

**Step 5 — forecast.** For each company with STATUS = OK, estimate P_UP: the probability that the
stock's return from the last close before the release to the close of the first full trading
session after it is HIGHER than the return of its sector ETF over the same period. In other words,
will the reaction beat the sector? Think about what the market already expects (estimates,
recent stock move, peers' reactions) and what could surprise it, not only whether the company is
doing well. Use 0.50 only when the evidence is truly balanced; be calibrated (a forecast of 0.70
should come true 7 times in 10). Also give CONFIDENCE (low, medium, high) and a REASON of at most
25 words naming the main evidence. N_SOURCES = the number of distinct sources you actually read
for that company.

**Step 6 — write the results.** The repository `llm-forecasts` is checked out in your working
directory. Append one row per sampled company to its file `predictions.csv` (the header row is
already there). Never edit, reorder or delete existing rows. Columns, in this order:

`prompt_version,run_utc,model,ticker,company,release_date,release_timing,p_up,confidence,status,reason,n_sources,n_eligible,sample_rule`

- `prompt_version`: 2.
- `release_date`: the scheduled date (YYYY-MM-DD); `release_timing`: BMO (before open) or AMC
  (after close).
- `status`: OK, ALREADY_RELEASED, or NO_INFO (P_UP empty).
- `n_eligible`: N_ELIGIBLE from step 1 (before removing companies already forecast).
- Quote any field containing a comma. Tickers as the calendar shows them (BRK.B, not BRK-B).
- If no calendar could be read, write a single row with ticker NONE and status NO_CALENDAR. If
  every eligible company was already forecast, write nothing and do not commit.

Then commit only `predictions.csv` with the message `forecasts <RUN_UTC> v2 (<number> rows)` and
push to `main`. If the push is rejected, pull with rebase and push again; never force-push.

**Step 7 — finish.** Reply with: the number of rows written, N_ELIGIBLE, and the list of eligible
tickers with their timing, marking those skipped as already forecast.

---

## Version 1 (2026-10-18 to 2026-10-19, Sonnet 5.5), kept for reference

PROMPT VERSION 1. You are making pre-earnings forecasts for a research test. Follow these steps
exactly and do not improvise on the format.

**Step 0 — record the time.** Note the current time in UTC (ISO format, e.g.
2026-10-20T18:00:00Z) as RUN_UTC, and the name of the model you are running as MODEL.

**Step 1 — who reports.** Using a public earnings calendar (for example
https://www.nasdaq.com/market-activity/earnings), list the US companies scheduled to report:
(a) TODAY after the market close, and (b) on the NEXT US trading day before the market open.
Keep only current S&P 500 members (check against
https://en.wikipedia.org/wiki/List_of_S%26P_500_companies). Call the count N_ELIGIBLE.

**Step 2 — sample.** If N_ELIGIBLE is 30 or fewer, take all of them. Otherwise sort the tickers
alphabetically, set k = ceiling(N_ELIGIBLE / 30), and take every k-th ticker starting at position
(day of the month of RUN_UTC, modulo k), counting from 0. Do not choose companies any other way.
Record the rule you applied as SAMPLE_RULE (for example "all" or "k=3, start=2").

**Step 3 — research each sampled company.** Look at what was public before RUN_UTC over the last
30 days: the company's own press releases and SEC filings, news coverage, guidance or
pre-announcements, analyst rating or estimate changes, results and commentary from competitors,
suppliers and customers that have already reported this season, and industry data. Do NOT use the
stock's price or after-hours trading after RUN_UTC.
**Hard rule:** if you find that the company has ALREADY published its results for this quarter
(the release is out before RUN_UTC), give no forecast: set STATUS = ALREADY_RELEASED and P_UP empty.

**Step 4 — forecast.** For each company with STATUS = OK, estimate P_UP: the probability that the
stock's return from the last close before the release to the close of the first full trading
session after it is HIGHER than the return of its sector ETF over the same period. In other words,
will the reaction beat the sector? Use 0.50 when you have no view; be calibrated (a forecast of
0.70 should come true 7 times in 10). Also give CONFIDENCE (low, medium, high) and a REASON of at
most 25 words naming the main evidence. Count the distinct sources you relied on as N_SOURCES.

**Step 5 — write the results.** The repository `llm-forecasts` is checked out in your working
directory. Append one row per sampled company to its file `predictions.csv` (the header row is
already there). Never edit, reorder or delete existing rows. Columns, in this order:

`prompt_version,run_utc,model,ticker,company,release_date,release_timing,p_up,confidence,status,reason,n_sources,n_eligible,sample_rule`

- `prompt_version`: 1.
- `release_date`: the scheduled date (YYYY-MM-DD); `release_timing`: BMO (before open) or AMC
  (after close), as the calendar shows it.
- `status`: OK, ALREADY_RELEASED, or NO_INFO (could not find enough to judge; P_UP empty).
- Quote any field containing a comma. Tickers as the calendar shows them (BRK.B, not BRK-B).
- If no calendar could be read, write a single row with ticker NONE and status NO_CALENDAR.

Then commit only `predictions.csv` with the message `forecasts <RUN_UTC> v1 (<number> rows)` and
push to `main`. If the push is rejected, pull with rebase and push again; never force-push.

**Step 6 — finish.** Reply with one line: the number of rows written and N_ELIGIBLE.

---
