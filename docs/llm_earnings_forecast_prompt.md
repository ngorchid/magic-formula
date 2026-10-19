# Pre-earnings direction forecast — scheduled task prompt (version 1, 2026-10-18)

Runs as a scheduled **cloud routine** (Claude Code on the web) with the GitHub repo
`ngorchid/llm-forecasts` checked out; the text between the two lines below is its prompt,
**unchanged**. Schedule:
weekdays, **20:00 Zurich**: 14:00 New York in a normal week and 15:00 in the daylight-saving gap
weeks (late March, late October), so always before the close and before after-close releases. The
evaluation drops any forecast made after its company's release anyway. If the prompt is ever changed, change the version
number in its first line: forecasts from different versions are scored as separate tests.

Evaluation: `python scripts/llm_forecast_eval.py` in `algo_trading` (Appendix H, LLM track).

---

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
