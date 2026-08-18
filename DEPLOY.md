# Windows deployment — Magic Formula paper trading

Runs `scripts/run_paper.py` every weekday at **16:30 CET** on the Windows box, alongside
the contract strategy. US names fill same-day (RTH).

**Why 16:30 and not later:** the three strategies share one IB account and margin is
claimed FIRST-COME-FIRST-SERVED — whoever runs last is the one the account-wide
liquidity floor blocks. This is the highest-Sharpe sleeve (0.96 vs trend 0.74, VRP 0.52)
and it is the collateral the other two borrow against, so it must go first. Order is
magic-formula 16:30 -> trend 18:00 -> options-vrp 21:30 CET.

16:30 CET is 10:30 ET, an hour after the US open (liquid, spreads settled) and — unlike
the previous 20:00 — while the European market is still open, so EU names now fill
same-day instead of queueing to the next EU open. All prices/fundamentals come from **yfinance**; **IB** is used only to place orders.

## 1. Clone

The Windows box already has the `bitbucket-picard` SSH alias (from the contract strategy):

```bat
cd C:\trading
git clone git@bitbucket-picard:picard_capital/magic-formula.git
cd magic-formula
git checkout live
```

## 2. Python env (lean — no cvxpy/vectorbt)

```bat
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements-paper.txt
```

## 3. Configure `.env`

```bat
copy .env.example .env
notepad .env
```

Fill in:
- `IB_PORT` — the port of the **magic-formula paper Gateway** on this box (its own instance;
  keep it distinct from the contract strategy's Gateway, e.g. `4002`).
- `IB_CLIENT_ID=5` — distinct from contract (3) and forex (1).
- `EMAIL_USER`, `EMAIL_PASS` (Gmail **App Password**), `TO_EMAIL` — for the daily report.
- `SEC_USER_AGENT` — only needed if you run EDGAR backtests here; not needed for paper.

## 4. IB Gateway (paper)

Run a **second** IB Gateway instance logged into the **new paper account** (DUR195822):
- **Configure → Settings → API → Settings:** enable ActiveX/Socket Clients; set the
  **Socket port** to match `IB_PORT`; trust `127.0.0.1`; uncheck **Read-Only API**.
- **Deactivate the API order precautions** (Presets → uncheck price %, total value, total
  size limits, or accept "deactivate for API orders") — otherwise orders sit at
  PendingSubmit and never fill. *(This bit us during testing.)*

## 5. Test before scheduling

Dry run (no orders, prints the picks + writes an HTML report, needs no Gateway):
```bat
call .venv\Scripts\activate.bat
python scripts\run_paper.py --dry-run --force
```
Then a real one during US market hours (Gateway must be up) to confirm a live fill:
```bat
python scripts\run_paper.py --force
```
Check `results\paper\run.log` and that the email arrived.

## 6. Schedule (weekdays 16:30 CET — must run FIRST)

Uses the box's **local time** — set the box to CET, or adjust the time. One-liner:

```bat
schtasks /Create /TN "MagicFormulaPaper" ^
  /TR "C:\trading\magic-formula\scripts\run_paper.bat" ^
  /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:30 /F
```
(Adjust the path.) The script self-skips weekends as a backstop. Output appends to
`results\paper\run.log`.

## 7. Operating notes

- **State** lives in `results\paper\state.json` on this box (positions, clocks, inception,
  realized P&L). It is NOT in git — this box is the source of truth. Back it up if you care
  about the track record. To **restart clean**: stop the task, delete `results\paper\`, flatten
  positions in TWS.
- **Monthly** the first run of a new calendar month does the heavy ~681-name universe pull
  (~20 min, cached to `results\paper\`); other days are fast.
- **Market data**: the "Error 162 / different IP" message is harmless — IB paper fills against
  the real market internally; the strategy never requests IB prices.
- **Live audit tags**: whenever new code is promoted to the LIVE run, tag that commit
  `live-YYYY-MM-DD` (annotated) and push it — `git tag -l "live-*"` then gives a timeline of
  what code was live when, and `git show live-YYYY-MM-DD` recalls the deployment details.
  First go-live: `live-2026-08-18` (acct U27760647, 50k).
- Freeze the strategy while it runs (no parameter tweaks) to keep the OOS track record valid.
