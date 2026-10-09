@echo off
REM Pre-market warm-up (weekday mornings): rebuild the monthly universe cache so the 16:00 live
REM run is always fast and trades before Oslo's 16:20 close. No orders, no email, no IB.
REM See scripts\warmup_universe.py.
cd /d "%~dp0.."
if not exist "results\paper" mkdir "results\paper"
call .venv\Scripts\activate.bat
echo ===== warmup %DATE% %TIME% ===== >> results\paper\warmup.log
python scripts\warmup_universe.py 1>> results\paper\warmup.log 2>&1
