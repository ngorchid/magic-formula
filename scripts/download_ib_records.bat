@echo off
REM Daily archive of IB's Flex records for the live account (the audit trail). Emails only on
REM failure or an incomplete trail. Tracked in git so a fresh clone keeps the task wrapper.
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
call .venv\Scripts\activate.bat
echo ===== ib records %DATE% %TIME% ===== >> logs\ib_records.log
python scripts\download_ib_records.py >> logs\ib_records.log 2>&1
