@echo off
REM Cross-strategy LIVE book summary email (weekdays 21:00 CET, after both sleeves' runs).
REM Reads magic-formula-live + trend-overlay-live state.json and the live account margin.
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
call .venv\Scripts\activate.bat
echo ===== book summary %DATE% %TIME% ===== >> logs\book_summary.log
python scripts\book_summary.py >> logs\book_summary.log 2>&1
