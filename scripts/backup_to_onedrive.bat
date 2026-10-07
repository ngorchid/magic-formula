@echo off
REM Nightly backup of the live trading state to OneDrive (TradingBackupOneDrive task, 23:30).
REM Emails on any problem. Tracked in git so a fresh clone keeps the task wrapper.
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
call .venv\Scripts\activate.bat
echo ===== backup %DATE% %TIME% ===== >> logs\backup.log
python scripts\backup_to_onedrive.py >> logs\backup.log 2>&1
