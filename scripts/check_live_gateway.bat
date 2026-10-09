@echo off
REM Live-Gateway health check (emails only on failure). Scheduled weekday mornings.
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
call .venv\Scripts\activate.bat
echo ===== health check %DATE% %TIME% ===== >> logs\health_check.log
python scripts\check_live_gateway.py >> logs\health_check.log 2>&1
