@echo off
REM Wrapper for the LIVE gateway (IBCWin-live, port 4001, account ngrein905 / U27760647).
REM
REM Adapted 2026-08-31. This was a straight copy of the paper/contract wrapper: it checked
REM ports 4003/4002 and the window title "LYNX Broker Gateway" -- ALL WRONG for live, whose
REM title is "IBKR Gateway" and whose port is 4001. The old copy saw the paper Gateway already
REM up on 4003 and skipped launching the live one, so it could never recover the live session.
REM
REM Behavior:
REM  1. If 4001 is already listening, the live Gateway is healthy -> skip.
REM  1b. (2026-10-09) Otherwise WAIT up to 180 s for 4001 to come back before killing anything.
REM     The Gateway restarts itself daily at 12:00 (IBC AutoRestartTime) and the port is closed
REM     for ~20-60 s; on 2026-10-09 this script killed that self-restart mid-way, which threw
REM     away the session and forced a fresh 2FA login (outage 12:00-15:00). A hung Gateway is
REM     still killed, just 3 minutes later. Backup: StartGatewayFresh.bat.bak-20261009.
REM  2. Otherwise kill any java Gateway using the LIVE settings path (C:\Jts-live) -- a stuck
REM     instance holding the session lock. Matched by settings/config PATH, never by window
REM     title (paper and live can share one) and never by any port other than 4001, so the
REM     4002 contract and 4003 paper Gateways are never touched.
REM  3. Launch via IBC's StartGateway.bat. A fresh login triggers the weekly 2FA push, which
REM     must be approved on IBKR Mobile -- so the recovery task runs in the MORNING, not at night.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = (Get-NetTCPConnection -LocalPort 4001 -State Listen -ErrorAction SilentlyContinue).OwningProcess; if ($p) { Write-Host ('Pre-flight: live Gateway already listening on 4001 (PID ' + $p + '), skipping launch.'); exit 42 }; Write-Host 'Pre-flight: 4001 closed - waiting up to 180 s in case the Gateway is restarting itself.'; $end = (Get-Date).AddSeconds(180); while ((Get-Date) -lt $end) { Start-Sleep -Seconds 10; $p = (Get-NetTCPConnection -LocalPort 4001 -State Listen -ErrorAction SilentlyContinue).OwningProcess; if ($p) { Write-Host ('Pre-flight: live Gateway came back on 4001 (PID ' + $p + '), skipping launch.'); exit 42 } }; $live = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -and ($_.CommandLine -match 'Jts-live' -or $_.CommandLine -match 'IBCWin-live') }; foreach ($z in $live) { Write-Host ('Pre-flight: killing stuck live Gateway PID ' + $z.ProcessId); Stop-Process -Id $z.ProcessId -Force -ErrorAction SilentlyContinue }; if ($live) { Start-Sleep -Seconds 5; Write-Host 'Pre-flight: session-clear wait done.' } else { Write-Host 'Pre-flight: no stuck live Gateway found.' }"

if errorlevel 42 (
    echo Skipping IBC launch - healthy live Gateway already up.
    exit /B 0
)

call "%~dp0StartGateway.bat" %*
