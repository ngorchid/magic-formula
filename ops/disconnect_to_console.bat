@echo off
REM Leave a Remote Desktop session WITHOUT leaving it display-less (2026-10-12).
REM
REM Closing the RDP window leaves the session DISCONNECTED, and a disconnected session has no
REM display: the IB Gateways (Java desktop apps) can then not open their login windows, so IBC
REM loops on "failed to display login dialog", never logs in and no 2FA push is ever sent
REM (2026-10-09: outage until an RDP reconnect at 14:50; 2026-10-12: 74 failed launches 08:00-09:22).
REM
REM This hands the session to the PC's physical console instead: it stays logged in WITH a display
REM (the RDP window closes as usual). Trade-off: the PC's own screen shows the unlocked desktop until
REM it locks itself or you reconnect.
REM
REM Needs administrator rights: run it from the "Disconnect to console" desktop shortcut (it asks
REM for UAC approval). If it fails, the window stays open with the error.

powershell -NoProfile -Command "$s = (Get-Process -Id $PID).SessionId; Write-Host ('Handing session ' + $s + ' to the console...'); tscon.exe $s /dest:console; exit $LASTEXITCODE"
if errorlevel 1 (
    echo tscon FAILED - the session was NOT moved. Is this window running as administrator?
    pause
)
