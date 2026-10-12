<#
One-time setup (run AS ADMINISTRATOR on the trading PC): keep the Windows session on the PC's own
display whenever a Remote Desktop connection ends, however it ends.  (2026-10-12)

WHY. A DISCONNECTED Remote Desktop session has no display, so the IB Gateways (Java desktop apps)
cannot open their login windows: IBC loops on "failed to display login dialog", never logs in, and
no 2FA push is sent (outages 2026-10-09 and 2026-10-12). The manual "Disconnect to console"
shortcut (ops/disconnect_to_console.bat) worked on 2026-10-12 10:04:56, but the Remote Desktop app
saw a dropped connection, AUTO-RECONNECTED 12 s later and took the session back; the window was
then closed normally at 10:20 and the 12:00 Gateway restart failed.

WHAT IT DOES
  1. Policy fDisableAutoReconnect=1 (HKLM\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services):
     Remote Desktop apps no longer reconnect on their own, from any device. Side effect: after a
     real network blip you reconnect by hand.
  2. Scheduled task "RDP_DisconnectToConsole" (runs as SYSTEM): on every Remote Desktop disconnect
     of this user, after 5 s, if the session is still Disconnected, hand it to the console
     (tscon <id> /dest:console). Closing the RDP window normally is then fine. If the session is
     already on the console (the shortcut was used) it does nothing. Each run is logged to
     C:\ProgramData\RdpToConsole\log.txt.
  The task's command is embedded in the task itself (no script file), so nothing a non-admin
  process can edit ever runs as SYSTEM.

TRADE-OFF. The PC's own screen shows the unlocked desktop until Windows locks it (a locked screen
does not stop the Gateways; only a disconnected session does).

Run (elevated PowerShell):
  powershell -ExecutionPolicy Bypass -File ops\setup_auto_console.ps1            # install
  powershell -ExecutionPolicy Bypass -File ops\setup_auto_console.ps1 -Check     # status only (no admin needed)
  powershell -ExecutionPolicy Bypass -File ops\setup_auto_console.ps1 -Uninstall # undo both
#>
param([switch]$Uninstall, [switch]$Check, [string]$User = $env:USERNAME)

$TaskName = 'RDP_DisconnectToConsole'
$PolicyKey = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services'
$LogDir = 'C:\ProgramData\RdpToConsole'

function Show-Status {
    $p = (Get-ItemProperty -Path $PolicyKey -Name fDisableAutoReconnect -ErrorAction SilentlyContinue).fDisableAutoReconnect
    Write-Host ("Auto-reconnect disabled (policy): " + $(if ($p -eq 1) { 'YES' } else { 'no' }))
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t) {
        $i = $t | Get-ScheduledTaskInfo
        Write-Host ("Task ${TaskName}: $($t.State), last run $($i.LastRunTime), last result $($i.LastTaskResult)")
    } elseif (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
            [Security.Principal.WindowsBuiltInRole]::Administrator)) {
        # A SYSTEM task is invisible to a non-elevated shell ("Access is denied"), installed or not.
        Write-Host "Task ${TaskName}: cannot be seen without admin rights (run -Check elevated); the log below shows whether it runs"
    } else { Write-Host "Task ${TaskName}: not installed" }
    if (Test-Path "$LogDir\log.txt") { Write-Host "Last log lines:"; Get-Content "$LogDir\log.txt" -Tail 5 }
}

if ($Check) { Show-Status; exit 0 }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "Run this from an ELEVATED PowerShell (right-click PowerShell > Run as administrator)." -ForegroundColor Red
    exit 1
}

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Remove-ItemProperty -Path $PolicyKey -Name fDisableAutoReconnect -ErrorAction SilentlyContinue
    Write-Host "Removed the task and the auto-reconnect policy."
    Show-Status; exit 0
}

# 1. No automatic reconnection from any Remote Desktop app.
if (-not (Test-Path $PolicyKey)) { New-Item -Path $PolicyKey -Force | Out-Null }
New-ItemProperty -Path $PolicyKey -Name fDisableAutoReconnect -PropertyType DWord -Value 1 -Force | Out-Null

# 2. On a Remote Desktop disconnect: hand the still-disconnected session to the console.
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$cmd = @"
`$u = '$User'
`$log = '$LogDir\log.txt'
`$line = (qwinsta | Select-String -Pattern ('\b' + [regex]::Escape(`$u) + '\s+(\d+)\s+Disc\b') | Select-Object -First 1)
if (`$line) {
    `$id = `$line.Matches[0].Groups[1].Value
    `$out = (tscon.exe `$id /dest:console 2>&1 | Out-String).Trim()
    Add-Content `$log ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + " session `$id disconnected -> tscon to console, exit `$LASTEXITCODE `$out")
} else {
    Add-Content `$log ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' no disconnected session for ' + `$u + ' (already on the console?) - nothing to do')
}
"@
$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $enc"

$cls = Get-CimClass -ClassName MSFT_TaskSessionStateChangeTrigger -Namespace Root/Microsoft/Windows/TaskScheduler
$trigger = New-CimInstance -CimClass $cls -ClientOnly -Property @{
    StateChange = 4                                   # TASK_SESSION_REMOTE_DISCONNECT
    UserId      = "$env:COMPUTERNAME\$User"
    Delay       = 'PT5S'                              # let the disconnect finish first
    Enabled     = $true
}
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'Keep the session on the PC display after a Remote Desktop disconnect, so the IB Gateways can show their login windows (magic-formula-live ops/setup_auto_console.ps1).' `
    -Force | Out-Null

Write-Host "Installed." -ForegroundColor Green
Show-Status
Write-Host "`nTest: close the Remote Desktop window normally, wait a minute, reconnect, then run with -Check:"
Write-Host "the log should say 'tscon to console, exit 0'."
