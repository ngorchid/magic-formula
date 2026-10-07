# Set (or change) the password for the encrypted .env archive in the OneDrive backup.
#
# Run it YOURSELF, once, in a PowerShell window:
#     powershell -ExecutionPolicy Bypass -File C:\Users\Nicolas\PycharmProjects\magic-formula-live\scripts\set_backup_password.ps1
#
# The password is stored DPAPI-encrypted (Export-Clixml of a SecureString): only THIS Windows
# user on THIS PC can decrypt it, which is what lets the nightly task run unattended. It never
# goes to the repo or to OneDrive. Keep the password in your password manager too: restoring the
# .env archive on any other machine needs it (7-Zip: right-click > 7-Zip > Extract).
$ErrorActionPreference = 'Stop'
$dir = Join-Path $env:LOCALAPPDATA 'TradingBackup'
$file = Join-Path $dir 'envzip.cred'
New-Item -ItemType Directory -Force $dir | Out-Null

$p1 = Read-Host 'New backup password (min. 12 characters)' -AsSecureString
$p2 = Read-Host 'Repeat it' -AsSecureString
$plain1 = [System.Net.NetworkCredential]::new('', $p1).Password
$plain2 = [System.Net.NetworkCredential]::new('', $p2).Password
if ($plain1 -ne $plain2) { Write-Host 'The two entries differ - nothing saved.' -ForegroundColor Red; exit 1 }
if ($plain1.Length -lt 12) { Write-Host 'Too short (min. 12) - nothing saved.' -ForegroundColor Red; exit 1 }
[System.Management.Automation.PSCredential]::new('trading-backup', $p1) | Export-Clixml -LiteralPath $file
Write-Host "Saved (DPAPI-encrypted, this Windows user only): $file" -ForegroundColor Green
Write-Host 'Remember to store the password in your password manager.'
