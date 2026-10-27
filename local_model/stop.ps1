[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $PSScriptRoot 'state.json'
$pythonExe = 'E:\Anaconda\envs\qwen3.5\python.exe'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'No managed service state.'; return }
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$serverProcess = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
if ($serverProcess) {
    if ($serverProcess.Path -ne $state.exe) { throw 'PID was reused by another program; refusing to stop it.' }
    $command = (Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)").CommandLine
    if (-not $command.Contains($state.model)) { throw 'Running command does not match the managed model.' }
    Stop-Process -Id $state.pid
    Wait-Process -Id $state.pid -Timeout 15 -ErrorAction SilentlyContinue
}
& $pythonExe (Join-Path $projectRoot 'Baseline\run.py') gpu stop --ledger $state.ledger --session $state.session
if ($LASTEXITCODE -ne 0) { throw 'GPU ledger close failed; retain state for inspection.' }
Remove-Item -LiteralPath $statePath
Write-Output 'Stopped model service and closed its GPU allocation.'
