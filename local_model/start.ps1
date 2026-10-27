[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = 'E:\Anaconda\envs\qwen3.5\python.exe'
$serverExe = Join-Path $PSScriptRoot 'runtime\b11871\llama-server.exe'
$modelFile = 'F:\Document\HKUST\2026fall\NLP\Qwen3.5_9B_4bit\Qwen3.5-9B-Q4_K_M.gguf'
$statePath = Join-Path $PSScriptRoot 'state.json'
$ledgerPath = Join-Path $projectRoot 'Baseline\runs\gpu_accounting.jsonl'
foreach ($requiredFile in @($pythonExe, $serverExe, $modelFile)) {
    if (-not (Test-Path -LiteralPath $requiredFile)) { throw "Missing required file: $requiredFile" }
}
if (Test-Path -LiteralPath $statePath) {
    $savedState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $existingProcess = Get-Process -Id $savedState.pid -ErrorAction SilentlyContinue
    if ($existingProcess) {
        if ($existingProcess.Path -ne $serverExe) { throw 'Saved PID belongs to a different program.' }
        $health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 5
        Write-Output "Already running: PID=$($savedState.pid), health=$($health.status)"
        return
    }
    throw 'Stale service state exists. Run stop.ps1 to close its allocation before restarting.'
}
if (Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue) {
    throw 'Port 8000 is already occupied; refusing to replace an unrelated service.'
}
$logDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$sessionName = '4070-qwen35-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $pythonExe (Join-Path $projectRoot 'Baseline\run.py') gpu start --ledger $ledgerPath --session $sessionName
if ($LASTEXITCODE -ne 0) { throw 'Could not open shared GPU allocation ledger.' }
$serverArguments = @(
    '--model', ('"{0}"' -f $modelFile),
    '--alias', 'Qwen3.5-9B-Q4_K_M',
    '--host', '127.0.0.1', '--port', '8000',
    '--ctx-size', '16384', '--parallel', '1',
    '--n-gpu-layers', '99', '--flash-attn', 'on',
    '--batch-size', '512', '--ubatch-size', '256',
    '--jinja', '--reasoning', 'off', '--no-context-shift'
)
$stdout = Join-Path $logDir ($sessionName + '.stdout.log')
$stderr = Join-Path $logDir ($sessionName + '.stderr.log')
$serverProcess = $null
try {
    $serverProcess = Start-Process -FilePath $serverExe -ArgumentList $serverArguments -WorkingDirectory (Split-Path -Parent $serverExe) -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $state = @{ pid=$serverProcess.Id; exe=$serverExe; model=$modelFile; alias='Qwen3.5-9B-Q4_K_M'; base_url='http://127.0.0.1:8000/v1'; session=$sessionName; ledger=$ledgerPath; stdout=$stdout; stderr=$stderr; started_utc=[DateTime]::UtcNow.ToString('o'); context_size=16384; parallel=1; reasoning='off' }
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding UTF8
    $deadline = (Get-Date).AddMinutes(4)
    while ((Get-Date) -lt $deadline) {
        $serverProcess.Refresh()
        if ($serverProcess.HasExited) { throw "Server exited. Inspect $stderr" }
        try {
            $health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2
            if ($health.status -eq 'ok') {
                Write-Output "Ready: http://127.0.0.1:8000/v1, PID=$($serverProcess.Id)"
                return
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
    throw "Model did not become ready before the startup timeout. Inspect $stderr"
} catch {
    if ($serverProcess -and -not $serverProcess.HasExited) { Stop-Process -Id $serverProcess.Id }
    & $pythonExe (Join-Path $projectRoot 'Baseline\run.py') gpu stop --ledger $ledgerPath --session $sessionName
    if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath }
    throw
}
