[CmdletBinding()]
param([string]$PythonExe = 'E:\Anaconda\envs\qwen3.5\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'state.json')) {
    $stateText = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'state.json') -Raw
    $stateText
    $state = $stateText | ConvertFrom-Json
    if ($state.python) { $PythonExe = $state.python }
    try { Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 3 | ConvertTo-Json } catch { Write-Output 'Service is not healthy.' }
} else { Write-Output 'Model service is stopped.' }
& $PythonExe (Join-Path $projectRoot 'Baseline\run.py') gpu status --ledger (Join-Path $projectRoot 'Baseline\runs\gpu_accounting.jsonl')
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
