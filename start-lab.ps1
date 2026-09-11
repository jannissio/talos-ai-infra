param([switch]$NoBrowser, [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskRunDir = Join-Path $taskRoot '.run'
$taskUrl = "http://127.0.0.1:$Port"
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'The local Python environment is missing. Follow simulation_lab/README.md to install it.'
}
New-Item -ItemType Directory -Path $taskRunDir -Force | Out-Null
$taskAlreadyRunning = $false
try {
    $taskState = Invoke-RestMethod -Uri "$taskUrl/api/state" -TimeoutSec 2
    $taskAlreadyRunning = $taskState.ready -and $taskState.controller -in @('manual_joint_targets', 'ground_truth_ik')
} catch { }
if (-not $taskAlreadyRunning) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is already used by another application. Run this script with -Port followed by another port."
    }
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-m', 'simulation_lab.server', '--port', $Port) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRunDir "server-$Port.stdout.log") -RedirectStandardError (Join-Path $taskRunDir "server-$Port.stderr.log") -PassThru
    @{pid=$taskProcess.Id; port=$Port; python=$taskPython; started_at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRunDir "server-$Port.json")
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 80; $taskAttempt++) {
        Start-Sleep -Milliseconds 250
        if ($taskProcess.HasExited) { break }
        try {
            $taskState = Invoke-RestMethod -Uri "$taskUrl/api/state" -TimeoutSec 2
            if ($taskState.ready) { $taskReady = $true; break }
        } catch { }
    }
    if (-not $taskReady) {
        throw "The simulator did not become ready. See .run/server-$Port.stderr.log."
    }
}
Write-Output "Talos is running at $taskUrl (dinner challenge and BenchLab practice)."
Write-Output 'Stop it with .\stop-lab.ps1 (use the same -Port if changed).'
if (-not $NoBrowser) { Start-Process $taskUrl }
