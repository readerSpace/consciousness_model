$benchmarkRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $benchmarkRoot "web\.logs"

function Stop-ProcessTree([int]$processId) {
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $processId" -ErrorAction SilentlyContinue
    foreach ($child in $children) { Stop-ProcessTree ([int]$child.ProcessId) }
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

function Stop-MatchingProcess([string]$pattern) {
    $matches = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $pattern }
    foreach ($match in $matches) { Stop-ProcessTree ([int]$match.ProcessId) }
}

foreach ($name in @("electron.pid", "vite.pid", "bridge.pid")) {
    $pidPath = Join-Path $logDir $name
    if (-not (Test-Path $pidPath)) { continue }
    $rawPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
    if (-not $rawPid -or $rawPid -notmatch '^\d+$') {
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        continue
    }
    $processId = [int]$rawPid
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        Stop-ProcessTree $processId
        Write-Host "Stopped $name ($processId)"
    } else {
        Write-Host "$name ($processId) is already stopped"
    }
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}
Stop-MatchingProcess "coding_world_benchmark\.coding_agent_bridge"
Stop-MatchingProcess "coding_world_benchmark\\web.*vite"
Stop-MatchingProcess "coding_world_benchmark\\web.*electron"
Write-Host "Conscious Coding Agent stopped."
