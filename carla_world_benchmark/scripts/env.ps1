# Dot-source this file to configure a shell:
#     . .\scripts\env.ps1                 standalone runner (Python 3.12)
#     . .\scripts\env.ps1 -Leaderboard    official leaderboard (Python 3.10)

param([switch]$Leaderboard)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "config.ps1")

$BenchRoot   = Split-Path -Parent $ScriptDir
$ProjectRoot = Split-Path -Parent $BenchRoot

$env:CARLA_ROOT           = $CarlaRoot
$env:LEADERBOARD_ROOT     = Join-Path $BenchRoot "external\leaderboard"
$env:SCENARIO_RUNNER_ROOT = Join-Path $BenchRoot "external\scenario_runner"
$env:BENCH_ROOT           = $BenchRoot
$env:PROJECT_ROOT         = $ProjectRoot
$env:CARLA_PORT           = $CarlaPort

# PythonAPI\carla carries the agents package (BasicAgent, LocalPlanner) that both
# the runner and the leaderboard agent use for path following.
$env:PYTHONPATH = @(
    (Join-Path $env:CARLA_ROOT "PythonAPI\carla"),
    $env:SCENARIO_RUNNER_ROOT,
    $env:LEADERBOARD_ROOT,
    $ProjectRoot
) -join ";"

$Selected = if ($Leaderboard) { $VenvLeaderboardRoot } else { $VenvRoot }
$Activate = Join-Path $Selected "Scripts\Activate.ps1"
if (Test-Path $Activate) { . $Activate }
else {
    Write-Host "[warn] no venv at $Selected"
    if ($Leaderboard) { Write-Host "[warn] create it with: .\scripts\setup_windows.ps1 -Leaderboard" }
}

Write-Host "CARLA_ROOT = $env:CARLA_ROOT"
Write-Host "PYTHONPATH = $env:PYTHONPATH"
Write-Host "venv       = $Selected"
$Resolved = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($Resolved -and -not $Resolved.StartsWith($Selected)) {
    # Another interpreter ahead on PATH silently breaks `import carla`.
    Write-Host "[warn] python resolves to $Resolved, not the venv above."
}
