# Run the capacity ablation with the standalone runner (no leaderboard needed).
# The CARLA server must already be running.
#
#     . .\scripts\env.ps1
#     .\scripts\run_experiment.ps1 -Town Town10HD -Seeds 3
#
# Add -NoRender for a large speed-up when you do not need to watch it drive.

param(
    [string]$Town     = "Town10HD",
    [int]$Seeds       = 3,
    [int]$Vehicles    = 60,
    [int]$Walkers     = 30,
    [double]$Timeout  = 300.0,
    [switch]$NoRender,
    [string[]]$Configs = @(
        "agent\configs\k4_workspace.json",
        "agent\configs\k1_workspace.json",
        "agent\configs\k4_random.json",
        "agent\configs\unbounded.json",
        "agent\configs\blind.json"
    )
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "env.ps1")
$BenchRoot = Split-Path -Parent $ScriptDir

$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "No standalone runner venv at $VenvRoot. Run: powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -SkipExtract"
}
& python -c "import carla" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "The active Python cannot import carla. Run: . .\scripts\env.ps1, then confirm python resolves under $VenvRoot."
}

Push-Location $BenchRoot
try {
    foreach ($Config in $Configs) {
        Write-Host ""
        Write-Host "================ $Config ================"
        $RunnerArgs = @(
            (Join-Path $BenchRoot "runner\carla_runner.py"),
            "--config", (Join-Path $BenchRoot $Config),
            "--port", $CarlaPort,
            "--tm-port", $TrafficManagerPort,
            "--town", $Town,
            "--seeds", $Seeds,
            "--vehicles", $Vehicles,
            "--walkers", $Walkers,
            "--timeout", $Timeout
        )
        if ($NoRender) { $RunnerArgs += "--no-render" }
        & python @RunnerArgs
        if ($LASTEXITCODE -ne 0) { throw "runner failed for $Config" }
    }
    Write-Host ""
    & python (Join-Path $BenchRoot "verify\analyze_ablation.py") --results-dir (Join-Path $BenchRoot "results")
}
finally { Pop-Location }
