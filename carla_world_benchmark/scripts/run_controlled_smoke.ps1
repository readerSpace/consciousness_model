# Run the first controlled CARLA smoke experiment.
# The CARLA server must already be running.
#
#     . .\scripts\env.ps1
#     .\scripts\run_controlled_smoke.ps1 -Scenario sudden_stop -Seed 0 -NoRender
#
# This is the first real-CARLA milestone: four conditions, one fixed episode
# each, and enough per-tick diagnostics in results/ to trace failures.

param(
    [ValidateSet("sudden_stop", "cut_in", "route_stop", "random_traffic")]
    [string]$Scenario = "sudden_stop",
    [string]$Town     = "Town10HD",
    [int]$Seed        = 0,
    [double]$Timeout  = 90.0,
    [switch]$NoRender,
    [string[]]$Configs = @(
        "agent\configs\k4_workspace.json",
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
        Write-Host "================ $Scenario / $Config ================"
        $RunnerArgs = @(
            (Join-Path $BenchRoot "runner\carla_runner.py"),
            "--config", (Join-Path $BenchRoot $Config),
            "--port", $CarlaPort,
            "--tm-port", $TrafficManagerPort,
            "--town", $Town,
            "--scenario", $Scenario,
            "--seeds", 1,
            "--first-seed", $Seed,
            "--vehicles", 0,
            "--walkers", 0,
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
