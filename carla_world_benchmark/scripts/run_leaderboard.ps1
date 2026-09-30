# Run one leaderboard evaluation with the consciousness agent.
#
#     . .\scripts\env.ps1
#     .\scripts\run_leaderboard.ps1 -AgentConfig agent\configs\k4_workspace.json
#
# Routes default to routes_devtest.xml (Town12), which is the shortest set.

param(
    [string]$AgentConfig = "agent\configs\k4_workspace.json",
    [string]$Routes      = "",
    [string]$RoutesSubset= "0",
    [int]$Repetitions    = 1,
    [string]$Track       = "SENSORS",
    [string]$Checkpoint  = "",
    [int]$Debug          = 0
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "config.ps1")
$BenchRoot = Split-Path -Parent $ScriptDir

if (-not $env:LEADERBOARD_ROOT) { throw "Run '. .\scripts\env.ps1' first." }
if (-not $Routes) { $Routes = Join-Path $env:LEADERBOARD_ROOT "data\routes_devtest.xml" }

$AgentConfigPath = if ([System.IO.Path]::IsPathRooted($AgentConfig)) { $AgentConfig }
                   else { Join-Path $BenchRoot $AgentConfig }
if (-not (Test-Path $AgentConfigPath)) { throw "Agent config not found: $AgentConfigPath" }

$ResultsDir = Join-Path $BenchRoot "results"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
if (-not $Checkpoint) {
    $Tag = [System.IO.Path]::GetFileNameWithoutExtension($AgentConfigPath)
    $Checkpoint = Join-Path $ResultsDir "$Tag.leaderboard.json"
}

# record_path inside the agent config is relative to the benchmark root.
Push-Location $BenchRoot
try {
    python (Join-Path $env:LEADERBOARD_ROOT "leaderboard\leaderboard_evaluator.py") `
        --host=127.0.0.1 `
        --port=$CarlaPort `
        --traffic-manager-port=$TrafficManagerPort `
        --routes=$Routes `
        --routes-subset=$RoutesSubset `
        --repetitions=$Repetitions `
        --track=$Track `
        --checkpoint=$Checkpoint `
        --agent=(Join-Path $BenchRoot "agent\consciousness_agent.py") `
        --agent-config=$AgentConfigPath `
        --debug=$Debug
}
finally { Pop-Location }

Write-Host "leaderboard result -> $Checkpoint"
