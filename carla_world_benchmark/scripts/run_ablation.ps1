# Run the same route under every workspace configuration, one after another.
# The CARLA server must already be running.
#
#     . .\scripts\env.ps1
#     .\scripts\run_ablation.ps1 -RoutesSubset 0

param(
    [string]$RoutesSubset = "0",
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

foreach ($Config in $Configs) {
    Write-Host ""
    Write-Host "================ $Config ================"
    & (Join-Path $ScriptDir "run_leaderboard.ps1") -AgentConfig $Config -RoutesSubset $RoutesSubset
}
