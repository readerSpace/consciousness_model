param(
    [switch]$Quiet
)

$script = Join-Path $PSScriptRoot "coding_world_benchmark\stop_coding_agent.ps1"
if ($Quiet) {
    & $script | Out-Null
} else {
    & $script
}
