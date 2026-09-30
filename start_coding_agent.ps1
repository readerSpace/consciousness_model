param(
    [int]$UiPort = 5173,
    [int]$BridgePort = 8787
)

& (Join-Path $PSScriptRoot "coding_world_benchmark\start_coding_agent.ps1") `
    -UiPort $UiPort -BridgePort $BridgePort
