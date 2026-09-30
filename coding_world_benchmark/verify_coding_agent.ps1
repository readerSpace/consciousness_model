param(
    [int]$BridgePort = 8787,
    [switch]$Offline
)

# 起動中のコーディングエージェントが意図した機能を保持しているか検証する。
#
# start_coding_agent.ps1 は UI をビルドして Electron を起動し、Electron が
# Python bridge を spawn する。その鎖のどこが切れても画面上は動いて見えるので、
# ここでは「UI が呼ぶ endpoint が実際に応答するか」「ビルド済み bundle が
# ソースより新しく、かつ中身に panel が入っているか」「広告しているコマンドが
# 実際に処理されるか」を実測する。
#
#   .\verify_coding_agent.ps1                 # 起動中のエージェント(8787)を検証
#   .\verify_coding_agent.ps1 -BridgePort 9000
#   .\verify_coding_agent.ps1 -Offline        # 自分で bridge を立てて検証（未起動でも可）

$ErrorActionPreference = "Stop"
$benchmarkRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $benchmarkRoot
$python = Get-Command python -ErrorAction Stop

$listening = [bool](Get-NetTCPConnection -LocalPort $BridgePort -State Listen -ErrorAction SilentlyContinue)
if (-not $Offline -and -not $listening) {
    Write-Host "ポート $BridgePort で bridge が待ち受けていません。" -ForegroundColor Yellow
    Write-Host "先に .\start_coding_agent.ps1 を実行するか、-Offline を付けて単体検証してください。"
    exit 2
}

Push-Location $projectRoot
try {
    if ($Offline) {
        & $python.Source -m coding_world_benchmark.agent_contract_verification
    } else {
        Write-Host "起動中のエージェント (127.0.0.1:$BridgePort) を検証します。" -ForegroundColor Cyan
        & $python.Source -m coding_world_benchmark.agent_contract_verification --port $BridgePort
    }
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($code -eq 0) {
    Write-Host "すべての項目が通りました。" -ForegroundColor Green
} else {
    Write-Host "失敗した項目があります。上の表の NG 行を確認してください。" -ForegroundColor Red
    Write-Host "bundle が古い場合は .\start_coding_agent.ps1 を実行し直すとビルドし直されます。"
}
exit $code
