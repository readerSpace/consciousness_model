param(
    [int]$UiPort = 5173,
    [int]$BridgePort = 8787
)

$benchmarkRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $benchmarkRoot
$web = Join-Path $benchmarkRoot "web"
$python = Get-Command python -ErrorAction Stop
$npm = Get-Command npm.cmd -ErrorAction Stop
$logDir = Join-Path $web ".logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:CODING_AGENT_BRIDGE_PORT = "$BridgePort"
$env:ELECTRON_DISABLE_GPU = "1"

function Test-PortInUse([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (Test-PortInUse $BridgePort) {
    $stopScript = Join-Path $benchmarkRoot "stop_coding_agent.ps1"
    if (Test-Path $stopScript) { & $stopScript | Out-Null }
    Start-Sleep -Milliseconds 500
}
if (Test-PortInUse $BridgePort) { throw "Bridge port $BridgePort is already in use by another application." }

if (-not (Test-Path (Join-Path $web "node_modules"))) {
    Write-Host "Installing JavaScript dependencies..."
    Push-Location $web
    & $npm.Source install
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm install failed" }
    Pop-Location
}

$env:CODING_AGENT_BRIDGE_PORT = "$BridgePort"
$build = Start-Process -Wait -PassThru -FilePath $npm.Source -ArgumentList "run", "build" -WorkingDirectory $web
if ($build.ExitCode -ne 0) { throw "UI build failed. See the terminal output above." }
$electronPath = Join-Path $web "node_modules\electron\dist\electron.exe"
if (-not (Test-Path $electronPath)) { throw "Electron is not installed. Run npm install in $web first." }
$electronProcess = Start-Process -PassThru -FilePath $electronPath -ArgumentList "--disable-gpu", "--disable-gpu-compositing", "." `
    -WorkingDirectory $web -RedirectStandardOutput (Join-Path $logDir "electron.out.log") -RedirectStandardError (Join-Path $logDir "electron.err.log")
Set-Content -LiteralPath (Join-Path $logDir "electron.pid") -Value $electronProcess.Id
Write-Host "Conscious Coding Agent Electron desktop app started."
