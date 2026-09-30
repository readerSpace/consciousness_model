# Start the CARLA server. Leave this shell running.
#     .\scripts\start_carla.ps1
#     .\scripts\start_carla.ps1 -Headless   off-screen rendering (still needs the GPU)

param([switch]$Headless, [int]$Port = 0, [string]$Quality = "")

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "config.ps1")
if ($Port -ne 0)  { $CarlaPort = $Port }
if ($Quality)     { $CarlaQuality = $Quality }

$Exe = Join-Path $CarlaRoot "CarlaUE4.exe"
if (-not (Test-Path $Exe)) { throw "Not found: $Exe  (run setup_windows.ps1 first)" }

$CarlaArgs = @("-quality-level=$CarlaQuality", "-carla-server", "-world-port=$CarlaPort")
if ($Headless) { $CarlaArgs += "-RenderOffScreen" }
else           { $CarlaArgs += @("-windowed", "-ResX=800", "-ResY=600") }

Write-Host "[carla] $Exe $($CarlaArgs -join ' ')"
& $Exe @CarlaArgs
