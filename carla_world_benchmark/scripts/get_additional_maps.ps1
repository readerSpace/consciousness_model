# Fetch and install AdditionalMaps for the already-extracted Windows CARLA.
#
#     powershell -ExecutionPolicy Bypass -File .\scripts\get_additional_maps.ps1
#
# AdditionalMaps is what carries Town12 and Town13 -- the maps every Leaderboard
# 2.0 route uses, and the only thing the extracted CARLA_Latest package is
# missing. It comes from the same nightly host as CARLA_Latest.zip itself, so it
# matches that build.

param(
    [switch]$SkipDownload,
    [string]$DownloadDir = "$env:USERPROFILE\Downloads\carla_pkgs"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "config.ps1")

$MapsUrl = "https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/Dev/AdditionalMaps_Latest.zip"
$MapsZip = Join-Path $DownloadDir "AdditionalMaps_Latest.zip"

if (-not (Test-Path (Join-Path $CarlaRoot "CarlaUE4.exe"))) {
    throw "No CarlaUE4.exe under $CarlaRoot. Extract CARLA_Latest.zip first (scripts/setup_windows.ps1)."
}

if (-not $SkipDownload) {
    New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
    if (Test-Path $MapsZip) {
        Write-Host "[skip] $MapsZip already downloaded ($([math]::Round((Get-Item $MapsZip).Length/1GB,1)) GB)"
    } else {
        Write-Host "[download] $MapsUrl"
        Write-Host "           -> $MapsZip  (several GB)"
        $ProgressPreference = "SilentlyContinue"   # progress rendering cripples large downloads
        Invoke-WebRequest -Uri $MapsUrl -OutFile $MapsZip -UseBasicParsing
    }
    # An HTML or XML error page saved under a .zip name is the failure mode that
    # wastes the most time, so check before spending minutes on extraction.
    $head = [System.IO.File]::ReadAllBytes($MapsZip)[0..1]
    if (-not ($head[0] -eq 0x50 -and $head[1] -eq 0x4B)) {
        Remove-Item $MapsZip
        throw "What downloaded is not a zip (no PK header). The host refused; the bad file was deleted."
    }
}

$Import = Join-Path $CarlaRoot "Import"
$ImportAssets = Join-Path $CarlaRoot "ImportAssets.bat"

if (Test-Path $ImportAssets) {
    Write-Host "[import] via ImportAssets.bat"
    New-Item -ItemType Directory -Force -Path $Import | Out-Null
    Copy-Item $MapsZip $Import -Force
    Push-Location $CarlaRoot
    try { & $ImportAssets } finally { Pop-Location }
} else {
    Write-Host "[extract] no ImportAssets.bat; unpacking into $CarlaRoot"
    $Tar = Join-Path $env:SystemRoot "System32\tar.exe"
    if (Test-Path $Tar) { & $Tar -xf $MapsZip -C $CarlaRoot }
    else { Expand-Archive -Path $MapsZip -DestinationPath $CarlaRoot -Force }
}

Write-Host ""
Write-Host "Done. Confirm the maps are there with the CARLA server running:"
Write-Host "  . .\scripts\env.ps1"
Write-Host "  python .\verify\verify_setup.py"
Write-Host "The 'maps installed' line should now list Town12 and Town13."
