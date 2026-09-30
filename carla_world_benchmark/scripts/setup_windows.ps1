# One-time setup from the CARLA package that ships with this repo.
#
#     powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
#
# Extracts CARLA_Latest.zip, creates a Python 3.12 virtual environment, and
# installs the shipped CARLA client plus the standalone runner's dependencies.
# Re-running is safe: an existing extraction and an existing venv are reused.
#
#   -SkipExtract        CARLA is already extracted at $CarlaRoot
#   -Leaderboard        also build the Python 3.10 environment for the official
#                       leaderboard evaluator (carla 0.9.16 cp310 wheel from
#                       PyPI + the relaxed dependency set + the 3.9 patches)
#   -Force              re-extract even if CarlaUE4.exe is already present

param(
    [switch]$SkipExtract,
    [switch]$Leaderboard,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "config.ps1")
$BenchRoot = Split-Path -Parent $ScriptDir

$RequiredGB = 21

# --- interpreters -----------------------------------------------------------
# CARLA clients only exist for specific CPython versions (3.12 for the wheel in
# the package, 3.10 for the leaderboard stack), and the system Python is rarely
# one of them. uv fetches a standalone build of the exact version, so nothing
# has to be installed from python.org by hand.

function Get-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($candidate in @("$env:USERPROFILE\.local\bin\uv.exe",
                             "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $candidate) { return $candidate }
    }
    Write-Host "[uv] installing to `$env:USERPROFILE\.local\bin"
    $previous = $env:PATH
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Write-Host "[uv] installer failed: $($_.Exception.Message)"
        return $null
    }
    $env:PATH = "$env:USERPROFILE\.local\bin;$previous"
    foreach ($candidate in @("$env:USERPROFILE\.local\bin\uv.exe",
                             "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $candidate) { return $candidate }
    }
    return (Get-Command uv -ErrorAction SilentlyContinue).Source
}

function New-CarlaVenv {
    # Create $Path on CPython $Version. Prefers uv; falls back to the py launcher
    # when uv is unavailable and that version happens to be installed already.
    param([string]$Path, [string]$Version, [string]$Uv)

    if (Test-Path (Join-Path $Path "Scripts\python.exe")) {
        Write-Host "[skip] venv already exists at $Path"
        return
    }
    if ($Uv) {
        Write-Host "[venv] $Path on CPython $Version (via uv)"
        & $Uv python install $Version
        if ($LASTEXITCODE -ne 0) { throw "uv could not provide CPython $Version" }
        & $Uv venv --python $Version $Path
        if ($LASTEXITCODE -ne 0) { throw "uv could not create the venv at $Path" }
        return
    }
    Write-Host "[venv] $Path on CPython $Version (via the py launcher)"
    & py "-$Version" -m venv $Path
    if ($LASTEXITCODE -ne 0) {
        throw ("Could not create a CPython $Version venv. Either let uv handle it " +
               "(check your network) or install CPython $Version (64-bit) from python.org.")
    }
}

function Install-Into {
    # uv pip works against any venv and does not need pip inside it, which a
    # uv-created venv does not have by default.
    param([string]$Python, [string]$Uv, [string[]]$Arguments)
    if ($Uv) {
        & $Uv pip install --python $Python @Arguments
    } else {
        & $Python -m pip install @Arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "install failed: $($Arguments -join ' ')" }
}

function Test-FreeSpace([string]$Path, [int]$NeededGB) {
    $Qualifier = (Split-Path -Qualifier ([System.IO.Path]::GetFullPath($Path)))
    $Drive = Get-PSDrive -Name $Qualifier.TrimEnd(':') -ErrorAction SilentlyContinue
    if (-not $Drive) { return }
    $FreeGB = [math]::Round($Drive.Free / 1GB, 1)
    Write-Host "[disk] $Qualifier has $FreeGB GB free (need about $NeededGB GB)"
    if ($Drive.Free -lt ($NeededGB * 1GB)) {
        throw "Not enough free space on $Qualifier. Point `$CarlaRoot in scripts/config.ps1 at another drive."
    }
}

# --- extract ----------------------------------------------------------------

$CarlaExe = Join-Path $CarlaRoot "CarlaUE4.exe"

if (-not $SkipExtract) {
    if (-not (Test-Path $CarlaZip)) { throw "Package not found: $CarlaZip" }
    if ((Test-Path $CarlaExe) -and -not $Force) {
        Write-Host "[skip] CarlaUE4.exe already present at $CarlaRoot (use -Force to re-extract)"
    } else {
        Test-FreeSpace $CarlaRoot $RequiredGB
        New-Item -ItemType Directory -Force -Path $CarlaRoot | Out-Null
        $Tar = Join-Path $env:SystemRoot "System32\tar.exe"
        $Started = Get-Date
        if (Test-Path $Tar) {
            # bsdtar reads zip and is several times faster than Expand-Archive
            # on an archive of this size (32855 files, 19.4 GB).
            Write-Host "[extract] tar -xf $CarlaZip -> $CarlaRoot  (expect 10-30 minutes)"
            & $Tar -xf $CarlaZip -C $CarlaRoot
            if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
        } else {
            Write-Host "[extract] Expand-Archive -> $CarlaRoot  (slow; expect an hour or more)"
            Expand-Archive -Path $CarlaZip -DestinationPath $CarlaRoot -Force
        }
        Write-Host ("[extract] done in {0:N1} minutes" -f ((Get-Date) - $Started).TotalMinutes)
    }
}

if (-not (Test-Path $CarlaExe)) {
    throw "CarlaUE4.exe not found under $CarlaRoot. Check `$CarlaRoot in scripts/config.ps1."
}

# --- python environment -----------------------------------------------------

$Uv = Get-Uv
if ($Uv) { Write-Host "[uv] $Uv" } else { Write-Host "[uv] unavailable; falling back to the py launcher" }

New-CarlaVenv -Path $VenvRoot -Version "3.12" -Uv $Uv
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
Install-Into -Python $VenvPython -Uv $Uv -Arguments @("--upgrade", "setuptools", "wheel")

$Dist = Join-Path $CarlaRoot "PythonAPI\carla\dist"
$Wheel = Get-ChildItem -Path $Dist -Filter "*win_amd64.whl" -ErrorAction SilentlyContinue |
         Select-Object -First 1
if (-not $Wheel) { throw "No CARLA client wheel found in $Dist" }
Write-Host "[pip] installing shipped client $($Wheel.Name)"
Install-Into -Python $VenvPython -Uv $Uv -Arguments @($Wheel.FullName)

Write-Host "[pip] standalone runner requirements"
Install-Into -Python $VenvPython -Uv $Uv -Arguments @("-r", (Join-Path $ScriptDir "requirements-standalone.txt"))

if ($Leaderboard) {
    # A second, separate environment. The leaderboard's dependencies do not build
    # for 3.12, but PyPI ships the same carla 0.9.16 client for cp310, so both
    # environments drive the identical server.
    New-CarlaVenv -Path $VenvLeaderboardRoot -Version "3.10" -Uv $Uv
    $LbPython = Join-Path $VenvLeaderboardRoot "Scripts\python.exe"
    Install-Into -Python $LbPython -Uv $Uv -Arguments @("--upgrade", "wheel")
    Write-Host "[pip] carla 0.9.16 client for CPython 3.10"
    Install-Into -Python $LbPython -Uv $Uv -Arguments @("carla==0.9.16")
    Write-Host "[pip] relaxed leaderboard dependency set"
    Install-Into -Python $LbPython -Uv $Uv -Arguments @("-r", (Join-Path $ScriptDir "requirements-leaderboard-py310.txt"))

    Write-Host "[patch] adapting the vendored leaderboard checkouts to Python 3.9+"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "apply_patches.ps1")
}

Write-Host ""
Write-Host "Setup finished."
foreach ($pair in @(@($VenvRoot, "3.12 standalone runner"), @($VenvLeaderboardRoot, "3.10 leaderboard"))) {
    $mark = if (Test-Path (Join-Path $pair[0] "Scripts\python.exe")) { "ready  " } else { "MISSING" }
    Write-Host ("  {0}  {1}  {2}" -f $mark, $pair[1], $pair[0])
}
Write-Host ""
Write-Host "Standalone runner (Python 3.12):"
Write-Host "  . .\scripts\env.ps1"
Write-Host "  .\scripts\start_carla.ps1              (leave running)"
Write-Host "  python .\verify\verify_setup.py        (in another shell)"
Write-Host "  .\scripts\run_experiment.ps1"
if ($Leaderboard) {
    Write-Host ""
    Write-Host "Official leaderboard (Python 3.10). Town12/Town13 are required:"
    Write-Host "  .\scripts\get_additional_maps.ps1     (once, if Town12 is missing)"
    Write-Host "  . .\scripts\env.ps1 -Leaderboard"
    Write-Host "  .\scripts\run_leaderboard.ps1 -AgentConfig agent\configs\k4_workspace.json"
}
