<#
.SYNOPSIS
    Layered verification of the real Windows execution path.

.DESCRIPTION
    Each layer is checked and recorded separately, so a failure says which layer
    broke instead of collapsing into "it did not run":

        1 shell      : this script is executing at all
        2 workspace  : the repository directory is visible from Windows
        3 python     : a Windows python interpreter starts
        4 pytest     : pytest is importable by that interpreter
        5 suite      : the test suite runs, with counted results

    Writes verification\local_windows_result.json (machine readable, consumed by
    environment_verification.py) and verification\local_windows_result.log (raw
    output).  `verified` is true only when every layer passed AND pytest reported
    a non-zero test count -- an environment failure is never counted as a pass.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\coding_world_benchmark\verify_local_windows.ps1
#>
[CmdletBinding()]
param(
    [string]$Repository = "",
    [string]$Package = "coding_world_benchmark"
)

$ErrorActionPreference = "Continue"
# The suite prints Japanese; without this the capture turns into mojibake or dies.
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

if (-not $Repository) {
    $Repository = Split-Path -Parent $PSScriptRoot
}

$layers = [ordered]@{}
$log = New-Object System.Collections.ArrayList

function Add-Layer {
    param([string]$Name, [bool]$Ok, [string]$Detail)
    $layers[$Name] = [ordered]@{ ok = $Ok; detail = $Detail }
    [void]$log.Add("[$(if ($Ok) { 'OK ' } else { 'NG ' })] $Name : $Detail")
}

# --- 1 shell ---------------------------------------------------------------
Add-Layer -Name "shell" -Ok $true -Detail "$($PSVersionTable.PSVersion) on $([System.Environment]::OSVersion.VersionString)"

# --- 2 workspace -----------------------------------------------------------
$packagePath = Join-Path $Repository $Package
$workspaceOk = Test-Path -LiteralPath $packagePath
$testFileCount = 0
if ($workspaceOk) {
    $testFileCount = @(Get-ChildItem -LiteralPath $packagePath -Filter "test_*.py" -File).Count
}
Add-Layer -Name "workspace" -Ok $workspaceOk -Detail "$packagePath (test files: $testFileCount)"

# --- 3 python --------------------------------------------------------------
$pythonExe = $null
$pythonVersion = ""
foreach ($candidate in @("python", "py -3", "python3")) {
    $parts = $candidate.Split(" ")
    $command = Get-Command $parts[0] -ErrorAction SilentlyContinue
    if (-not $command) { continue }
    try {
        $output = & $parts[0] @($parts[1..($parts.Length - 1)] + @("--version")) 2>&1 | Out-String
    } catch {
        continue
    }
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = $candidate
        $pythonVersion = $output.Trim()
        break
    }
}
Add-Layer -Name "python" -Ok ([bool]$pythonExe) -Detail $(if ($pythonExe) { "$pythonExe -> $pythonVersion" } else { "no working interpreter found" })

# --- 4 pytest --------------------------------------------------------------
$pytestOk = $false
$pytestVersion = ""
if ($pythonExe -and $workspaceOk) {
    $parts = $pythonExe.Split(" ")
    $arguments = @($parts[1..($parts.Length - 1)]) + @("-m", "pytest", "--version")
    $pytestVersion = (& $parts[0] @arguments 2>&1 | Out-String).Trim()
    $pytestOk = ($LASTEXITCODE -eq 0)
}
Add-Layer -Name "pytest" -Ok $pytestOk -Detail $pytestVersion

# --- 5 suite ---------------------------------------------------------------
$passed = 0
$failed = 0
$errors = 0
$suiteOutput = ""
$exitCode = $null
if ($pytestOk) {
    Push-Location $Repository
    $parts = $pythonExe.Split(" ")
    $arguments = @($parts[1..($parts.Length - 1)]) + @("-m", "pytest", $Package, "-q")
    $suiteOutput = (& $parts[0] @arguments 2>&1 | Out-String)
    $exitCode = $LASTEXITCODE
    Pop-Location
    foreach ($pattern in @(@("passed", [ref]$passed), @("failed", [ref]$failed), @("error", [ref]$errors))) {
        $match = [regex]::Match($suiteOutput, "(\d+)\s+$($pattern[0])")
        if ($match.Success) { $pattern[1].Value = [int]$match.Groups[1].Value }
    }
    $counted = $passed + $failed + $errors
    Add-Layer -Name "suite" -Ok ($counted -gt 0) -Detail "passed=$passed failed=$failed errors=$errors exit=$exitCode"
} else {
    Add-Layer -Name "suite" -Ok $false -Detail "skipped: pytest layer did not pass"
}

# --- result ----------------------------------------------------------------
$allOk = $true
foreach ($key in $layers.Keys) { if (-not $layers[$key].ok) { $allOk = $false } }

$result = [ordered]@{
    environment    = "local_windows"
    device         = $env:COMPUTERNAME
    date           = (Get-Date).ToString("yyyy-MM-dd")
    timestamp      = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    repository     = $Repository
    test_file_count = $testFileCount
    python_version = $pythonVersion
    pytest_version = $pytestVersion
    layers         = $layers
    passed         = $passed
    failed         = $failed
    errors         = $errors
    exit_code      = $exitCode
    verified       = ($allOk -and (($passed + $failed + $errors) -gt 0))
}

$outputDirectory = Join-Path $packagePath "verification"
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}
$jsonPath = Join-Path $outputDirectory "local_windows_result.json"
$logPath = Join-Path $outputDirectory "local_windows_result.log"

$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($jsonPath, ($result | ConvertTo-Json -Depth 6), $utf8)
[System.IO.File]::WriteAllText($logPath, (($log -join "`r`n") + "`r`n`r`n" + $suiteOutput), $utf8)

Write-Host ""
foreach ($line in $log) { Write-Host $line }
Write-Host ""
Write-Host "verified = $($result.verified)  (passed=$passed failed=$failed errors=$errors)"
Write-Host "result   : $jsonPath"
if (-not $result.verified) { exit 1 }
