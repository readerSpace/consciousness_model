# Windows-side helper for the WSL2 route. Checks that WSL2 and the GPU are in
# place, then hands over to the Linux scripts.
#
#     powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_wsl.ps1
#     powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_wsl.ps1 -Distro Ubuntu
#     powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_wsl.ps1 -SelfTest
#
# Everything after the preflight runs inside the distro; this script only gets
# you there and prints the exact commands.

param(
    [string]$Distro = "",
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"

function ConvertTo-CleanLines {
    <#
      wsl.exe writes UTF-16LE unless WSL_UTF8 is set. Captured into a PowerShell
      variable that decodes as UTF-8 or ANSI, every character comes back followed
      by a NUL: the string prints as "Ubuntu" but is not equal to "Ubuntu", so
      `wsl -d $Distro` fails with WSL_E_DISTRO_NOT_FOUND. Strip the NULs even
      when WSL_UTF8 is set, for WSL builds that predate that variable.
    #>
    param([object[]]$Raw)
    $out = @()
    foreach ($line in $Raw) {
        if ($null -eq $line) { continue }
        # [string] cast picks String.Replace(string,string); the char overload
        # rejects an empty replacement.
        $text = ($line | Out-String).Replace([string][char]0, '').Trim()
        if ($text) { $out += $text }
    }
    return ,$out
}

function ConvertTo-WslPath {
    <#
      C:\Users\x\proj  ->  /mnt/c/Users/x/proj
      Done here rather than through `wslpath`, because a failed wslpath call
      prints its error to stdout and would be captured as if it were the path --
      which is exactly how a broken distro name turned into a "directory" named
      after an error message. Also avoids an encoding hop for non-ASCII names.
    #>
    param([string]$WindowsPath)
    if ($WindowsPath -match '^\\\\') {
        throw "UNC paths are not reachable as /mnt/<drive>: $WindowsPath"
    }
    if ($WindowsPath -notmatch '^([A-Za-z]):[\\/](.*)$') {
        throw "Not an absolute Windows path with a drive letter: $WindowsPath"
    }
    $drive = $Matches[1].ToLower()
    $rest = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$rest".TrimEnd('/')
}

if ($SelfTest) {
    # Pure-logic checks. Run these anywhere, including on Linux PowerShell.
    $failures = 0
    function Assert-Equal($expected, $actual, $label) {
        if ($expected -ceq $actual) { Write-Host "PASS  $label" }
        else { Write-Host "FAIL  $label`n      expected [$expected]`n      actual   [$actual]"; $script:failures++ }
    }
    function Assert-Throws($block, $label) {
        try { & $block; Write-Host "FAIL  $label (no exception)"; $script:failures++ }
        catch { Write-Host "PASS  $label" }
    }

    $mangled = "U`0b`0u`0n`0t`0u`0"
    Assert-Equal "Ubuntu" (ConvertTo-CleanLines @($mangled))[0] "UTF-16 remnant NULs are stripped"
    Assert-Equal 2 (ConvertTo-CleanLines @("Ubuntu", "", "  ", "rancher-desktop")).Count "blank lines dropped"
    Assert-Equal "/mnt/c/Users/neko5/Documents/projects/carla_world_benchmark" `
        (ConvertTo-WslPath "C:\Users\neko5\Documents\projects\carla_world_benchmark") "drive path converted"
    Assert-Equal "/mnt/d/work" (ConvertTo-WslPath "D:\work\") "trailing separator trimmed"
    Assert-Equal "/mnt/c/x/意識モデル/y" (ConvertTo-WslPath "C:\x\意識モデル\y") "non-ASCII path preserved"
    Assert-Throws { ConvertTo-WslPath "\\server\share\x" } "UNC path rejected"
    Assert-Throws { ConvertTo-WslPath "relative\path" } "relative path rejected"

    Write-Host ""
    Write-Host "$failures failure(s)"
    exit $failures
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchRoot = Split-Path -Parent $ScriptDir

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    throw "wsl.exe not found. Install WSL2 with:  wsl --install"
}

$env:WSL_UTF8 = "1"
$PreviousEncoding = [Console]::OutputEncoding
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

function Get-WslLines([string[]]$WslArgs) {
    return ConvertTo-CleanLines (& wsl @WslArgs 2>&1)
}

try {
    Write-Host "[wsl] installed distributions:"
    & wsl --list --verbose

    $Names = Get-WslLines @("--list", "--quiet")
    if (-not $Names) { throw "No WSL distribution installed. Run:  wsl --install -d Ubuntu-22.04" }

    if ($Distro) {
        if ($Names -notcontains $Distro) {
            throw "Distribution '$Distro' is not installed. Available: $($Names -join ', ')"
        }
    } else {
        # rancher-desktop ships two service distros that cannot host this build.
        $Usable = @($Names | Where-Object { $_ -notlike "rancher-desktop*" })
        if (-not $Usable) { $Usable = $Names }
        $Distro = $Usable[0]
    }
    Write-Host "[wsl] using distribution: $Distro"

    $WslBench = ConvertTo-WslPath ([System.IO.Path]::GetFullPath($BenchRoot))
    Write-Host "[wsl] benchmark root inside WSL: $WslBench"

    # Confirm the distro can actually see it before trusting the path.
    $Probe = Get-WslLines @("-d", $Distro, "--", "bash", "-lc",
                            "test -f '$WslBench/scripts/wsl/00_check.sh' && echo FOUND || echo MISSING")
    if ($Probe -notcontains "FOUND") {
        throw ("Could not reach $WslBench from $Distro (got: $($Probe -join ' ')). " +
               "If the Windows drive is not mounted there, check /etc/wsl.conf, " +
               "then run 'wsl --shutdown' and retry.")
    }

    Write-Host ""
    Write-Host "[wsl] running the preflight..."
    & wsl -d $Distro -- bash -lc "cd '$WslBench' && bash scripts/wsl/00_check.sh"
    $PreflightExit = $LASTEXITCODE

    Write-Host ""
    if ($PreflightExit -ne 0) {
        Write-Host "Preflight reported failures. Fix those before downloading tens of GB of package."
    } else {
        Write-Host "Preflight clean. Continue inside WSL:"
    }
    Write-Host ""
    Write-Host "  wsl -d $Distro"
    Write-Host "  cd '$WslBench'"
    Write-Host "  bash scripts/wsl/01_setup.sh"
    Write-Host "  source scripts/wsl/env.sh"
    Write-Host "  bash scripts/wsl/start_carla.sh        # leave running"
    Write-Host "  # then in a second WSL shell:"
    Write-Host "  cd '$WslBench' && source scripts/wsl/env.sh && python verify/verify_setup.py"
    Write-Host "  bash scripts/wsl/run_ablation.sh"
}
finally {
    try { [Console]::OutputEncoding = $PreviousEncoding } catch { }
}
