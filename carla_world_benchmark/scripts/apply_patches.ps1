# Windows counterpart of apply_patches.sh: make the vendored leaderboard-2.0 and
# scenario_runner-2.0 checkouts run on modern Python. Idempotent.
#
#     powershell -ExecutionPolicy Bypass -File .\scripts\apply_patches.ps1
#
# ElementTree.Element.getchildren() was REMOVED in Python 3.9, and these branches
# still call it, so no routes file parses on 3.9+. list(elem) is the documented
# replacement and behaves identically on 3.7, so this is safe for every setup.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchRoot = Split-Path -Parent $ScriptDir

$Targets = @(
    "external\leaderboard\leaderboard\utils\route_parser.py",
    "external\scenario_runner\srunner\tools\scenario_parser.py",
    "external\scenario_runner\srunner\tools\route_parser.py"
)

$changed = 0
foreach ($rel in $Targets) {
    $path = Join-Path $BenchRoot $rel
    if (-not (Test-Path $path)) { Write-Host "SKIP  $rel (not present)"; continue }
    $text = [System.IO.File]::ReadAllText($path)
    if ($text -match '\.getchildren\(\)') {
        $patched = [regex]::Replace($text, '([A-Za-z_][A-Za-z0-9_]*)\.getchildren\(\)', 'list($1)')
        [System.IO.File]::WriteAllText($path, $patched)
        Write-Host "PATCH $rel  (getchildren -> list)"
        $changed++
    } else {
        Write-Host "OK    $rel  (already patched)"
    }
}

$remaining = Select-String -Path (Join-Path $BenchRoot "external\*\*\*.py") -Pattern '\.getchildren\(\)' -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "$changed file(s) patched"
