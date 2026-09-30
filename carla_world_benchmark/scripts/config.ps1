# User-editable settings. Edit these first, then dot-source env.ps1 in any shell
# where you want to run an experiment.

# The CARLA package shipped with this repo (7.8 GB zip, 19.4 GB extracted).
$CarlaZip = Join-Path $PSScriptRoot "CARLA_Latest.zip"

# Where that zip is extracted. Already extracted here, so this points at it.
# It is 19.4 GB and sits inside the project folder; .gitignore excludes it.
$CarlaRoot = Join-Path (Split-Path -Parent $PSScriptRoot) "CARLA_Latest"

# Two interpreters, because the two experiments need different clients.
#
#   3.12  the wheel shipped inside CARLA_Latest.zip
#         (carla-0.9.16-cp312-cp312-win_amd64.whl) -> runner/carla_runner.py
#   3.10  PyPI also publishes carla 0.9.16 for cp310, and the leaderboard stack
#         needs an interpreter that its dependencies still build for
#         -> the official leaderboard evaluator
#
# Both talk to the same 0.9.16 server, so only one CARLA install is needed.
$PythonLauncherArgs = "-3.12"
$LeaderboardPythonArgs = "-3.10"

# Virtual environment locations.
$VenvRoot = "$env:USERPROFILE\.venvs\carla_py312"
$VenvLeaderboardRoot = "$env:USERPROFILE\.venvs\carla_lb_py310"

# CARLA server settings.
$CarlaPort = 2000
$TrafficManagerPort = 8000
$CarlaQuality = "Epic"
