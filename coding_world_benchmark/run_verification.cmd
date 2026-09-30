@echo off
REM Double-click entry point for the local Windows verification run.
REM Writes coding_world_benchmark\verification\local_windows_result.json
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0verify_local_windows.ps1"
echo.
pause
