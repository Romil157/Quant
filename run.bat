@echo off
REM Quant Research Platform - Run Script
REM This script syncs dependencies and runs the quant platform CLI

echo =========================================
echo Quant Research Platform
echo =========================================
echo.

echo [1/2] Syncing dependencies with uv...
uv sync
if errorlevel 1 (
    echo ERROR: Failed to sync dependencies
    pause
    exit /b 1
)

echo.
echo [2/2] Starting Quant platform...
echo.

uv run python -m quant %*

echo.
echo =========================================
echo Done.
echo =========================================
pause