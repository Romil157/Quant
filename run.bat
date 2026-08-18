@echo off
title Quant Research Platform
cls

:MENU
cls
echo ===================================================
echo           Quant Research Platform
echo ===================================================
echo.
echo Please select an option:
echo.
echo   [1] Start Production API Server (http://localhost:8000/docs)
echo   [2] View Platform Status and CLI Config
echo   [3] Run Backtest Simulation
echo   [4] Download Market Data (Mock / Parquet / yfinance)
echo   [5] Run Unit and Security Test Suite (Pytest)
echo   [6] Run Code Linter and Type Checker (Ruff and Mypy)
echo   [7] Launch Streamlit Dashboard (http://localhost:8501)
echo   [8] Exit
echo.
echo ===================================================
set /p CHOICE="Enter choice [1-8]: "

if "%CHOICE%"=="1" goto API
if "%CHOICE%"=="2" goto CLI
if "%CHOICE%"=="3" goto BACKTEST
if "%CHOICE%"=="4" goto DOWNLOAD
if "%CHOICE%"=="5" goto TEST
if "%CHOICE%"=="6" goto LINT
if "%CHOICE%"=="7" goto DASHBOARD
if "%CHOICE%"=="8" goto EXIT

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

:API
cls
echo Starting Production API Server...
echo API Swagger Docs available at http://localhost:8000/docs
echo Press Ctrl+C to stop the server.
echo.
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run python -m quant.production.api
    goto DONE
)
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe -m quant.production.api
    goto DONE
)
python -m quant.production.api
goto DONE

:CLI
cls
echo Running Quant CLI Status and Config...
echo.
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run python -m quant show-config
    goto DONE
)
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe -m quant show-config
    goto DONE
)
python -m quant show-config
goto DONE

:BACKTEST
cls
echo Running Backtest Simulation...
echo.
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run python scripts/run_backtest.py --config configs/backtest.yaml --strategy buy_and_hold
    goto DONE
)
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe scripts/run_backtest.py --config configs/backtest.yaml --strategy buy_and_hold
    goto DONE
)
python scripts/run_backtest.py --config configs/backtest.yaml --strategy buy_and_hold
goto DONE

:DOWNLOAD
cls
echo Downloading Sample Market Data...
echo.
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run python scripts/download_data.py --symbols AAPL MSFT GOOGL --start 2023-01-01 --end 2023-12-31 --provider mock
    goto DONE
)
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe scripts/download_data.py --symbols AAPL MSFT GOOGL --start 2023-01-01 --end 2023-12-31 --provider mock
    goto DONE
)
python scripts/download_data.py --symbols AAPL MSFT GOOGL --start 2023-01-01 --end 2023-12-31 --provider mock
goto DONE

:TEST
cls
echo Running Unit and Security Test Suite...
echo.
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run pytest
    goto DONE
)
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe -m pytest
    goto DONE
)
python -m pytest
goto DONE

:LINT
cls
echo Running Ruff Linter and Mypy Type Checker...
echo.
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run ruff check .
    uv run mypy src/quant
    goto DONE
)
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\ruff.exe check .
    .\.venv\Scripts\mypy.exe src/quant
    goto DONE
)
ruff check .
mypy src/quant
goto DONE

:DASHBOARD
cls
echo Launching Streamlit Dashboard at http://localhost:8501 ...
echo Press Ctrl+C to stop.
echo.
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run streamlit run dashboard/app.py
    goto DONE
)
if exist ".\.venv\Scripts\streamlit.exe" (
    .\.venv\Scripts\streamlit.exe run dashboard/app.py
    goto DONE
)
streamlit run dashboard/app.py
goto DONE

:DONE
echo.
echo ===================================================
echo Press any key to return to menu...
pause >nul
goto MENU

:EXIT
echo Exiting...
exit /b 0