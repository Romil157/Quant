#!/usr/bin/env bash
# Quant Research Platform launcher (macOS / Linux).
# Mirrors run.bat — pick a task and it runs it with the best available Python.

set -u

run_with_uv () {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
    return 0
  fi
  return 1
}

run_python () {
    if run_with_uv "$@"; then
        return 0
    fi
    if [ -x .venv/bin/python ]; then
        .venv/bin/python "$@"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        python3 "$@"
        return 0
    fi
    echo "No Python interpreter found. Install uv or set up .venv." >&2
    return 1
}

menu () {
  clear
  cat <<EOF
===================================================
           Quant Research Platform
===================================================

Please select an option:

  [1] Start Production API Server (http://localhost:8000/docs)
  [2] View Platform Status and CLI Config
  [3] Run Backtest Simulation
  [4] Download Market Data (Mock / Parquet / yfinance)
  [5] Run Unit and Security Test Suite (Pytest)
  [6] Run Code Linter and Type Checker (Ruff and Mypy)
  [7] Launch Streamlit Dashboard
  [8] Exit
EOF
  printf "Enter choice [1-8]: "
  read -r choice
  case "$choice" in
    1) api ;;
    2) cli ;;
    3) backtest ;;
    4) download ;;
    5) tests ;;
    6) lint ;;
    7) dashboard ;;
    8) exit 0 ;;
    *) echo "Invalid choice."; sleep 1 ;;
  esac
  echo
  echo "Press any key to return to the menu..."
  read -r
  menu
}

api () {
  echo "Starting Production API Server..."
  echo "Swagger docs: http://localhost:8000/docs  (Ctrl+C to stop)"
  run_python -m quant.production.api
}

cli () {
  echo "Running Quant CLI status and config..."
  run_python -m quant show-config
}

backtest () {
  echo "Running backtest simulation (buy_and_hold)..."
  run_python scripts/run_backtest.py --config configs/backtest.yaml --strategy buy_and_hold
}

download () {
  echo "Downloading sample market data (mock)..."
  run_python scripts/download_data.py --symbols AAPL MSFT GOOGL \
      --start 2023-01-01 --end 2023-12-31 --provider mock
}

tests () {
  echo "Running unit and security test suite..."
  run_python -m pytest
}

lint () {
  echo "Running Ruff and Mypy..."
  if command -v uv >/dev/null 2>&1; then
    uv run ruff check .
    uv run mypy src/quant
    return 0
  fi
  if [ -x .venv/bin/ruff ]; then
    .venv/bin/ruff check .
    .venv/bin/mypy src/quant
    return 0
  fi
  ruff check .
  mypy src/quant
}

dashboard () {
  echo "Launching Streamlit dashboard at http://localhost:8501 ..."
  if command -v uv >/dev/null 2>&1; then
    uv run streamlit run dashboard/app.py
    return 0
  fi
  if [ -x .venv/bin/streamlit ]; then
    .venv/bin/streamlit run dashboard/app.py
    return 0
  fi
  streamlit run dashboard/app.py
}

menu
