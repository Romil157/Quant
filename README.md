# Quant Research Platform

A production‑quality quantitative finance research and backtesting platform built for a single developer. Runs natively on Windows with Python, PowerShell, Parquet, DuckDB/SQLite.

## Quick start (PowerShell)

```powershell
cd C:\Users\ROMIL\Desktop\Quant
uv sync
uv run pytest
uv run python -m quant
```

## Project layout

```
Quant/
├─ configs/            # YAML configuration files
├─ data/               # Raw, processed, cache, metadata (Parquet + DuckDB)
├─ notebooks/          # Exploratory notebooks
├─ reports/            # Generated research reports
├─ scripts/            # Helper CLI scripts
├─ src/quant/          # Core package
│   ├─ data/           # Providers, loaders, validation, cleaning, schemas
│   ├─ features/       # Technical, statistical, volatility, microstructure
│   ├─ factors/        # Factor research
│   ├─ signals/        # Signal generation
│   ├─ strategies/     # Alpha strategies
│   ├─ portfolio/      # Portfolio construction & position sizing
│   ├─ risk/           # Risk engine & stress testing
│   ├─ backtest/       # Event‑driven backtester
│   ├─ models/         # ML models
│   ├─ research/       # Walk‑forward, experiment tracking, reports
│   ├─ execution/      # Broker abstractions, paper/live
│   ├─ analytics/      # Performance analytics
│   ├─ api/            # FastAPI services
│   ├─ dashboard/      # Streamlit dashboard
│   ├─ config/         # Configuration loading
│   └─ utils/          # Shared utilities
└─ tests/              # Unit, integration, backtest tests
```

## Configuration

Edit `configs/development.yaml` (or `research.yaml`, `backtest.yaml`, `paper.yaml`) to change parameters. All secrets go into a local `.env` file (never committed).

## License

Proprietary – internal use only.