# Quant — Quantitative Finance Research & Backtesting Platform

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/Ruff-checked-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Mypy](https://img.shields.io/badge/Mypy-checked-2E5C9D?logo=python&logoColor=white)](https://mypy-lang.org/)
[![Pytest](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red)](#license)

A modular, production-hardened quantitative-finance research, backtesting, and production engine built with Python 3.12, FastAPI, Pandas/Polars, PyArrow, scikit-learn, and Streamlit. Designed for high performance, modularity, and strict operational security.

---

## Table of contents

- [Key features](#key-features)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Configuration reference](#configuration-reference)
- [Command line interface (CLI)](#command-line-interface-cli)
- [Helper scripts](#helper-scripts)
- [Built-in strategies](#built-in-strategies)
- [Production REST API](#production-rest-api)
- [Streamlit dashboard](#streamlit-dashboard)
- [Quality assurance & testing](#quality-assurance--testing)
- [Security model](#security-model--audit-compliance)
- [Architecture overview](#architecture-overview)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Key features

- **Event-driven backtester** — In-memory engine with realistic transaction-cost modeling (commission, spread, slippage, market impact), portfolio tracking, drawdown-aware risk reduction, and `210`+ test scenarios.
- **Real portfolio construction** — Rolling volatility and covariance powering Inverse Volatility, Volatility Targeting, Risk Parity, Minimum Variance, Mean-Variance, and Maximum Sharpe optimizers.
- **Built-in strategies** — `buy_and_hold`, `momentum`, `mean_reversion`, `breakout` (Donchian + ATR), `macd` (trend-filtered), `dual_momentum` (Antonacci), and `pair_trading` (z-score). Dispatched from a single `STRATEGY_REGISTRY` used by the CLI scripts, the API, and the dashboard.
- **Machine-learning engine** — Time-series cross-validation (`TimeSeriesCV`), feature engineering pipelines, online learning ensembles (`OnlineEnsemble`), SGD-based Passive-Aggressive (sklearn 1.10-proof), drift detection, and automated model comparison.
- **Portfolio & risk management** — Multi-asset portfolio construction (Equal Weight, Risk Parity, Inverse Volatility, Minimum Variance, Volatility Targeting, Mean-Variance, Maximum Sharpe) with VaR, CVaR, and stress testing.
- **Production REST API** — FastAPI server with security headers, API-key auth, request guardrails, TrustedHost middleware, and Prometheus metrics.
- **Walk-forward research pipeline** — Standardized walk-forward validation with parameter sensitivity, multiple-testing correction (Bonferroni), and HTML reports.
- **Cross-strategy benchmark** — One-command comparison of all strategies on identical data/costs with HTML report.
- **Cross-platform launcher** — `run.bat` (Windows) and `run.sh` (macOS/Linux) menus for backtest, research, download, tests, lint, dashboard, and API.
- **Research dashboard** — Streamlit app rendering equity curve, drawdown, daily-returns distribution, and headline risk metrics from the analytics module.
- **Security & secret redaction** — Automatic secret masking in logs/representations, CORS hardening, path-traversal protection, and safe YAML parsing.
- **Monitoring & alerting** — Structured logging (`structlog`), health-check system, job scheduler (`APScheduler`), and configurable alert triggers.
- **Look-ahead bias free** — All features, signals, and engine components audited and verified free of look-ahead bias.

---

## Project structure

```
Quant/
├─ .github/workflows/ci.yml    # CI matrix (lint + typecheck + tests) on ubuntu+windows
├─ configs/                    # YAML configs (dev, backtest, paper, research)
├─ data/                       # Raw, processed, cache, and metadata (Parquet + DuckDB)
├─ dashboard/                  # Streamlit research dashboard
│   └─ app.py
├─ docs/                       # Documentation
│   ├─ RESEARCH_STANDARD.md
│   ├─ LOOKAHEAD_AUDIT.md
│   ├─ ALERTING.md
│   ├─ DRIFT_POLICY.md
│   └─ PAPER_TRADING.md
├─ models/                     # Persisted ML artifacts
├─ notebooks/                  # Exploratory research notebooks
├─ reports/                    # Generated performance & backtest reports
├─ scripts/                    # CLI scripts (download, validate, backtest, research, benchmark, report)
├─ tests/                      # Unit + integration + security test suite (~248 tests)
│   ├─ unit/
│   └─ integration/            # Cross-module flows
├─ src/quant/                  # Core platform package
│   ├─ __init__.py
│   ├─ __main__.py             # `python -m quant`
│   ├─ cli.py                  # Typer CLI
│   ├─ analytics/              # Performance & risk factor calcs (Sharpe, Sortino, VaR, beta)
│   ├─ backtest/               # Event-driven engine, execution cost sim, types
│   ├─ brokers/                # Execution adapters (Alpaca, base broker interface)
│   ├─ config/                 # Pydantic + YAML configuration loader
│   ├─ data/                   # Providers (mock, parquet, yfinance), validation, cleaning
│   ├─ features/               # Microstructure, statistical, technical, volatility
│   ├─ ml/                     # Models, TimeSeriesCV, feature pipeline, online learning
│   ├─ paper/                  # Paper trading engine, state, data feed, risk
│   ├─ portfolio/              # Portfolio optimization & constraint engine
│   ├─ production/             # FastAPI server, monitoring, scheduler, alerting, config
│   ├─ research/               # Walk-forward analysis, experiment tracker, report generator
│   ├─ risk/                   # Risk engine, VaR/CVaR, and stress testing
│   └─ strategies/             # Built-in alpha strategies & signal generation
├─ run.bat                     # Windows launcher (menu)
├─ run.sh                      # macOS / Linux launcher (menu)
├─ pyproject.toml
├─ uv.lock
└─ README.md
```

---

## Quick start

```powershell
git clone <your-repo-url> Quant
cd Quant
uv sync --all-extras --dev           # installs runtime + dev + yfinance optional
uv run python -m quant hello         # health check
uv run pytest                        # ~220 tests
uv run python -m quant.production.api  # browse http://localhost:8000/docs
```

Or, with the launcher menu:

```powershell
.\run.bat          # Windows
./run.sh           # macOS / Linux
```

---

## Installation

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or a standard `venv`

### uv (recommended)

```powershell
uv sync --all-extras --dev           # core + dev + yfinance optional extra
```

### Standard virtualenv

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[live]"             # zsh/PowerShell users: keep the quotes on -e ".[live]"
# Optional dev tools:
pip install ruff mypy pytest pytest-cov
```

> The optional `live` extra pulls in `yfinance` so the `yfinance` data provider works offline-free. Without it, `download_data --provider yfinance` raises a clear `ImportError`. The default `mock` provider works out of the box.

### Environment variables

Copy `.env.example` to `.env` and configure as needed:

```powershell
Copy-Item .env.example .env
```

---

## Configuration reference

The platform reads settings from dataclass defaults, YAML files under `configs/`, and environment variable overrides. All sensitive fields (`password`, `secret_key`, `api_key`, `sentry_dsn`) are masked with `***` in string representations, JSON exports, and logs.

| Environment variable      | Description                                                          | Default                                                  |
| :------------------------ | :------------------------------------------------------------------- | :------------------------------------------------------- |
| `QUANT_API_KEY` (& aliases `API_KEY`, `SECRET_KEY` for JWT) | API key required on protected endpoints when set | unset (disabled in development; **required** in production) |
| `CORS_ORIGINS`            | Comma-separated list of allowed CORS origins                          | `http://localhost:3000,http://localhost:8000`            |
| `DB_HOST` / `DB_PORT`     | PostgreSQL host / port                                                | `localhost` / `5432`                                     |
| `DB_NAME` / `DB_USER` / `DB_PASSWORD` | PostgreSQL connection credentials                       | `quant` / `quant` / `""`                                 |
| `REDIS_HOST` / `REDIS_PORT` | Redis connection                                                     | `localhost` / `6379`                                     |
| `API_HOST` / `API_PORT`   | API server bind                                                       | `0.0.0.0` / `8000`                                       |
| `ENVIRONMENT`             | `development` or `production` (production enforces auth + TrustedHost) | `development`                                            |
| `DEBUG`                   | Boolean string (`true` / `false`)                                    | `false`                                                  |
| `LOG_LEVEL`               | `DEBUG` / `INFO` / `WARNING` / `ERROR`                               | `INFO`                                                   |
| `SENTRY_DSN`              | Sentry DSN monitored by `MonitoringConfig`                           | unset                                                    |

---

## Command line interface (CLI)

```powershell
uv run python -m quant --help
uv run python -m quant hello
uv run python -m quant show-config --config-path configs/development.yaml
```

Commands:

- `hello` — simple health check.
- `show-config` — load a YAML config and pretty-print the resulting `AppConfig`.

---

## Helper scripts

All scripts are real (no `TODO` stubs). Use `--help` for full options.

```powershell
# 1) Download data (mock / parquet / yfinance)
uv run python scripts/download_data.py --symbols AAPL MSFT --start 2023-01-01 --end 2023-12-31 --provider mock

# 2) Validate data quality (prints per-validator PASS / FAIL)
uv run python scripts/validate_data.py --symbols AAPL --start 2023-01-01 --end 2023-12-31

# 3) Run a backtest (real engine; prints summary + report path when --report)
uv run python scripts/run_backtest.py --config configs/backtest.yaml --strategy buy_and_hold
uv run python scripts/run_backtest.py --strategy momentum --lookback 63 --top-n 3 --symbols AAPL MSFT

# 4) Run walk-forward validation (cost-realistic, parameter sweep, HTML report)
uv run python scripts/run_backtest.py --strategy momentum --walk-forward --start 2020-01-01 --end 2023-12-31 --symbols AAPL MSFT GOOGL

# 5) Run a walk-forward experiment (prints per-fold metrics + experiment id when tracking enabled)
uv run python scripts/run_research.py --config configs/research.yaml --strategy dual_momentum

# 6) Generate an HTML report from saved artifacts
uv run python scripts/generate_report.py --results reports/experiments/<exp_id> --format html

# 7) Cross-strategy benchmark (all strategies, identical pipeline, HTML report)
uv run python scripts/run_benchmark.py --start 2020-01-01 --end 2023-12-31 --symbols SPY QQQ IWM
```

---

## Built-in strategies

All strategies live in `quant.strategies` and dispatch from `STRATEGY_REGISTRY`.

| Registry name   | Class                       | Description                                                            |
| :-------------- | :-------------------------- | :-------------------------------------------------------------------- |
| `buy_and_hold`  | `BuyAndHoldStrategy`        | Equal-weight long on every symbol, established on the first bar.      |
| `momentum`      | `MomentumStrategy`          | Time-series momentum across one or more lookback windows.             |
| `mean_reversion`| `MeanReversionStrategy`     | RSI + Bollinger Band + z-score entry, mean-reversion exit logic.       |
| `breakout`      | `BreakoutStrategy`          | Donchian-channel breakout with ATR trailing stop.                     |
| `macd`          | `MACDMomentumStrategy`      | MACD crossover filtered by a slow moving-average trend filter.        |
| `dual_momentum` | `DualMomentumStrategy`      | Antonacci dual-momentum: absolute + relative momentum with cash shell.|
| `pair_trading`  | `PairTradingStrategy`       | Z-score of two-symbol price ratio; enters at |z|>entry_z.              |

Use them programmatically:

```python
from quant.strategies import create_strategy

strategy = create_strategy("momentum", lookback=63, top_n=5, rebalance_freq=21)
```

---

## Production REST API

```powershell
uv run python -m quant.production.api    # serves on 0.0.0.0:8000
```

### Endpoints

| Method | Path                                  | Auth      | Description                                   |
| :----- | :------------------------------------ | :-------- | :-------------------------------------------- |
| GET    | `/health`                             | Public    | Aggregate health check.                       |
| GET    | `/health/{check_name}`               | Public    | Detailed per-check health.                     |
| GET    | `/metrics`                            | Public    | Prometheus text-format metrics.               |
| GET    | `/ready` / `/live`                    | Public    | Kubernetes readiness / liveness probes.         |
| GET    | `/docs` / `/redoc`                    | Public    | Swagger / ReDoc OpenAPI UIs.                  |
| POST   | `/api/v1/backtest`                    | Protected | Run an event-driven backtest (real engine).    |
| POST   | `/api/v1/ml/experiment`              | Protected | Train a single ML pipeline model.             |
| POST   | `/api/v1/ml/compare`                  | Protected | Compare multiple ML algorithms.               |
| POST   | `/api/v1/data/download`               | Protected | Download market data via a provider.          |
| GET    | `/api/v1/data/validate/{symbol}`      | Protected | Data quality validation.                     |
| GET    | `/api/v1/config`                      | Protected | Sanitized production configuration (secrets masked). |

### Authentication

When `QUANT_API_KEY` is set, protected endpoints require an `X-API-Key` header compared with `hmac.compare_digest` (constant-time). If `QUANT_API_KEY` is unset **and** `ENVIRONMENT=production`, the API server refuses to start (fail-closed). In development, auth is optional.

### Request guardrails

- Max symbols per request: **50**
- Max date range: **3,650 days** (10 years)
- Invalid ISO date / reversed range → `422 Unprocessable Content`
- Unknown strategy name → `400 Bad Request` listing the valid registry names

---

## Streamlit dashboard

```powershell
uv run streamlit run dashboard/app.py     # http://localhost:8501
```

The sidebar lets you pick symbols, date range, data provider (`mock`, `parquet`, `yfinance`), and a strategy. Tabs render the equity curve, drawdown, daily-returns distribution, risk metrics (VaR / CVaR / annualized volatility), and a strategy-card preview.

> Research-only — not a live trading UI.

---

## Quality assurance & testing

```powershell
# Lint
uv run ruff check .

# Type-check
uv run mypy src/quant

# Tests (~220 passing, including integration tests)
uv run pytest

# Coverage
uv run pytest --cov=quant --cov-report=term
```

CI (`.github/workflows/ci.yml`) runs the same matrix on `ubuntu-latest` and `windows-latest` for every push and pull request.

---

## Security model & audit compliance

- **Authentication**: Constant-time API key verification (`hmac.compare_digest`) on REST endpoints; fail-closed when `ENVIRONMENT=production` and no key.
- **TrustedHost middleware**: Wildcard `*` hosts rejected in production.
- **CORS hardening**: Empty or wildcard origins rejected in production when `allow_credentials=True`.
- **Input sanitization**: Path-traversal defense in `ParquetProvider` via `^[A-Za-z0-9_.-]+$` regex + path-relative-resolution checks.
- **Safe parsing**: `yaml.safe_load` used strictly for all configs.
- **Secret redaction**: `password`, `secret_key`, `api_key`, `sentry_dsn` masked in `__repr__`/`__str__`, JSON exports, and structured logs.
- **SQL injection**: `ExperimentTracker.update_experiment` uses a static whitelist of column combinations to build UPDATE statements (no dynamic identifier substitution).
- **Look-ahead bias audit**: All features, signals, and engine components verified free of look-ahead bias. See `docs/LOOKAHEAD_AUDIT.md`.
- **Ignore rules**: `.env`, `*.sqlite`, `*.duckdb`, `*.db`, `data/raw/**/*.parquet` all excluded via `.gitignore`.

---

## Architecture overview

```
                                ┌─────────────────┐
                                │ Data Providers   │
                                │  mock / parquet  │
                                │  / yfinance      │
                                └────────┬────────┘
                                         │ OHLCV
                                         ▼
                                ┌─────────────────┐
                                │ Features        │
                                │  technical      │
                                │  statistical    │
                                │  volatility     │
                                │  microstructure │
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Strategies      │
                                │  (registry)     │
                                └────────┬────────┘
                                         │ signals
                                         ▼
   ┌────────────────┐            ┌─────────────────┐
   │ ML pipeline    │◀──────────│ Backtest engine │
   │  cv / models / │  forecasts│ + Risk limits    │
   │  online / wf   │            │ + Portfolio     │
   └────────────────┘            └────────┬────────┘
                                          │ results
                                          ▼
                                 ┌─────────────────┐
                                 │ Research         │
                                 │  walk-forward    │
                                 │  experiment db   │
                                 │  HTML/MD report  │
                                 └────────┬────────┘
                                          │
                       ┌───────────────────┴────────────────────┐
                       ▼                                        ▼
                 ┌────────────────┐                  ┌────────────────┐
                 │ FastAPI         │                  │ Streamlit      │
                 │  + auth         │                  │ dashboard      │
                 │  + metrics      │                  └────────────────┘
                 │  + scheduler    │
                 │  + alerting     │
                 └────────────────┘
```

---

## Roadmap

- Paper-trading wiring (currently a `configs/paper.yaml` placeholder).
- Broker adapter SDKs (Interactive Brokers, Alpaca).
- Dashboard v2: live positions, P&L, and authenticated review of experiments.
- Native Polars path through features / backtest for large universes.
- `tests/integration/` suite for cross-module end-to-end flows.

---

## Contributing

Before opening a PR:

1. `uv run ruff check .` (auto-fixes: `uv run ruff check . --fix`)
2. `uv run mypy src/quant`
3. `uv run pytest -q`
4. Use conventional commit prefixes (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `security:`).
5. Branches: `feat/<topic>` or `fix/<topic>`.

---

## License

Proprietary — Internal use only.