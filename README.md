# Quant — Quantitative Finance Research & Backtesting Platform

A modular, production-hardened quantitative finance research, backtesting, and production engine built with Python 3.12, FastAPI, Pandas/Polars, and PyArrow. Designed for high performance, modularity, and strict operational security.

---

## Key Features

- **Event-Driven Backtester**: In-memory event engine with detailed transaction cost modeling (commission, spread, slippage, and market impact).
- **Machine Learning Engine**: Time-series cross-validation (`TimeSeriesCV`), feature engineering pipelines, online learning ensembles (`OnlineEnsemble`), and automated model comparison.
- **Portfolio & Risk Management**: Multi-asset portfolio construction (Equal Weight, Risk Parity, Inverse Volatility, Minimum Variance, Volatility Targeting) with VaR, CVaR, and stress testing.
- **Production REST API**: FastAPI server equipped with security headers, API key authentication, input guardrails, and Prometheus metrics.
- **Security & Secret Redaction**: Automatic secret masking in logs/representations, CORS origin hardening, path traversal protection, and safe YAML parsing.
- **Monitoring & Alerting**: Structured logging (`structlog`), health check system, job scheduler (`APScheduler`), and configurable alert triggers.

---

## Project Structure

```
Quant/
├─ configs/                  # YAML configuration files (dev, backtest, paper, research)
├─ data/                     # Raw, processed, cache, and metadata storage (Parquet + DuckDB)
├─ notebooks/                # Exploratory research notebooks
├─ reports/                  # Generated performance & backtest reports
├─ scripts/                  # Helper CLI scripts (download, validate, backtest, research)
├─ src/quant/                # Core platform package
│   ├─ backtest/             # Event-driven backtesting engine & execution cost simulator
│   ├─ config/               # YAML configuration loader
│   ├─ data/                 # Providers (Parquet, Mock), validation, & cleaning
│   ├─ features/             # Microstructure, statistical, technical, & volatility features
│   ├─ ml/                   # ML models, TimeSeriesCV, feature pipeline, online learning
│   ├─ portfolio/            # Portfolio optimization & constraint engine
│   ├─ production/           # Production API, config, monitoring, scheduler, & alerting
│   │   ├─ alerts.py         # Alert rules & notification handlers
│   │   ├─ api.py            # FastAPI production server with auth & guardrails
│   │   ├─ config.py         # Dataclass config hierarchy with secret masking
│   │   ├─ monitoring.py     # Structured logging & Prometheus metrics
│   │   └─ scheduler.py      # Automated task scheduler
│   ├─ research/             # Walk-forward analysis, experiment tracker, report generator
│   ├─ risk/                 # Risk engine, VaR/CVaR, and stress testing
│   └─ strategies/           # Alpha strategies & signal generation
└─ tests/                    # Unit, integration, & security test suite
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- `uv` (recommended package manager) or standard `venv`

### Installation

1. **Clone the repository and enter the directory:**
   ```powershell
   cd C:\Users\ROMIL\Desktop\Quant
   ```

2. **Sync dependencies using `uv`:**
   ```powershell
   uv sync
   ```
   *Alternatively, using standard virtualenv:*
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -e .[dev]
   ```

3. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and configure optional production variables:
   ```powershell
   Copy-Item .env.example .env
   ```

---

## Configuration & Environment Reference

The platform reads settings from dataclass defaults, YAML files under `configs/`, and environment variable overrides.

| Environment Variable | Description | Default |
| :--- | :--- | :--- |
| `QUANT_API_KEY` | API Key required for authenticated `/api/v1/*` routes | Disabled if unset |
| `CORS_ORIGINS` | Comma-separated list of allowed CORS domains | `http://localhost:3000,http://localhost:8000` |
| `DB_HOST` / `DB_PORT` | PostgreSQL database host and port | `localhost:5432` |
| `DB_USER` / `DB_PASSWORD`| Database connection credentials | `quant` / `""` |
| `SECRET_KEY` | JWT / Security secret key | `""` |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

*Note: All sensitive fields (`password`, `secret_key`, `api_key`, `sentry_dsn`) are automatically masked with `***` in string representations, JSON exports, and logs.*

---

## Usage

### 1. Command Line Interface (CLI)

Run the CLI module to verify configuration or inspect settings:
```powershell
uv run python -m quant hello
uv run python -m quant show-config --config-path configs/development.yaml
```

### 2. Helper Scripts

- **Download Market Data**:
  ```powershell
  uv run python scripts/download_data.py --symbols AAPL MSFT --start 2023-01-01 --end 2023-12-31 --provider mock
  ```

- **Run Backtest**:
  ```powershell
  uv run python scripts/run_backtest.py --config configs/backtest.yaml --strategy buy_and_hold
  ```

### 3. Production REST API Server

Start the production FastAPI server:
```powershell
uv run python -m quant.production.api
```

#### API Endpoints & Protection:

- **Public Endpoints**:
  - `GET /health` — Application health check
  - `GET /metrics` — Prometheus metrics export
  - `GET /ready` & `GET /live` — Kubernetes readiness/liveness probes
- **Protected Endpoints** (Requires `X-API-Key` header when `QUANT_API_KEY` is configured):
  - `POST /api/v1/backtest` — Run event-driven backtest
  - `POST /api/v1/ml/experiment` — Train ML pipeline model
  - `POST /api/v1/ml/compare` — Compare ML algorithms
  - `POST /api/v1/data/download` — Download market data
  - `GET /api/v1/config` — View sanitized configuration

#### Request Guardrails:
- **Maximum symbols per request**: 50
- **Maximum date range**: 3,650 days (10 years)
- Requests exceeding guardrails or invalid input syntax return `422 Unprocessable Content`.

---

## Quality Assurance & Testing

Run the full verification suite to confirm codebase health:

```powershell
# 1. Run Unit & Security Test Suite (190+ tests)
uv run pytest

# 2. Run Ruff Linter
uv run ruff check .

# 3. Run Mypy Type Checker
uv run mypy src/quant
```

---

## Security Model & Audit Compliance

- **Authentication**: Constant-time API key verification (`hmac.compare_digest`) on REST endpoints.
- **Input Sanitization**: Path traversal defense in `ParquetProvider` using symbol character validation (`^[A-Za-z0-9_.-]+$`) and path relative resolution checks.
- **Safe Parsing**: YAML configs are loaded strictly using `yaml.safe_load`.
- **Ignore Rules**: Sensitive files (`.env`, `*.sqlite`, `*.duckdb`, `*.db`) are strictly excluded via `.gitignore`.

---

## License

Proprietary – Internal use only.