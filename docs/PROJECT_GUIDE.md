# Trustos Quantitative Finance Platform: Comprehensive Technical & Architectural Guide

## 1. Executive Summary & Core Philosophy

Trustos is an institutional-grade quantitative research, backtesting, machine learning, and paper-trading platform developed in Python 3.12. It is engineered to bridge the gap between mathematical strategy research and production execution while eliminating the common pitfalls of algorithmic trading: look-ahead bias, unrealistically low transaction cost assumptions, data leakage during model training, and unhandled operational risk.

### Core Tenets
1. **Look-Ahead Bias Immunity**: Market data, feature transforms, and model training enforce strict time-boundary isolation. Information is never accessed before its simulated availability timestamp.
2. **Realistic Execution & Cost Modeling**: Every backtest and simulation incorporates commission, bid-ask spread, linear slippage, and market impact models.
3. **Mathematical Rigor**: Risk management, volatility estimation, and portfolio optimization utilize robust statistical and financial formulations (Garman-Klass, Parkinson, Rogers-Satchell, Cornish-Fisher VaR, Risk Parity, and Purged Time-Series CV).
4. **Resilient Production Architecture**: Live-like paper trading features SQLite Write-Ahead Logging (WAL) state persistence, pre-trade risk tripwires, structured JSON logging, Prometheus metrics, and a hardened FastAPI REST API.

---

## 2. End-to-End System Architecture

The following diagram illustrates the flow of data, computation, and execution across the Trustos subsystems:

```
+-----------------------------------------------------------------------------------+
|                               1. DATA LAYER                                       |
|  - MockDataProvider / ParquetDataProvider / DuckDB / YFinance                     |
|  - Schema Validation (monotonic timestamps, gap detection, OHLC consistency)      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           2. FEATURE ENGINEERING                                  |
|  - Technical (EMA, RSI, MACD, Bollinger, ATR)                                     |
|  - Statistical (Rolling z-score, moments, skewness, kurtosis)                     |
|  - Microstructure & Volatility (Garman-Klass, Parkinson, Rogers-Satchell)         |
+--------------------+------------------------------------+-------------------------+
                     |                                    |
                     v                                    v
+--------------------+--------------+   +----------------+--------------------------+
|      3. STRATEGY ENGINE           |   |       4. MACHINE LEARNING ENGINE          |
|  - BaseSignalStrategy             |   |  - Purged & Embargoed TimeSeriesCV        |
|  - Momentum / Mean Reversion      |   |  - BaseMLModel Estimators                 |
|  - Breakout / MACD Trend          |   |  - OnlineEnsemble (SGD & Passive-Aggr.)   |
|  - Dual Momentum / Pairs Trading  |   |  - Feature Selection & Scaling            |
+--------------------+--------------+   +----------------+--------------------------+
                     \                                    /
                      \                                  /
                       v                                v
+-----------------------------------------------------------------------------------+
|                        5. PORTFOLIO & RISK ALLOCATION                             |
|  - Portfolio Construction (Equal Weight, Inv-Vol, Risk Parity, Max Sharpe, Min-Var)|
|  - Risk Engine (Parametric, Historical, Cornish-Fisher, Monte Carlo VaR / CVaR)   |
|  - Exposure Constraints & Drawdown Tripwires                                      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                  6. EXECUTION & SIMULATION SUBSYSTEMS                             |
|  +-------------------------------------+   +------------------------------------+ |
|  |     Event-Driven BacktestEngine     |   |          PaperEngine (Live)        | |
|  | - Slippage, Spread, Commission      |   | - Async Bar Feed Ingestion         | |
|  | - Order/Fill Lifecycle Tracking     |   | - SQLite WAL State Persistence     | |
|  | - Vectorized & Event Simulation     |   | - Pre-trade Risk Limit Enforcement | |
|  +-------------------------------------+   +------------------------------------+ |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                   7. RESEARCH REPORTING & PRODUCTION API                          |
|  - Research Reports: Interactive HTML/Plotly, Drawdown Waterfalls, Markdown/JSON  |
|  - FastAPI Server: Key Authentication, Security Headers, Rate Limiting            |
|  - APScheduler: Cron/Interval Engine & Automated Health/Drift Monitoring          |
|  - Prometheus Collector & Multi-Channel Alerting (Log, Webhook, Email)            |
+-----------------------------------------------------------------------------------+
```

---

## 3. Subsystem Breakdown

### 3.1 Data Layer (`quant.data`)
The data layer is responsible for ingesting, validating, and standardizing financial time-series data.

- **Data Providers**:
  - `MockDataProvider`: Generates synthetic multi-asset geometric Brownian motion price paths with realistic volatility and volume clustering for unit testing and deterministic simulation.
  - `ParquetDataProvider`: High-performance columnar storage reader utilizing PyArrow and DuckDB for fast querying of historical tick/bar data.
  - `YFinanceDataProvider`: Live and historical downloader for global equities and indices.
- **Validation & Cleaning (`quant.data.validation`)**:
  - Validates strict monotonic index ordering.
  - Enforces OHLC physical consistency: `High >= max(Open, Close)` and `Low <= min(Open, Close)`.
  - Detects data gaps, stale bars, zero volume anomalies, and spurious outliers.

---

### 3.2 Feature Engineering (`quant.features`)
The feature layer computes quantitative factors across four major domains:

1. **Technical Indicators (`quant.features.technical`)**:
   - Trend: Exponential Moving Average (EMA), Simple Moving Average (SMA), Double EMA.
   - Momentum: Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Stochastic Oscillator.
   - Volatility Bands: Bollinger Bands (rolling mean +/- k * rolling std), Average True Range (ATR).
2. **Statistical Features (`quant.features.statistical`)**:
   - Rolling Z-scores: $(P_t - \mu_{t, w}) / \sigma_{t, w}$.
   - Rolling Skewness and Kurtosis for tail-risk estimation.
   - Rolling autocorrelation and variance ratios.
3. **Volatility Estimators (`quant.features.volatility`)**:
   - Implements advanced extreme-value volatility estimators that extract significantly more information per bar than standard close-to-close returns.
4. **Market Microstructure (`quant.features.microstructure`)**:
   - Roll measure for effective spread.
   - Amihud illiquidity ratio: $|R_t| / \text{Volume}_t$.
   - Kyle's lambda price impact parameter.

---

### 3.3 Alpha Strategies (`quant.strategies`)
All strategies inherit from `BaseSignalStrategy` and adhere to a unified interface: `generate_signals(data: pd.DataFrame, current_time: datetime) -> dict[str, float]`.

- **`BuyAndHoldStrategy`**: Passive benchmark strategy allocating target weights to designated symbols.
- **`MomentumStrategy`**: Cross-sectional and time-series momentum ranking assets based on lookback return percentiles.
- **`MeanReversionStrategy`**: Pairs Bollinger Band bounds with RSI overbought/oversold levels (e.g., RSI < 30 and price < lower band triggers long).
- **`BreakoutStrategy`**: Donchian channel breakout strategy with ATR volatility filters and trailing channel exits.
- **`MACDStrategy`**: Signal line crossover strategy filtered by long-term trend directional confirmation.
- **`DualMomentumStrategy`**: Gary Antonacci dual momentum combining relative strength across assets with absolute momentum (cash filter when trend is negative).
- **`PairsTradingStrategy`**: Statistical arbitrage strategy computing rolling spread z-scores between cointegrated asset pairs.

---

### 3.4 Portfolio Construction & Optimization (`quant.portfolio`)
Converts raw strategy signals into portfolio allocation weights under strict capital constraints:

- **Equal Weight**: Allocates $w_i = 1 / N$ across active signals.
- **Inverse Volatility**: Allocates inversely proportional to recent rolling volatility: $w_i \propto 1 / \sigma_i$.
- **Volatility Targeting**: Scales aggregate portfolio exposure up or down to maintain a constant annualized portfolio volatility target:
  $$\text{Leverage} = \frac{\sigma_{\text{target}}}{\sigma_{\text{realized}}}$$
- **Risk Parity**: Optimizes portfolio weights such that each asset contributes an equal amount of marginal risk to total portfolio variance:
  $$\text{RC}_i = w_i \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}} = \frac{1}{N} \sqrt{w^T \Sigma w}$$
- **Mean-Variance (Markowitz)**: Maximizes expected return for a target variance level under covariance matrix $\Sigma$.
- **Maximum Sharpe Ratio**: Solves for weights maximizing $(w^T \mu - r_f) / \sqrt{w^T \Sigma w}$.
- **Constraint Enforcement (`PortfolioConstraints`)**:
  - `max_position`: Maximum allowable weight in a single asset.
  - `max_gross_exposure`: Total absolute leverage constraint $\sum |w_i| \le L_{\max}$.
  - `max_net_exposure`: Directional market exposure constraint $|\sum w_i| \le N_{\max}$.

---

### 3.5 Risk Management Engine (`quant.risk`)
The risk engine continuously calculates downside risk, tail-risk metrics, and enforces circuit breakers:

- **Value-at-Risk (VaR)**:
  - **Parametric (Gaussian)**: $\text{VaR}_\alpha = -(\mu - z_\alpha \sigma)$.
  - **Historical Simulation**: Empirical quantile of historical return distributions.
  - **Cornish-Fisher Expansion**: Adjusted for empirical skewness ($S$) and excess kurtosis ($K$):
    $$z_{CF} = z_\alpha + \frac{S}{6}(z_\alpha^2 - 1) + \frac{K}{24}(z_\alpha^3 - 3z_\alpha) - \frac{S^2}{36}(2z_\alpha^3 - 5z_\alpha)$$
  - **Monte Carlo Simulation**: Simulates geometric price trajectories under multivariate Student-t or Normal copulas.
- **Conditional Value-at-Risk (CVaR / Expected Shortfall)**: Expected loss given that loss exceeds the VaR threshold:
  $$\text{CVaR}_\alpha = -E[R \mid R \le -\text{VaR}_\alpha]$$
- **Drawdown Circuit Breakers**:
  - Monitors high-water equity peaks: $\text{DD}_t = (E_t - \text{Peak}_t) / \text{Peak}_t$.
  - When drawdown crosses critical thresholds (e.g., 15%), the engine automatically reduces gross exposure by 50% or closes positions.

---

### 3.6 Backtesting Engine (`quant.backtest`)
Provides an event-driven backtesting environment with realistic market simulation:

- **Order Management System (OMS)**: Supports Market, Limit, and Stop orders with unique order IDs, timestamps, quantities, and execution status tracking.
- **Execution Cost Simulation (`ExecutionConfig`)**:
  - **Commissions**: Fixed per-trade bps fee.
  - **Bid-Ask Spread**: Half-spread penalty on market orders.
  - **Linear Slippage**: Slippage penalty proportional to trade volume.
  - **Market Impact**: Non-linear price displacement:
    $$\Delta P = \gamma \cdot \text{Spread} \cdot \sqrt{\frac{\text{Order Quantity}}{\text{Bar Volume}}}$$
- **Accounting & Mark-to-Market**: Tracks cash balance, realized PnL, unrealized PnL, margin requirements, and equity curves bar-by-bar.

---

### 3.7 Machine Learning Subsystem (`quant.ml`)
Provides machine-learning capabilities designed specifically for financial time-series:

- **Cross-Validation (`TimeSeriesCV`, `PurgedCV`, `EmbargoedCV`)**:
  - Prevents data leakage by ensuring training splits strictly precede test splits in time.
  - **Purging**: Removes training samples whose label evaluation window overlaps with the test set.
  - **Embargoing**: Inserts a buffer period after the test set to eliminate serial correlation leakage.
- **Model Wrapper Architecture (`BaseMLModel`)**:
  - Unified estimator dispatch for linear models (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`), tree ensembles (`RandomForest`, `GradientBoosting`, `ExtraTrees`, `HistGradientBoosting`), kernel methods (`SVR`, `SVC`), neural networks (`MLP`), and gradient boosted trees (`XGBoost`, `LightGBM`, `CatBoost`).
  - Integrated feature scaling (`StandardScaler`) and importance extraction.
- **Online Learning (`OnlineEnsemble`)**:
  - Incremental SGD-based regressors and classifiers for real-time model updating on streaming data.
  - Adaptive ensemble weighting using exponential decay of historical validation loss.

---

### 3.8 Research & Walk-Forward Optimization (`quant.research`)
- **`ExperimentTracker`**: Persists experiment parameters, feature configs, backtest results, and metrics in structured JSON/Parquet storage.
- **`WalkForwardAnalysis`**:
  - Employs rolling or expanding window out-of-sample validation to detect strategy overfitting.
  - Evaluates parameter stability across market regimes.
  - Applies multiple testing statistical adjustments (Bonferroni / Holm-Bonferroni correction).
- **Automated Research Reports (`ResearchReport`)**:
  - Generates HTML reports with interactive Plotly equity curves, monthly returns heatmaps, drawdown waterfalls, and comprehensive performance tables.
  - Generates Markdown and JSON summaries for automated CI/CD pipelines.

---

### 3.9 Paper Trading Engine (`quant.paper`)
Enables live-simulated trading without risking actual capital:

- **Asynchronous Execution Loop**: Ingests real-time or simulated bar feeds via async generator loops.
- **State Persistence (`PaperState`)**:
  - SQLite backend using Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and normal synchronous mode.
  - Persists positions, orders, fills, cash balances, and equity snapshots.
  - Provides instant recovery across process restarts.
- **Pre-Trade Risk Checks**:
  - Verifies position size limits, gross exposure limits, and net exposure limits before submitting orders.

---

### 3.10 Production Infrastructure & REST API (`quant.production`)
- **FastAPI Application (`quant.production.api`)**:
  - Endpoints for executing backtests, running ML experiments, checking strategy status, and managing paper trading.
  - Security hardening: API Key authentication (`X-API-Key`), TrustedHost middleware, CORS configuration, rate limiting, and security response headers.
  - Automatic masking of sensitive secrets and API tokens in all logs and outputs.
- **Monitoring & Metrics (`quant.production.monitoring`)**:
  - Centralized Prometheus metrics collector tracking request latency, backtest run counts, error rates, and risk breach events.
  - Structured JSON logging via `structlog`.
- **Job Scheduler (`quant.production.scheduler`)**:
  - `APScheduler` wrapper supporting cron expressions and interval triggers for automated data downloads, model retraining, and health checks.
  - `add_job` with typed triggers and monitored execution envelopes.
- **Alerting Framework (`quant.production.alerts`)**:
  - Multi-channel notification dispatch (Console/Log, Webhook, Email).
  - Configurable alert rules with operator dispatch tables for latency thresholds, error spikes, and drawdown breaches.

---

## 4. Key Mathematical Formulations

### 4.1 Volatility Estimators

1. **Close-to-Close Volatility**:
   $$\sigma_{CC} = \sqrt{\frac{252}{N - 1} \sum_{i=1}^N (r_i - \bar{r})^2}$$

2. **Parkinson High-Low Volatility**:
   $$\sigma_P = \sqrt{\frac{252}{N} \sum_{i=1}^N \frac{\ln(H_i / L_i)^2}{4 \ln 2}}$$

3. **Garman-Klass Volatility** (incorporates Open, High, Low, Close):
   $$\sigma_{GK} = \sqrt{\frac{252}{N} \sum_{i=1}^N \left[ 0.5 \left( \ln \frac{H_i}{L_i} \right)^2 - (2\ln 2 - 1) \left( \ln \frac{C_i}{O_i} \right)^2 \right]}$$

4. **Rogers-Satchell Volatility** (handles non-zero drift):
   $$\sigma_{RS} = \sqrt{\frac{252}{N} \sum_{i=1}^N \left[ \ln \left( \frac{H_i}{C_i} \right) \ln \left( \frac{H_i}{O_i} \right) + \ln \left( \frac{L_i}{C_i} \right) \ln \left( \frac{L_i}{O_i} \right) \right]}$$

---

### 4.2 Performance & Risk Ratios

1. **Annualized Sharpe Ratio**:
   $$\text{Sharpe} = \frac{\bar{R} - R_f}{\sigma_R} \times \sqrt{252}$$

2. **Sortino Ratio** (downside risk only):
   $$\text{Sortino} = \frac{\bar{R} - \text{Target}}{\sigma_{\text{downside}}} \times \sqrt{252}, \quad \sigma_{\text{downside}} = \sqrt{\frac{1}{N} \sum_{R_i < \text{Target}} (R_i - \text{Target})^2}$$

3. **Calmar Ratio**:
   $$\text{Calmar} = \frac{\text{CAGR}}{|\text{Max Drawdown}|}$$

4. **Compound Annual Growth Rate (CAGR)**:
   $$\text{CAGR} = \left( \frac{E_{\text{end}}}{E_{\text{start}}} \right)^{\frac{252}{N}} - 1$$

---

## 5. Developer & Quant Workflows

### 5.1 Implementing a Custom Strategy
To create a new quantitative strategy:
1. Subclass `BaseSignalStrategy` from `quant.strategies.signals`.
2. Implement `generate_signals(self, data: pd.DataFrame, current_time: datetime) -> dict[str, float]`.
3. Register the strategy in `STRATEGY_REGISTRY` in `quant.strategies.signals`.

```python
from quant.strategies.signals import BaseSignalStrategy, STRATEGY_REGISTRY
import pandas as pd
from datetime import datetime

class CustomRSIStrategy(BaseSignalStrategy):
    def __init__(self, period: int = 14, lower: float = 30, upper: float = 70):
        super().__init__()
        self.period = period
        self.lower = lower
        self.upper = upper

    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> dict[str, float]:
        signals = {}
        symbols = self._extract_symbols(data)
        for sym in symbols:
            close = self._get_close_series(data, sym)
            if len(close) < self.period:
                continue
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(self.period).mean()
            loss = (-delta.clip(upper=0)).rolling(self.period).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi = 100 - (100 / (1 + rs))
            
            latest_rsi = rsi.iloc[-1]
            if latest_rsi < self.lower:
                signals[sym] = 1.0  # Long
            elif latest_rsi > self.upper:
                signals[sym] = 0.0  # Flat
        return signals

STRATEGY_REGISTRY["custom_rsi"] = CustomRSIStrategy
```

### 5.2 Running a Full Backtest
```bash
python scripts/run_backtest.py --strategy momentum --symbols SPY QQQ IWM --start 2020-01-01 --end 2024-01-01
```

### 5.3 Executing Walk-Forward Validation & Report Generation
```bash
python scripts/run_research.py --strategy breakout --report html
```

### 5.4 Starting the Paper Trading Engine
```bash
python -m quant.paper.engine configs/paper.yaml
```

### 5.5 Launching the Production REST API
```bash
uvicorn quant.production.api:app --host 0.0.0.0 --port 8000
```

---

## 6. Testing & Quality Assurance
- **Unit & Integration Suite**: 248 comprehensive test scenarios across `tests/unit` and `tests/integration`.
- **Static Analysis & Type Safety**:
  - `ruff check .`: Strict linting and formatting enforcement.
  - `mypy src/quant`: Static type verification across 46 source modules.
- **Emoji Compliance**: Verified zero emojis across codebase, docstrings, and artifacts.
