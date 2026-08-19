# Paper Trading System

This document describes the paper trading system for the Quant platform.

## Overview

The paper trading system simulates live trading using real-time or near-real-time market data, with realistic execution costs and risk management. It bridges the gap between backtesting and live trading.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Feed      │────▶│  Signal Generator │────▶│  Paper Engine    │
│  (yfinance/     │     │  (Strategies)     │     │  (ExecutionSim)  │
│   polygon/etc)  │     │                   │     │                  │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  State Store     │
                                               │  (SQLite)        │
                                               │  - Positions     │
                                               │  - Cash          │
                                               │  - Orders        │
                                               │  - Fills         │
                                               └──────────────────┘
```

## Components

### 1. Paper Engine (`src/quant/paper/engine.py`)
- Extends `BacktestEngine` for live-like operation
- Uses `ExecutionSimulator` for realistic fills
- Processes signals bar-by-bar in real-time

### 2. State Persistence (`src/quant/paper/state.py`)
- SQLite database with WAL mode
- Tables: `positions`, `cash`, `orders`, `fills`, `account_snapshots`
- ACID transactions for atomic updates
- Checkpointing every N bars

### 3. Data Feed (`src/quant/paper/data_feed.py`)
- Abstract base class `DataFeed`
- Implementations: `YFinanceFeed`, `PolygonFeed`, `MockFeed`
- Handles: reconnection, rate limiting, data validation

### 4. Risk Management (`src/quant/paper/risk.py`)
- Real-time position limits
- Max drawdown monitoring
- Daily loss limits
- Pre-trade checks

### 4. Scheduler Integration (`src/quant/production/scheduler.py`)
- Uses existing `JobScheduler` for scheduled runs
- Configurable cron triggers (e.g., daily at market close)
- Health checks every 5 minutes

## Configuration

`configs/paper.yaml`:

```yaml
paper_trading:
  enabled: true
  broker: "simulated"  # or "alpaca"
  account_id: "paper_account_001"
  initial_capital: 100000
  base_currency: "USD"
  
  market_hours:
    open: "09:30"
    close: "16:00"
    timezone: "America/New_York"
  
  execution:
    commission_bps: 2.0
    spread_bps: 1.0
    slippage_bps: 2.0
    fill_probability: 1.0
    partial_fill_prob: 0.0
    
  risk:
    max_position_pct: 0.10
    max_gross_exposure: 1.0
    max_net_exposure: 0.5
    max_daily_loss: 0.05
    max_drawdown: 0.15
    position_limit_check: "pre_trade"
    
  data:
    provider: "yfinance"  # or "mock", "polygon"
    symbols: ["SPY", "QQQ", "IWM"]
    timeframe: "1m"
    
  monitoring:
    log_trades: true
    log_positions: true
    log_pnl: true
    alert_on_risk_breach: true
    
  reporting:
    daily_summary: true
    eod_positions: true
    trade_log_path: "reports/paper/trades.parquet"
    position_log_path: "reports/paper/positions.parquet"
    pnl_log_path: "reports/paper/pnl.parquet"
    
  state:
    db_path: "data/paper/state.db"
    checkpoint_interval: 100  # bars
```

## Running Paper Trading

```bash
# Interactive mode
uv run python scripts/run_paper.py --config configs/paper.yaml

# Daemon mode (via scheduler)
uv run python -c "
from quant.production.scheduler import get_scheduler, setup_default_jobs
from quant.paper.engine import PaperEngine
from quant.production.config import get_config

config = get_config()
engine = PaperEngine.from_config(config.paper_trading)
scheduler = get_scheduler()

# Add paper trading job
scheduler.add_job(
    engine.run_daily_cycle,
    'cron',
    'daily_paper_trading',
    hour=16, minute=5,  # 5 min after market close
)

scheduler.start()
"
```

## Running the Paper Trading Script

```bash
# Quick start
uv run python scripts/run_paper.py --config configs/paper.yaml

# With overrides
uv run python scripts/run_paper.py --config configs/paper.yaml --symbols SPY QQQ --provider yfinance
```

## State Persistence

### SQLite Schema

```sql
-- Positions table
CREATE TABLE positions (
    symbol TEXT PRIMARY KEY,
    quantity REAL NOT NULL,
    avg_price REAL NOT NULL,
    market_value REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cash table
CREATE TABLE cash (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    amount REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL,
    status TEXT NOT NULL,
    filled_quantity REAL DEFAULT 0,
    avg_fill_price REAL DEFAULT 0,
    commission REAL DEFAULT 0,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fills table
CREATE TABLE fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- Account snapshots
CREATE TABLE account_snapshots (
    timestamp TIMESTAMP PRIMARY KEY,
    cash REAL NOT NULL,
    total_value REAL NOT NULL,
    gross_exposure REAL NOT NULL,
    net_exposure REAL NOT NULL
);
```

## Risk Management

### Pre-Trade Checks
1. **Position limit**: `position_pct <= max_position_pct`
2. **Gross exposure**: `sum(abs(position_value)) <= max_gross_exposure * equity`
3. **Net exposure**: `sum(position_value) <= max_net_exposure * equity`
4. **Daily loss**: `daily_pnl >= -max_daily_loss * equity`
5. **Max drawdown**: `current_drawdown <= max_drawdown`

### On Breach
- Log alert
- Cancel pending orders for symbol
- Reduce position by 50% (configurable)
- Alert via monitoring system

## Reporting

### Daily Summary (auto-generated at market close)
- P&L summary (realized + unrealized)
- Position snapshot
- Trade log
- Risk metrics (VaR, Sharpe, max DD)

### EOD Files (Parquet)
- `trades.parquet`: All fills with timestamps
- `positions.parquet`: End-of-day positions
- `pnl.parquet`: Minute-by-minute P&L

## Monitoring & Alerting

### Key Metrics (Prometheus)
- `quant_paper_cash` - Current cash
- `quant_paper_equity` - Total equity
- `quant_paper_positions` - Number of open positions
- `quant_paper_daily_pnl` - Realized + unrealized P&L
- `quant_paper_max_drawdown` - Current drawdown
- `quant_paper_rejection_rate` - Order rejection rate
- `quant_paper_data_latency_ms` - Data feed latency

### Alerts
| Alert | Condition | Severity |
|-------|-----------|----------|
| `DrawdownBreach` | `drawdown > max_drawdown` | CRITICAL |
| `DailyLossBreach` | `daily_pnl < -max_daily_loss` | CRITICAL |
| `StaleData` | `last_price_update > 2 * interval` | WARNING |
| `OrderRejectionSpike` | `rejection_rate_5min > 10%` | WARNING |
| `PositionLimitBreach` | `position_pct > max_position_pct` | WARNING |

## Testing

```bash
# Unit tests
uv run pytest tests/unit/paper/ -v

# Integration test (2-week simulation)
uv run pytest tests/integration/test_paper_trading.py -v

# Stress test (high frequency)
uv run pytest tests/stress/test_paper_high_freq.py -v
```

## Extending for Live Broker

The paper engine uses an abstract `ExecutionBackend` interface:

```python
class ExecutionBackend(ABC):
    @abstractmethod
    async def submit_order(self, order: Order) -> Fill:
        ...
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        ...
    
    @abstractmethod
    async def get_positions(self) -> dict[str, Position]:
        ...
    
    @abstractmethod
    async def get_account(self) -> Account:
        ...
```

Implement `AlpacaBackend`, `IBKRBackend`, etc. by subclassing.

---

*Last updated: 2026-08-18*