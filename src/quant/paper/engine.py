"""Paper trading engine - extends backtest engine for live-like operation."""
from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.backtest.execution import ExecutionConfig
from quant.backtest.types import Fill, Order, OrderSide, OrderType, Position
from quant.portfolio.construction import PortfolioConstraints
from quant.production.monitoring import get_logger, get_metrics_collector


@dataclass
class PaperConfig:
    """Paper trading configuration."""
    enabled: bool = False
    broker: str = "simulated"  # "simulated", "alpaca"
    account_id: str = "paper_account_001"
    initial_capital: float = 100_000
    base_currency: str = "USD"
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "America/New_York"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data_provider: str = "mock"
    symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    timeframe: str = "1m"
    max_position_pct: float = 0.10
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.5
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.15
    position_limit_check: str = "pre_trade"
    state_db_path: str = "data/paper/state.db"
    checkpoint_interval: int = 100
    log_trades: bool = True
    log_positions: bool = True
    log_pnl: bool = True
    alert_on_risk_breach: bool = True


class PaperState:
    """Persistent state for paper trading using SQLite."""

    def __init__(self, db_path: str, initial_capital: float = 100_000):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self._init_db()

    @contextmanager
    def _connect(self):
        """Context manager for SQLite connections ensuring clean closure."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize SQLite database with WAL mode."""
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    market_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Cash table (single row)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cash (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    amount REAL NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("INSERT OR IGNORE INTO cash (id, amount) VALUES (1, ?)", (self.initial_capital,))

            # Orders table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
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
                )
            """)

            # Fills table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id)
                )
            """)

            # Account snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    timestamp TIMESTAMP PRIMARY KEY,
                    cash REAL NOT NULL,
                    total_value REAL NOT NULL,
                    gross_exposure REAL NOT NULL,
                    net_exposure REAL NOT NULL
                )
            """)

    def get_cash(self) -> float:
        """Get current cash balance."""
        with self._connect() as conn:
            row = conn.execute("SELECT amount FROM cash WHERE id = 1").fetchone()
            return row[0] if row else 0.0

    def set_cash(self, amount: float) -> None:
        """Set cash balance."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE cash SET amount = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                (amount,)
            )

    def get_position(self, symbol: str) -> dict | None:
        """Get position for a symbol."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT symbol, quantity, avg_price, market_value, unrealized_pnl, realized_pnl FROM positions WHERE symbol = ?",
                (symbol,)
            ).fetchone()
            if row:
                return {
                    "symbol": row[0],
                    "quantity": row[1],
                    "avg_price": row[2],
                    "market_value": row[3],
                    "unrealized_pnl": row[4],
                    "realized_pnl": row[5],
                }
            return None

    def set_position(self, symbol: str, quantity: float, avg_price: float,
                     market_value: float, unrealized_pnl: float, realized_pnl: float) -> None:
        """Set or update position."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions
                (symbol, quantity, avg_price, market_value, unrealized_pnl, realized_pnl, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (symbol, quantity, avg_price, market_value, unrealized_pnl, realized_pnl))

    def delete_position(self, symbol: str) -> None:
        """Delete position (when flat)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

    def get_all_positions(self) -> dict[str, dict]:
        """Get all positions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, quantity, avg_price, market_value, unrealized_pnl, realized_pnl FROM positions"
            ).fetchall()
            return {
                row[0]: {
                    "symbol": row[0],
                    "quantity": row[1],
                    "avg_price": row[2],
                    "market_value": row[3],
                    "unrealized_pnl": row[4],
                    "realized_pnl": row[5],
                }
                for row in rows
            }

    def save_order(self, order: Order) -> None:
        """Save order to database."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders
                (order_id, symbol, side, quantity, order_type, limit_price, status,
                 filled_quantity, avg_fill_price, commission, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id, order.symbol, order.side.value, order.quantity,
                order.order_type.value, order.limit_price, order.status.value,
                order.filled_quantity, order.avg_fill_price, order.commission,
                order.timestamp.isoformat() if order.timestamp else None
            ))

    def save_fill(self, fill: Fill) -> None:
        """Save fill to database."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO fills
                (fill_id, order_id, symbol, side, quantity, price, commission, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fill.fill_id, fill.order_id, fill.symbol, fill.side.value,
                fill.quantity, fill.price, fill.commission,
                fill.timestamp.isoformat() if fill.timestamp else None
            ))

    def save_snapshot(self, timestamp: datetime, cash: float, total_value: float,
                      gross_exposure: float, net_exposure: float) -> None:
        """Save account snapshot."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO account_snapshots
                (timestamp, cash, total_value, gross_exposure, net_exposure)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp.isoformat(), cash, total_value, gross_exposure, net_exposure))

    def load_portfolio_state(self, portfolio) -> None:
        """Load portfolio state from database."""
        # Load cash
        portfolio.cash = self.get_cash()

        # Load positions
        positions = self.get_all_positions()
        for symbol, pos in positions.items():
            if pos["quantity"] != 0:
                portfolio.positions[symbol] = Position(
                    symbol=pos["symbol"],
                    quantity=pos["quantity"],
                    avg_price=pos["avg_price"],
                    market_value=pos["market_value"],
                    unrealized_pnl=pos["unrealized_pnl"],
                    realized_pnl=pos["realized_pnl"],
                )

    def get_account_history(self, limit: int = 100) -> list[dict]:
        """Get recent account snapshots."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, cash, total_value, gross_exposure, net_exposure "
                "FROM account_snapshots ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [
                {
                    "timestamp": row[0],
                    "cash": row[1],
                    "total_value": row[2],
                    "gross_exposure": row[3],
                    "net_exposure": row[4],
                }
                for row in rows
            ]

    def get_snapshots(self, limit: int = 100) -> list[dict]:
        """Get recent account snapshots (alias for get_account_history)."""
        return self.get_account_history(limit)


class PaperDataFeed:
    """Abstract base class for paper trading data feeds."""

    def __init__(self, symbols: list[str], timeframe: str = "1m"):
        self.symbols = symbols
        self.timeframe = timeframe

    async def get_latest_bar(self, symbol: str) -> dict | None:
        """Get latest bar for a symbol. Returns dict with OHLCV."""
        raise NotImplementedError

    async def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical data for warm-up."""
        raise NotImplementedError

    async def subscribe(self, callback: Callable[[str, dict], Any]) -> None:
        """Subscribe to real-time bars."""
        raise NotImplementedError

    async def unsubscribe(self, callback: Callable[[str, dict], Any]) -> None:
        """Unsubscribe from real-time bars."""
        raise NotImplementedError


class MockDataFeed(PaperDataFeed):
    """Mock data feed for testing."""

    def __init__(self, symbols: list[str], timeframe: str = "1m", seed: int = 42):
        super().__init__(symbols, timeframe)
        self.seed = seed
        self._current_prices = dict.fromkeys(symbols, 100.0)
        self._rng = __import__('numpy').random.default_rng(seed)

    async def get_latest_bar(self, symbol: str) -> dict | None:
        """Generate mock bar."""
        if symbol not in self._current_prices:
            return None

        price = self._current_prices[symbol]
        # Simple random walk
        change = self._rng.normal(0, 0.001)
        price *= (1 + change)
        self._current_prices[symbol] = price

        _ = price * 0.0001  # spread (unused)
        return {
            "open": price * (1 + self._rng.normal(0, 0.0005)),
            "high": price * (1 + abs(self._rng.normal(0, 0.002))),
            "low": price * (1 - abs(self._rng.normal(0, 0.002))),
            "close": price,
            "volume": int(self._rng.lognormal(13, 0.5)),
            "timestamp": datetime.now(),
        }

    async def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Generate historical mock data."""
        dates = pd.bdate_range(start=start, end=end, freq="B")
        n = len(dates)
        np_rng = __import__('numpy').random.default_rng(self.seed)

        returns = np_rng.normal(0.0001, 0.015, n)
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "open": prices * (1 + np_rng.normal(0, 0.0005, n)),
            "high": prices * (1 + abs(np_rng.normal(0, 0.005, n))),
            "low": prices * (1 - abs(np_rng.normal(0, 0.005, n))),
            "close": prices,
            "volume": np_rng.lognormal(13, 0.5, n).astype(int),
        }, index=pd.DatetimeIndex(dates, name="timestamp"))

        return df


class PaperEngine(BacktestEngine):
    """Paper trading engine - extends backtest engine for live operation."""

    def __init__(self, config: PaperConfig):
        # Create base backtest config
        bt_config = BacktestConfig(
            initial_capital=config.initial_capital,
            execution=config.execution,
            portfolio_constraints=PortfolioConstraints(
                max_position=config.max_position_pct,
                max_gross_exposure=config.max_gross_exposure,
                max_net_exposure=config.max_net_exposure,
                long_only=True,
            ),
            max_drawdown=config.max_drawdown,
            max_drawdown_action="reduce_exposure",
        )

        super().__init__(bt_config)

        self.paper_config = config
        self.logger = get_logger("paper.engine")
        self.metrics = get_metrics_collector()

        # State persistence
        self.state = PaperState(config.state_db_path, config.initial_capital)
        self.checkpoint_interval = config.checkpoint_interval
        self.bars_since_checkpoint = 0

        # Data feed
        self.data_feed: PaperDataFeed | None = None
        self._setup_data_feed()

        # Load persisted state
        self._load_state()

        # Risk monitoring
        self.daily_pnl = 0.0
        self.peak_equity = config.initial_capital
        self.max_drawdown_hit = False
        self.daily_start_equity = config.initial_capital

    def _setup_data_feed(self) -> None:
        """Initialize data feed based on config."""
        if self.paper_config.data_provider == "mock":
            self.data_feed = MockDataFeed(self.paper_config.symbols, self.paper_config.timeframe)
        elif self.paper_config.data_provider == "yfinance":
            # Would import and use YFinanceFeed here
            raise NotImplementedError("yfinance feed not yet implemented")
        else:
            self.data_feed = MockDataFeed(self.paper_config.symbols, self.paper_config.timeframe)

    def _load_state(self) -> None:
        """Load persisted state."""
        try:
            self.state.load_portfolio_state(self.portfolio)
            # Load peak equity from snapshots
            history = self.state.get_account_history(1)
            if history:
                self.peak_equity = max(self.peak_equity, history[0]["total_value"])
            self.logger.info("state_loaded", cash=self.portfolio.cash,
                           positions=len(self.portfolio.positions))
        except Exception as e:
            self.logger.warning("state_load_failed", error=str(e))

    def _checkpoint(self) -> None:
        """Save checkpoint."""
        try:
            # Save positions
            for symbol, pos in self.portfolio.positions.items():
                if pos.quantity != 0:
                    self.state.set_position(
                        symbol, pos.quantity, pos.avg_price, pos.market_value,
                        pos.unrealized_pnl, pos.realized_pnl
                    )
                else:
                    self.state.delete_position(symbol)

            # Save cash
            self.state.set_cash(self.portfolio.cash)

            # Save snapshot
            self.state.save_snapshot(
                datetime.now(), self.portfolio.cash, self.portfolio.total_value,
                self.portfolio.gross_exposure, self.portfolio.net_exposure
            )

            self.bars_since_checkpoint = 0
            self.logger.debug("checkpoint_saved")
        except Exception as e:
            self.logger.error("checkpoint_failed", error=str(e))

    def _check_risk_limits(self) -> bool:  # type: ignore[override]
        """Check and enforce risk limits."""
        current_equity = self.portfolio.total_value

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            # Recovered to a new peak -> re-arm the drawdown tripwire.
            self.max_drawdown_hit = False

        # Check drawdown
        if self.peak_equity > 0 and (drawdown := (self.peak_equity - current_equity) / self.peak_equity) > self.paper_config.max_drawdown and not self.max_drawdown_hit:
                    self.logger.warning("drawdown_breach", drawdown=drawdown,
                                      max_allowed=self.paper_config.max_drawdown)
                    self.metrics.record_risk_event("drawdown_breach")
                    if self.paper_config.alert_on_risk_breach:
                        from quant.production.alerts import Alert, AlertLevel, AlertManager
                        am = AlertManager()
                        alert = Alert(
                            id=f"drawdown_{int(time.time())}",
                            name="DrawdownBreach",
                            level=AlertLevel.CRITICAL,
                            component="risk",
                            message=f"Drawdown breach: {drawdown:.2%} > {self.paper_config.max_drawdown:.2%}",
                            value=drawdown,
                            threshold=self.paper_config.max_drawdown,
                        )
                        am.fire_alert("drawdown_breach", alert.message, drawdown, self.paper_config.max_drawdown)
                    self.max_drawdown_hit = True
                    return False

        # Daily loss limit
        daily_pnl = current_equity - self.daily_start_equity
        daily_loss_pct = -daily_pnl / self.daily_start_equity if self.daily_start_equity > 0 else 0
        if daily_loss_pct > self.paper_config.max_daily_loss:
            self.logger.warning("daily_loss_breach", loss_pct=daily_loss_pct,
                              limit=self.paper_config.max_daily_loss)
            self.metrics.record_risk_event("daily_loss_breach")
            return False

        return True

    def _reduce_exposure(self) -> None:
        """Reduce exposure by half for all positions when risk limits are breached."""
        self.logger.info("reducing_exposure")
        for symbol, pos in list(self.portfolio.positions.items()):
            if pos.quantity == 0:
                continue
            price = self.current_prices.get(symbol)
            if not price or price <= 0:
                continue
            reduce_qty = abs(pos.quantity) / 2.0
            if reduce_qty <= 0:
                continue
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            order = Order(
                symbol=symbol,
                side=side,
                quantity=reduce_qty,
                order_type=OrderType.MARKET,
                timestamp=self.current_time,
            )
            self._submit_order(order)

    def _check_pre_trade_risk(self, symbol: str, quantity: float, price: float, side: OrderSide) -> bool:
        """Pre-trade risk checks."""
        target_value = abs(quantity) * price
        current_equity = self.portfolio.total_value

        # Position size limit
        if self.paper_config.max_position_pct > 0:
            current_pos = self.portfolio.get_position(symbol)
            current_value = current_pos.market_value if current_pos else 0
            new_value = current_value + (target_value if side == OrderSide.BUY else -target_value)
            if abs(new_value) > self.paper_config.max_position_pct * self.portfolio.total_value:
                self.logger.warning("position_limit_exceeded", symbol=symbol,
                                  target_pct=abs(new_value)/current_equity)
                return False

        # Gross exposure
        gross_after = self.portfolio.gross_exposure + target_value
        if gross_after > self.paper_config.max_gross_exposure * self.portfolio.total_value:
            self.logger.warning("gross_exposure_limit_exceeded")
            return False

        # Net exposure
        net_change = target_value if side == OrderSide.BUY else -target_value
        net_after = self.portfolio.net_exposure + net_change
        if abs(net_after) > self.paper_config.max_net_exposure * self.portfolio.total_value:
            self.logger.warning("net_exposure_limit_exceeded")
            return False

        return True

    async def run_cycle(self) -> None:
        """Run one paper trading cycle."""
        self.logger.info("cycle_start")
        start_time = time.time()

        try:
            # Update daily start equity at midnight
            now = datetime.now()
            if now.hour == 0 and now.minute < 5:
                self.daily_start_equity = self.portfolio.total_value
                self.daily_pnl = 0.0

            # Get latest bars for all symbols
            bar_data = {}
            if self.data_feed is not None:
                for symbol in self.paper_config.symbols:
                    bar = await self.data_feed.get_latest_bar(symbol)
                    if bar:
                        bar_data[symbol] = bar

            if not bar_data:
                self.logger.warning("no_data_received")
                return

            # Update market data and portfolio
            self._update_market_data_from_bars(bar_data)
            self.portfolio.mark_to_market(self.current_prices)

            # Check risk limits
            if not self._check_risk_limits():
                self.logger.warning("risk_limits_exceeded_reducing_exposure")
                self._reduce_exposure()
                return

            # Generate signals
            bar_df = self._build_bar_dataframe(bar_data)
            if self.strategy:
                signals = self.strategy.generate_signals(bar_df, datetime.now())

                # Construct portfolio
                target_weights = self._construct_portfolio(signals)

                # Rebalance with pre-trade risk checks
                for symbol, target_weight in target_weights.items():
                    current_pos = self.portfolio.get_position(symbol)
                    current_value = current_pos.market_value if current_pos else 0
                    target_value = self.portfolio.total_value * target_weight
                    diff = target_value - current_value

                    if abs(diff) > self.portfolio.total_value * 0.001:  # 10 bps threshold
                        price = self.current_prices.get(symbol, 0)
                        if price > 0:
                            qty = diff / price
                            if qty != 0:
                                order_side = OrderSide.BUY if qty > 0 else OrderSide.SELL

                                # Pre-trade risk check
                                if self._check_pre_trade_risk(symbol, abs(qty),
                                                              self.current_prices.get(symbol, price),
                                                              order_side):
                                    order = Order(
                                        symbol=symbol,
                                        side=order_side,
                                        quantity=abs(qty),
                                        order_type=OrderType.MARKET,
                                        timestamp=datetime.now(),
                                    )
                                    self._submit_order(order)

            # Record state
            self._record_state()
            self.bars_since_checkpoint += 1

            # Checkpoint periodically
            if self.bars_since_checkpoint >= self.checkpoint_interval:
                self._checkpoint()

            duration = time.time() - start_time
            self.logger.info("cycle_complete", duration=duration)

        except Exception as e:
            self.logger.error("cycle_failed", error=str(e))
            self.metrics.record_error("paper_engine", type(e).__name__)
            raise

    def _update_market_data_from_bars(self, bar_data: dict[str, dict]) -> None:
        """Update market data from bar dictionary."""
        for symbol, bar in bar_data.items():
            if "close" in bar:
                self.current_prices[symbol] = bar["close"]
            if "bid" in bar:
                self.current_bid[symbol] = bar["bid"]
            if "ask" in bar:
                self.current_ask[symbol] = bar["ask"]

            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(self.current_prices[symbol])

    def _build_bar_dataframe(self, bar_data: dict[str, dict]) -> pd.DataFrame:
        """Build DataFrame from bar data for strategy."""
        dfs = []
        for symbol, bar in bar_data.items():
            df = pd.DataFrame({
                (symbol, "open"): [bar.get("open", bar.get("close", 0))],
                (symbol, "high"): [bar.get("high", bar.get("close", 0))],
                (symbol, "low"): [bar.get("low", bar.get("close", 0))],
                (symbol, "close"): [bar.get("close", 0)],
                (symbol, "volume"): [bar.get("volume", 0)],
            }, index=[datetime.now()])
            dfs.append(df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    @classmethod
    def from_config(cls, paper_config) -> PaperEngine:
        """Create PaperEngine from config object."""
        config = PaperConfig(
            enabled=paper_config.enabled,
            broker=paper_config.broker,
            account_id=paper_config.account_id,
            initial_capital=paper_config.initial_capital,
            base_currency=paper_config.base_currency,
            market_open=paper_config.market_open,
            market_close=paper_config.market_close,
            timezone=paper_config.timezone,
            execution=ExecutionConfig(
                commission_bps=paper_config.execution.commission_bps,
                spread_bps=paper_config.execution.spread_bps,
                slippage_bps=paper_config.execution.slippage_bps,
                fill_probability=paper_config.execution.fill_probability,
                partial_fill_prob=paper_config.execution.partial_fill_prob,
            ),
            data_provider=paper_config.data.provider,
            symbols=paper_config.data.symbols,
            timeframe=paper_config.data.timeframe,
            max_position_pct=paper_config.risk.max_position_pct,
            max_gross_exposure=paper_config.risk.max_gross_exposure,
            max_net_exposure=paper_config.risk.max_net_exposure,
            max_daily_loss=paper_config.risk.max_daily_loss,
            max_drawdown=paper_config.risk.max_drawdown,
            position_limit_check=paper_config.risk.position_limit_check,
            state_db_path=paper_config.state.db_path,
            checkpoint_interval=paper_config.state.checkpoint_interval,
            log_trades=paper_config.monitoring.log_trades,
            log_positions=paper_config.monitoring.log_positions,
            log_pnl=paper_config.monitoring.log_pnl,
            alert_on_risk_breach=paper_config.monitoring.alert_on_risk_breach,
        )
        return cls(config)


def create_paper_engine_from_yaml(config_path: str) -> PaperEngine:
    """Create PaperEngine from YAML config file."""
    from pathlib import Path

    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paper_config = cfg.get("paper_trading", {})

    config = PaperConfig(
        enabled=paper_config.get("enabled", False),
        broker=paper_config.get("broker", "simulated"),
        account_id=paper_config.get("account_id", "paper_account_001"),
        initial_capital=paper_config.get("initial_capital", 100_000),
        base_currency=paper_config.get("base_currency", "USD"),
        market_open=paper_config.get("market_open", "09:30"),
        market_close=paper_config.get("market_close", "16:00"),
        timezone=paper_config.get("timezone", "America/New_York"),
        execution=ExecutionConfig(
            commission_bps=paper_config.get("execution", {}).get("commission_bps", 2.0),
            spread_bps=paper_config.get("execution", {}).get("spread_bps", 1.0),
            slippage_bps=paper_config.get("execution", {}).get("slippage_bps", 2.0),
            fill_probability=paper_config.get("execution", {}).get("fill_probability", 1.0),
            partial_fill_prob=paper_config.get("execution", {}).get("partial_fill_prob", 0.0),
        ),
        data_provider=paper_config.get("data", {}).get("provider", "mock"),
        symbols=paper_config.get("data", {}).get("symbols", ["SPY", "QQQ", "IWM"]),
        timeframe=paper_config.get("data", {}).get("timeframe", "1m"),
        max_position_pct=paper_config.get("risk", {}).get("max_position_pct", 0.10),
        max_gross_exposure=paper_config.get("risk", {}).get("max_gross_exposure", 1.0),
        max_net_exposure=paper_config.get("risk", {}).get("max_net_exposure", 0.5),
        max_daily_loss=paper_config.get("risk", {}).get("max_daily_loss", 0.05),
        max_drawdown=paper_config.get("risk", {}).get("max_drawdown", 0.15),
        position_limit_check=paper_config.get("risk", {}).get("position_limit_check", "pre_trade"),
        state_db_path=paper_config.get("state", {}).get("db_path", "data/paper/state.db"),
        checkpoint_interval=paper_config.get("state", {}).get("checkpoint_interval", 100),
        log_trades=paper_config.get("monitoring", {}).get("log_trades", True),
        log_positions=paper_config.get("monitoring", {}).get("log_positions", True),
        log_pnl=paper_config.get("monitoring", {}).get("log_pnl", True),
        alert_on_risk_breach=paper_config.get("monitoring", {}).get("alert_on_risk_breach", True),
    )
    return PaperEngine(config)


def run_paper_trading(config_path: str = "configs/paper.yaml") -> None:
    """Run paper trading loop."""

    engine = create_paper_engine_from_yaml(config_path)

    if not engine.paper_config.enabled:
        engine.logger.warning("paper_trading_disabled")
        return

    async def run_loop():
        engine.logger.info("paper_trading_started")
        try:
            while True:
                await engine.run_cycle()
                # Sleep based on timeframe
                await asyncio.sleep(60)  # 1 minute for 1m timeframe
        except KeyboardInterrupt:
            engine.logger.info("paper_trading_stopped_by_user")
        except Exception as e:
            engine.logger.error("paper_trading_failed", error=str(e))
            raise
        finally:
            engine._checkpoint()
            engine.logger.info("paper_trading_stopped")

    asyncio.run(run_loop())


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/paper.yaml"
    run_paper_trading(config_path)
