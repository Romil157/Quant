"""Event-driven backtesting engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from quant.backtest.execution import ExecutionConfig, ExecutionSimulator
from quant.backtest.types import (
    Account,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Trade,
)
from quant.portfolio.construction import (
    ConstructionMethod,
    PortfolioConstraints,
    equal_weight,
    inverse_volatility,
    minimum_variance,
    risk_parity,
    volatility_targeting,
)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 1_000_000
    start_date: str | None = None
    end_date: str | None = None
    timeframe: str = "1d"

    # Execution
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Portfolio
    construction_method: ConstructionMethod = ConstructionMethod.EQUAL_WEIGHT
    portfolio_constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)

    # Risk
    max_drawdown: float = 0.20
    max_drawdown_action: str = "reduce_exposure"

    # Benchmark
    benchmark_symbols: list[str] = field(default_factory=list)


class Strategy:
    """Base strategy class."""

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        """
        Generate signals for current bar.

        Returns:
            Series with symbol as index and signal strength (-1 to 1) as values
        """
        raise NotImplementedError


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.execution = ExecutionSimulator(config.execution)

        # State
        self.portfolio = Portfolio(cash=config.initial_capital, initial_capital=config.initial_capital)
        self.current_time: datetime | None = None
        self.current_prices: dict[str, float] = {}
        self.current_bid: dict[str, float] = {}
        self.current_ask: dict[str, float] = {}

        # History
        self.orders: list[Order] = []
        self.fills: list[Fill] = []
        self.account_history: list[Account] = []
        self.trades: list[Trade] = []
        self.daily_returns: list[float] = []
        self.equity_curve: list[float] = []

        # Risk
        self.peak_equity = config.initial_capital
        self.max_drawdown_hit = False

        # Strategy
        self.strategy: Strategy | None = None

    def set_strategy(self, strategy: Strategy) -> None:
        self.strategy = strategy

    def run(
        self,
        data: dict[str, pd.DataFrame],
        strategy: Strategy | None = None,
    ) -> dict:
        """
        Run backtest on provided data.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
            strategy: Optional strategy (uses self.strategy if not provided)

        Returns:
            Dict with backtest results
        """
        if strategy:
            self.set_strategy(strategy)

        if not self.strategy:
            raise ValueError("No strategy set")

        # Align data to common timeline
        aligned_data = self._align_data(data)

        # Run simulation bar by bar
        for timestamp, bar_data in aligned_data.iterrows():
            self.current_time = timestamp
            self._update_market_data(bar_data)

            # Mark to market
            self.portfolio.mark_to_market(self.current_prices)

            # Risk checks
            self._check_risk_limits()

            # Generate signals - convert Series to DataFrame for strategy
            bar_df = bar_data.to_frame().T
            bar_df.index = [timestamp]
            signals = self.strategy.generate_signals(bar_df, timestamp)

            # Portfolio construction
            target_weights = self._construct_portfolio(signals)

            # Rebalance
            self._rebalance(target_weights)

            # Record state
            self._record_state()

        # Finalize
        return self._generate_results()

    def _align_data(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align multi-symbol data to common timeline."""
        # Use first symbol's index as base
        base_symbol = list(data.keys())[0]
        base_index = data[base_symbol].index

        # Reindex all to base index (forward fill)
        aligned = {}
        for symbol, df in data.items():
            aligned[symbol] = df.reindex(base_index).ffill()

        # Create multi-index DataFrame with (symbol, field) columns
        return pd.concat(aligned, axis=1)

    def _update_market_data(self, bar_data: pd.Series) -> None:
        """Update current market prices from bar data (row)."""
        for col in bar_data.index:
            if isinstance(col, tuple):
                symbol, field = col
            else:
                continue

            if field == 'close':
                self.current_prices[symbol] = bar_data[col]
            elif field == 'bid':
                self.current_bid[symbol] = bar_data[col]
            elif field == 'ask':
                self.current_ask[symbol] = bar_data[col]

        # Default bid/ask from close if not provided
        for symbol in self.current_prices:
            if symbol not in self.current_bid:
                self.current_bid[symbol] = self.current_prices[symbol] * 0.9999
            if symbol not in self.current_ask:
                self.current_ask[symbol] = self.current_prices[symbol] * 1.0001

    def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        current_equity = self.portfolio.total_value

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            # Recovered to a new peak -> re-arm the drawdown tripwire.
            self.max_drawdown_hit = False

        # Check drawdown
        if self.peak_equity <= 0:
            return
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.config.max_drawdown:
            if self.config.max_drawdown_action == "reduce_exposure" and not self.max_drawdown_hit:
                # Reduce every open position by half once per drawdown episode.
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
            self.max_drawdown_hit = True

    def _construct_portfolio(self, signals: pd.Series) -> pd.Series:
        """Construct target portfolio from signals."""
        # Filter active signals
        active_signals = signals[signals != 0]

        if len(active_signals) == 0:
            return pd.Series(dtype=float)

        # Map construction method
        method = self.config.construction_method
        constraints = self.config.portfolio_constraints

        if method == ConstructionMethod.EQUAL_WEIGHT:
            return equal_weight(active_signals, constraints)

        elif method == ConstructionMethod.INVERSE_VOLATILITY:
            # Need volatility estimates - use rolling 20-day for now
            volatilities = pd.Series(0.02, index=active_signals.index)  # placeholder
            return inverse_volatility(active_signals, volatilities, constraints)

        elif method == ConstructionMethod.VOLATILITY_TARGETING:
            volatilities = pd.Series(0.02, index=active_signals.index)
            target = constraints.target_volatility or 0.15
            return volatility_targeting(active_signals, volatilities, target, constraints)

        elif method == ConstructionMethod.RISK_PARITY:
            # Need covariance matrix
            n = len(active_signals)
            cov = np.eye(n) * 0.0004
            cov_df = pd.DataFrame(cov, index=active_signals.index, columns=active_signals.index)
            return risk_parity(cov_df, constraints)

        elif method == ConstructionMethod.MINIMUM_VARIANCE:
            n = len(active_signals)
            cov = np.eye(n) * 0.0004
            cov_df = pd.DataFrame(cov, index=active_signals.index, columns=active_signals.index)
            return minimum_variance(cov_df, constraints)

        else:
            # Default to equal weight
            return equal_weight(active_signals, constraints)

    def _rebalance(self, target_weights: pd.Series) -> None:
        """Rebalance portfolio to target weights."""
        current_value = self.portfolio.total_value

        for symbol, target_weight in target_weights.items():
            target_value = current_value * target_weight
            current_pos = self.portfolio.get_position(symbol)
            current_value_pos = current_pos.market_value

            diff = target_value - current_value_pos

            if abs(diff) > current_value * 0.001:  # 10 bps threshold
                price = self.current_prices.get(symbol, 0)
                if price > 0:
                    qty = diff / price

                    if qty != 0:
                        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
                        order = Order(
                            symbol=symbol,
                            side=side,
                            quantity=abs(qty),
                            order_type=OrderType.MARKET,
                            timestamp=self.current_time,
                        )

                        self._submit_order(order)

    def _submit_order(self, order: Order) -> None:
        """Submit order for execution."""
        order.status = OrderStatus.SUBMITTED
        self.orders.append(order)

        # Simulate fill
        price = self.current_prices.get(order.symbol)
        bid = self.current_bid.get(order.symbol)
        ask = self.current_ask.get(order.symbol)

        if price is None:
            order.status = OrderStatus.REJECTED
            return

        fill = self.execution.simulate_fill(order, price, bid, ask)

        if fill:
            self._process_fill(fill, order)
        else:
            order.status = OrderStatus.REJECTED

    def _process_fill(self, fill: Fill, order: Order) -> None:
        """Process fill and update portfolio."""
        order.filled_quantity = fill.quantity
        order.avg_fill_price = fill.price
        order.commission = fill.commission
        order.status = OrderStatus.FILLED

        self.fills.append(fill)

        # Update portfolio: cash impact sign depends on side. Previously the
        # engine subtracted fill.net_value for both BUY and SELL, which
        # incorrectly charged cash (instead of crediting) when selling.
        qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        if fill.side == OrderSide.BUY:
            # Buyer pays notional plus commission.
            self.portfolio.cash -= fill.value + fill.commission
        else:
            # Seller receives notional minus commission.
            self.portfolio.cash += fill.value - fill.commission
        self.portfolio.update_position(fill.symbol, qty, fill.price)

    def _record_state(self) -> None:
        """Record current portfolio state."""
        equity = self.portfolio.total_value
        self.equity_curve.append(equity)

        # Daily return
        if len(self.equity_curve) > 1:
            ret = (equity / self.equity_curve[-2]) - 1
            self.daily_returns.append(ret)

        # Account snapshot
        account = Account(
            timestamp=self.current_time or datetime.now(),
            cash=self.portfolio.cash,
            positions=self.portfolio.positions.copy(),
            total_value=equity,
            gross_exposure=self.portfolio.gross_exposure,
            net_exposure=self.portfolio.net_exposure,
        )
        self.account_history.append(account)

    def _generate_results(self) -> dict:
        """Generate backtest results."""
        returns = pd.Series(self.daily_returns)
        equity = pd.Series(self.equity_curve)

        return {
            'equity_curve': equity,
            'returns': returns,
            'orders': self.orders,
            'fills': self.fills,
            'trades': self.trades,
            'account_history': self.account_history,
            'final_equity': equity.iloc[-1] if len(equity) > 0 else self.config.initial_capital,
            'total_return': (equity.iloc[-1] / self.config.initial_capital - 1) if len(equity) > 0 else 0,
            'max_drawdown_hit': self.max_drawdown_hit,
        }
