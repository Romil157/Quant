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
    maximum_sharpe,
    mean_variance,
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
        # Price history for volatility/covariance calculations
        self.price_history: dict[str, list[float]] = {}

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

        # Strategy & validation tracking
        self.strategy: Strategy | None = None
        self.target_weights_history: list[dict[str, float]] = []

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
        for timestamp_raw, bar_data in aligned_data.iterrows():
            ts: datetime = timestamp_raw if isinstance(timestamp_raw, datetime) else pd.Timestamp(timestamp_raw).to_pydatetime()
            self.current_time = ts
            self._update_market_data(bar_data)

            # Mark to market
            self.portfolio.mark_to_market(self.current_prices)

            # Risk checks
            self._check_risk_limits()

            # Generate signals - convert Series to DataFrame for strategy
            bar_df = bar_data.to_frame().T
            bar_df.index = [ts]
            signals = self.strategy.generate_signals(bar_df, ts)

            # Portfolio construction
            target_weights = self._construct_portfolio(signals)
            if len(target_weights) > 0:
                self.target_weights_history.append(target_weights.to_dict())

            # Rebalance
            self._rebalance(target_weights)

            # Record state
            self._record_state()

        # Finalize & validate
        results = self._generate_results()
        self._validate_results(results)
        return results

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

        # Update price history for volatility/covariance calculations
        for symbol, price in self.current_prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)

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

    def _get_rolling_expected_returns(self, symbols: list[str], window: int = 60) -> pd.Series:
        """Calculate rolling expected returns for each symbol from price history."""
        expected_returns: dict[str, float] = {}
        for symbol in symbols:
            if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                prices = pd.Series(self.price_history[symbol])
                returns = prices.pct_change().dropna()
                if len(returns) > 0:
                    exp_ret = returns.iloc[-window:].mean()
                    expected_returns[symbol] = exp_ret if not np.isnan(exp_ret) else 0.0
                else:
                    expected_returns[symbol] = 0.0
            else:
                expected_returns[symbol] = 0.0
        return pd.Series(expected_returns, index=symbols)

    def _get_rolling_covariance(self, symbols: list[str], window: int = 60) -> pd.DataFrame:
        """Calculate rolling covariance matrix for symbols from price history with shrinkage."""
        return_dict = {}
        for symbol in symbols:
            if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                prices = pd.Series(self.price_history[symbol])
                returns = prices.pct_change().dropna()
                if len(returns) > 0:
                    return_dict[symbol] = returns

        return_matrix = pd.DataFrame(return_dict)

        if len(return_matrix) < 2 or len(return_matrix.columns) < len(symbols):
            # Compute individual variances where possible
            variances = []
            for s in symbols:
                if s in return_matrix and len(return_matrix[s]) > 1:
                    v = float(np.var(return_matrix[s].to_numpy(dtype=float), ddof=1))
                    variances.append(v if v > 1e-8 and not np.isnan(v) else 0.0004)
                else:
                    variances.append(0.0004)
            return pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)

        # Use available history up to window
        recent_returns = return_matrix.iloc[-window:]
        cov_df = recent_returns.cov().reindex(index=symbols, columns=symbols).fillna(0.0)
        cov_val = np.asarray(cov_df.to_numpy(), dtype=float)
        diag_var = np.diag(np.diag(cov_val))
        for i in range(len(symbols)):
            if diag_var[i, i] <= 1e-8 or np.isnan(diag_var[i, i]):
                diag_var[i, i] = 0.0004
                cov_val[i, i] = 0.0004

        shrinkage = 0.15 if len(recent_returns) < window else 0.01
        shrunk_cov = (1.0 - shrinkage) * cov_val + shrinkage * diag_var
        return pd.DataFrame(shrunk_cov, index=symbols, columns=symbols)

    def _construct_portfolio(self, signals: pd.Series) -> pd.Series:
        """Construct target portfolio from signals."""
        # Filter active signals
        active_signals = signals[signals != 0]

        if len(active_signals) == 0:
            return pd.Series(dtype=float)

        symbols = list(active_signals.index)

        # Map construction method
        method = self.config.construction_method
        constraints = self.config.portfolio_constraints

        if method == ConstructionMethod.EQUAL_WEIGHT:
            return equal_weight(active_signals, constraints)

        elif method == ConstructionMethod.INVERSE_VOLATILITY:
            # Calculate rolling volatility for each symbol
            volatilities = {}
            for symbol in symbols:
                if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                    prices = pd.Series(self.price_history[symbol])
                    returns = prices.pct_change().dropna()
                    if len(returns) > 1:
                        vol = returns.iloc[-20:].std()
                        volatilities[symbol] = vol if not np.isnan(vol) and vol > 1e-6 else 0.02
                    else:
                        volatilities[symbol] = 0.02
                else:
                    volatilities[symbol] = 0.02
            vol_series = pd.Series(volatilities, index=symbols)
            return inverse_volatility(active_signals, vol_series, constraints)

        elif method == ConstructionMethod.VOLATILITY_TARGETING:
            volatilities = {}
            for symbol in symbols:
                if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                    prices = pd.Series(self.price_history[symbol])
                    returns = prices.pct_change().dropna()
                    if len(returns) > 1:
                        vol = returns.iloc[-20:].std()
                        volatilities[symbol] = vol if not np.isnan(vol) and vol > 1e-6 else 0.02
                    else:
                        volatilities[symbol] = 0.02
                else:
                    volatilities[symbol] = 0.02
            vol_series = pd.Series(volatilities, index=symbols)
            target = constraints.target_volatility or 0.15
            return volatility_targeting(active_signals, vol_series, target, constraints)

        elif method == ConstructionMethod.RISK_PARITY:
            # Calculate rolling covariance matrix
            cov_df = self._get_rolling_covariance(symbols, window=60)
            return risk_parity(cov_df, constraints)

        elif method == ConstructionMethod.MINIMUM_VARIANCE:
            cov_df = self._get_rolling_covariance(symbols, window=60)
            return minimum_variance(cov_df, constraints)

        elif method == ConstructionMethod.MEAN_VARIANCE:
            # Calculate expected returns and covariance matrix
            expected_returns = self._get_rolling_expected_returns(symbols, window=60)
            cov_df = self._get_rolling_covariance(symbols, window=60)
            risk_aversion = getattr(constraints, 'risk_aversion', 1.0)
            return mean_variance(expected_returns, cov_df, risk_aversion, constraints)

        elif method == ConstructionMethod.MAXIMUM_SHARPE:
            # Calculate expected returns and covariance matrix
            expected_returns = self._get_rolling_expected_returns(symbols, window=60)
            cov_df = self._get_rolling_covariance(symbols, window=60)
            risk_free_rate = getattr(constraints, 'risk_free_rate', 0.0)
            return maximum_sharpe(expected_returns, cov_df, risk_free_rate, constraints)

        else:
            # Default to equal weight
            return equal_weight(active_signals, constraints)

    def _rebalance(self, target_weights: pd.Series) -> None:
        """Rebalance portfolio to target weights."""
        current_value = self.portfolio.total_value

        for symbol_raw, target_weight_raw in target_weights.items():
            symbol: str = str(symbol_raw)
            target_weight: float = float(target_weight_raw)
            target_value = current_value * target_weight
            current_pos = self.portfolio.get_position(symbol)
            current_value_pos = current_pos.market_value

            diff = target_value - current_value_pos

            if abs(diff) > current_value * 0.001:  # 10 bps threshold
                price = self.current_prices.get(symbol, 0.0)
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

    def _validate_results(self, results: dict) -> None:
        """Strategy validation layer: assert weights sum to 1, no NaNs, and check degradation."""
        # 1. Check equity curve and returns for NaNs / Infs
        equity = results.get("equity_curve", pd.Series(dtype=float))
        if len(equity) > 0 and (equity.isna().any() or np.isinf(equity).any()):
            raise ValueError("Strategy validation error: equity curve contains NaN or Inf values")

        returns = results.get("returns", pd.Series(dtype=float))
        if len(returns) > 0 and (returns.isna().any() or np.isinf(returns).any()):
            raise ValueError("Strategy validation error: returns contain NaN or Inf values")

        # 2. Check target weights history
        if self.target_weights_history:
            for idx, w_dict in enumerate(self.target_weights_history):
                if not w_dict:
                    continue
                w_series = pd.Series(w_dict)
                if w_series.isna().any() or np.isinf(w_series).any():
                    raise ValueError(f"Strategy validation error: target weights at step {idx} contain NaN or Inf")
                w_sum = w_series.sum()
                if abs(w_sum - 1.0) > 1e-3 and abs(w_sum) > 1e-4:
                    raise ValueError(f"Strategy validation error: target weights at step {idx} sum to {w_sum:.4f} (expected ~1.0)")

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
            'target_weights_history': self.target_weights_history,
        }

