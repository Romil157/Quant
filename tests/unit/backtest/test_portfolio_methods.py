"""Unit tests for portfolio construction methods wired into BacktestEngine."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestConfig, BacktestEngine, Strategy
from quant.backtest.execution import ExecutionConfig
from quant.portfolio.construction import ConstructionMethod, PortfolioConstraints


class MultiSymbolSignalStrategy(Strategy):
    """Strategy generating signals across multiple symbols."""

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.bars = 0

    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> pd.Series:
        self.bars += 1
        # Generate positive signals on every bar
        return pd.Series(dict.fromkeys(self.symbols, 1.0))


def create_multisymbol_data(symbols: list[str], days: int = 120) -> dict[str, pd.DataFrame]:
    """Create distinct OHLCV time series for multiple symbols."""
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    data = {}
    for i, symbol in enumerate(symbols):
        np.random.seed(100 + i * 37)
        # Give each symbol distinct volatility and trend
        vol = 0.01 * (1 + i * 0.5)
        drift = 0.0005 * (i + 1)
        returns = np.random.normal(drift, vol, days)
        prices = 100 * np.exp(np.cumsum(returns))

        data[symbol] = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "volume": np.random.randint(10000, 50000, days),
            },
            index=dates,
        )
    return data


@pytest.mark.parametrize(
    "method",
    [
        ConstructionMethod.EQUAL_WEIGHT,
        ConstructionMethod.INVERSE_VOLATILITY,
        ConstructionMethod.VOLATILITY_TARGETING,
        ConstructionMethod.RISK_PARITY,
        ConstructionMethod.MINIMUM_VARIANCE,
        ConstructionMethod.MEAN_VARIANCE,
        ConstructionMethod.MAXIMUM_SHARPE,
    ],
)
def test_all_construction_methods_execute_in_engine(method: ConstructionMethod):
    """Test that every ConstructionMethod enum value runs successfully in BacktestEngine."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = create_multisymbol_data(symbols, days=100)

    config = BacktestConfig(
        initial_capital=100_000,
        construction_method=method,
        portfolio_constraints=PortfolioConstraints(
            max_position=0.8,
            max_gross_exposure=1.0,
            long_only=True,
            target_volatility=0.15,
        ),
        execution=ExecutionConfig(commission_bps=1.0, spread_bps=1.0, slippage_bps=1.0),
    )

    engine = BacktestEngine(config)
    strategy = MultiSymbolSignalStrategy(symbols)
    results = engine.run(data, strategy)

    assert "equity_curve" in results
    assert "returns" in results
    assert len(results["equity_curve"]) == 100
    assert results["final_equity"] > 0


def test_construction_methods_produce_distinct_weights():
    """Verify that different construction methods produce non-identical weight distributions."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = create_multisymbol_data(symbols, days=100)

    weights_by_method: dict[ConstructionMethod, pd.Series] = {}

    for method in [
        ConstructionMethod.EQUAL_WEIGHT,
        ConstructionMethod.INVERSE_VOLATILITY,
        ConstructionMethod.MINIMUM_VARIANCE,
        ConstructionMethod.MEAN_VARIANCE,
    ]:
        config = BacktestConfig(
            initial_capital=100_000,
            construction_method=method,
            portfolio_constraints=PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0, long_only=True),
        )
        engine = BacktestEngine(config)

        # Pre-populate price history to simulate warmed-up state
        for s in symbols:
            engine.price_history[s] = list(data[s]["close"].values)

        active_signals = pd.Series(dict.fromkeys(symbols, 1.0))
        target_weights = engine._construct_portfolio(active_signals)
        weights_by_method[method] = target_weights

        assert len(target_weights) == len(symbols)
        assert abs(target_weights.sum() - 1.0) < 1e-4

    # Equal weight should be exactly 1/3 each
    eq_w = weights_by_method[ConstructionMethod.EQUAL_WEIGHT]
    for s in symbols:
        assert abs(eq_w[s] - 1 / 3) < 1e-5

    # Inverse vol should not be all equal because volatilities differ
    iv_w = weights_by_method[ConstructionMethod.INVERSE_VOLATILITY]
    assert not np.allclose(iv_w.values, eq_w.values, atol=1e-3)
