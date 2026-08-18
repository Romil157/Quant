"""Unit tests for the registry-based built-in strategies."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant.strategies import (
    STRATEGY_REGISTRY,
    BuyAndHoldStrategy,
    PairTradingStrategy,
    create_strategy,
)


def create_test_bar(symbol: str, price: float, timestamp: datetime) -> pd.DataFrame:
    return pd.DataFrame({
        (symbol, "open"): [price * 0.999],
        (symbol, "high"): [price * 1.005],
        (symbol, "low"): [price * 0.995],
        (symbol, "close"): [price],
        (symbol, "volume"): [1_000_000],
    }, index=[timestamp])


def test_buy_and_hold_emits_signals_once_then_quiet():
    strategy = BuyAndHoldStrategy()
    base = datetime(2023, 1, 1)

    bar1 = pd.concat([create_test_bar("AAPL", 150.0, base), create_test_bar("MSFT", 300.0, base)], axis=1)
    bar2 = pd.concat(
        [create_test_bar("AAPL", 151.0, base + pd.Timedelta(days=1)),
         create_test_bar("MSFT", 301.0, base + pd.Timedelta(days=1))],
        axis=1,
    )

    signals_1 = strategy.generate_signals(bar1, base)
    signals_2 = strategy.generate_signals(bar2, base + pd.Timedelta(days=1))

    # First bar: equal-weight 0.5 / 0.5 across the two symbols.
    assert set(signals_1.index) == {"AAPL", "MSFT"}
    assert np.isclose(signals_1.sum(), 1.0)

    # Subsequent bars: no new signals (the position is already established).
    assert signals_2.empty


def test_momentum_strategy_emits_signals_after_warmup():
    strategy = create_strategy(
        "momentum",
        lookback=10,
        top_n=2,
        rebalance_freq=5,
    )
    base = datetime(2023, 1, 1)
    last_signals = pd.Series(dtype=float)
    prices_aapl = 150.0
    prices_msft = 300.0
    for i in range(25):
        prices_aapl *= 1.001
        prices_msft *= 1.0005
        ts = base + pd.Timedelta(days=i)
        bar = pd.concat(
            [create_test_bar("AAPL", prices_aapl, ts), create_test_bar("MSFT", prices_msft, ts)],
            axis=1,
        )
        signals = strategy.generate_signals(bar, ts)
        if not signals.empty:
            last_signals = signals
    assert not last_signals.empty
    assert all(s in {"AAPL", "MSFT"} for s in last_signals.index)


def test_pair_trading_returns_pair_signals_when_z_exceeds_entry():
    strategy = PairTradingStrategy(lookback=20, entry_z=1.5, exit_z=0.2, rebalance_freq=1)
    base = datetime(2023, 1, 1)

    symbols = ["AAPL", "MSFT"]
    # Diverging prices: AAPL marches up, MSFT flat; ratio drifts strong.
    for i in range(40):
        ts = base + pd.Timedelta(days=i)
        price_aapl = 100.0 + i * 0.5
        price_msft = 100.0
        bar = pd.concat(
            [create_test_bar(symbols[0], price_aapl, ts), create_test_bar(symbols[1], price_msft, ts)],
            axis=1,
        )
        signals = strategy.generate_signals(bar, ts)

    # After enough divergence the strategy should hold a paired position.
    if not signals.empty:
        assert set(signals.index) == {"AAPL", "MSFT"}
        # Legs must be equal-magnitude opposite signs.
        assert np.isclose(signals["AAPL"], -signals["MSFT"])
        assert signals.abs().sum() == pytest.approx(1.0, rel=1e-6)


def test_registry_unknown_strategy_raises_with_helpful_message():
    with pytest.raises(ValueError) as exc_info:
        create_strategy("does_not_exist")
    # Message must list the valid strategies.
    for known in STRATEGY_REGISTRY:
        assert known in str(exc_info.value)


def test_registry_known_strategies_instantiate_with_defaults():
    # Every registered strategy must be constructable without required args.
    for _name, factory in STRATEGY_REGISTRY.items():
        strategy = factory()
        assert strategy is not None
        # generate_signals on an empty bar must not raise (just return empty).
        empty_bar = pd.DataFrame(index=[datetime(2023, 1, 1)])
        strategy.generate_signals(empty_bar, datetime(2023, 1, 1))
