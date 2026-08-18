"""Strategies package."""
from quant.strategies.builtins import (
    STRATEGY_REGISTRY,
    BreakoutStrategy,
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    PairTradingStrategy,
    create_strategy,
)
from quant.strategies.signals import (
    BreakoutSignalStrategy,
    DualMomentumStrategy,
    MACDMomentumStrategy,
    MeanReversionSignalStrategy,
    MomentumSignalStrategy,
)

__all__ = [
    "BuyAndHoldStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "PairTradingStrategy",
    "MomentumSignalStrategy",
    "MeanReversionSignalStrategy",
    "BreakoutSignalStrategy",
    "MACDMomentumStrategy",
    "DualMomentumStrategy",
    "STRATEGY_REGISTRY",
    "create_strategy",
]
