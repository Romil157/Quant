"""Strategies package."""
from quant.strategies.builtins import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    PairTradingStrategy,
)
from quant.strategies.signals import (
    BreakoutSignalStrategy,
    DualMomentumStrategy,
    MACDMomentumStrategy,
    MeanReversionSignalStrategy,
    MomentumSignalStrategy,
)

__all__ = [
    "MomentumStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "PairTradingStrategy",
    "MomentumSignalStrategy",
    "MeanReversionSignalStrategy",
    "BreakoutSignalStrategy",
    "MACDMomentumStrategy",
    "DualMomentumStrategy",
]
