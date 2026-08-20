"""Built-in strategies for backtesting.

These wrap the indicator-driven strategies in :mod:`quant.strategies.signals`
behind simpler default parameter sets, plus :class:`BuyAndHoldStrategy` and
:class:`PairTradingStrategy`, and expose a :data:`STRATEGY_REGISTRY` so the
helper scripts and REST API can dispatch a strategy by name.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pandas as pd

from quant.backtest.engine import Strategy
from quant.features import z_score
from quant.strategies.signals import (
    BreakoutSignalStrategy,
    DualMomentumStrategy,
    MACDMomentumStrategy,
    MeanReversionSignalStrategy,
    MomentumSignalStrategy,
    extract_symbols,
)


class BuyAndHoldStrategy(Strategy):
    """Equal-weight long on every available symbol, established on the first bar."""

    def __init__(self, symbols: list[str] | None = None):
        self.symbols = list(symbols) if symbols is not None else None
        self._signaled = False

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        if self._signaled:
            return pd.Series(dtype=float)

        symbols = self.symbols or _extract_symbols(data)
        if not symbols:
            return pd.Series(dtype=float)

        weight = 1.0 / len(symbols)
        self._signaled = True
        return pd.Series(dict.fromkeys(symbols, weight))


class MomentumStrategy(MomentumSignalStrategy):
    """Time-series momentum strategy (wraps :class:`MomentumSignalStrategy`)."""

    def __init__(
        self,
        lookback: int = 126,
        rebalance_freq: int = 21,
        top_n: int = 10,
        min_momentum: float = 0.0,
    ):
        super().__init__(
            lookback_windows=[lookback],
            rebalance_freq=rebalance_freq,
            top_n=top_n,
            min_momentum=min_momentum,
        )


class MeanReversionStrategy(MeanReversionSignalStrategy):
    """Mean reversion strategy (wraps :class:`MeanReversionSignalStrategy`)."""

    def __init__(
        self,
        rsi_window: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rebalance_freq: int = 5,
    ):
        super().__init__(
            rsi_window=rsi_window,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            bb_window=bb_window,
            bb_std=bb_std,
            rebalance_freq=rebalance_freq,
        )


class BreakoutStrategy(BreakoutSignalStrategy):
    """Breakout strategy (wraps :class:`BreakoutSignalStrategy`)."""

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        atr_window: int = 14,
        atr_mult: float = 2.0,
        rebalance_freq: int = 1,
    ):
        super().__init__(
            entry_window=entry_window,
            exit_window=exit_window,
            atr_window=atr_window,
            atr_mult=atr_mult,
            rebalance_freq=rebalance_freq,
        )


class PairTradingStrategy(Strategy):
    """Statistical-arbitrage pair trading on a two-symbol universe.

    Computes the rolling z-score of the price ratio between the first two
    symbols. Enters a long/short pair when ``|z|`` exceeds ``entry_z`` and
    flattens when it returns inside ``exit_z``.
    """

    def __init__(
        self,
        lookback: int = 63,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        rebalance_freq: int = 1,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.rebalance_freq = rebalance_freq
        self.last_rebalance: datetime | None = None
        self.ratio_history: pd.Series | None = None
        self.current_state: int = 0  # -1, 0, +1

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        if self.last_rebalance is not None:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.rebalance_freq:
                return pd.Series(dtype=float)

        self.last_rebalance = current_time

        symbols = _extract_symbols(data)
        if len(symbols) < 2:
            return pd.Series(dtype=float)

        leg_a, leg_b = symbols[0], symbols[1]
        price_a = _close_price(data, leg_a)
        price_b = _close_price(data, leg_b)
        if price_a is None or price_b is None or price_b <= 0:
            return pd.Series(dtype=float)

        ratio = price_a / price_b
        if self.ratio_history is None:
            self.ratio_history = pd.Series(dtype=float)
        self.ratio_history.loc[current_time] = ratio

        if len(self.ratio_history) < self.lookback:
            return pd.Series(dtype=float)

        z = z_score(self.ratio_history, self.lookback).iloc[-1]
        if pd.isna(z):
            return pd.Series(dtype=float)

        if self.current_state == 0 and abs(z) > self.entry_z:
            self.current_state = 1 if z > 0 else -1
        elif self.current_state != 0 and abs(z) < self.exit_z:
            self.current_state = 0

        if self.current_state == 0:
            return pd.Series(dtype=float)

        # Equal-dollar legs: long the underperformer, short the outperformer.
        if self.current_state > 0:
            # ratio high -> a rich, b cheap -> short a, long b
            return pd.Series({leg_a: -0.5, leg_b: 0.5})
        return pd.Series({leg_a: 0.5, leg_b: -0.5})


def _extract_symbols(data: pd.DataFrame) -> list[str]:
    return extract_symbols(data)


def _close_price(data: pd.DataFrame, symbol: str) -> float | None:
    col = (symbol, "close")
    if col in data.columns:
        value = data[col].iloc[-1]
        if pd.notna(value):
            return float(value)
    return None


STRATEGY_REGISTRY: dict[str, Callable[..., Strategy]] = {
    "buy_and_hold": BuyAndHoldStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "macd": MACDMomentumStrategy,
    "dual_momentum": DualMomentumStrategy,
    "pair_trading": PairTradingStrategy,
}


def create_strategy(name: str, **params) -> Strategy:
    """Instantiate a strategy by registry name with keyword parameters."""
    if name not in STRATEGY_REGISTRY:
        valid = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy {name!r}. Valid strategies: {valid}")
    return STRATEGY_REGISTRY[name](**params)
