"""Mock market data provider for testing and development."""
from datetime import datetime

import numpy as np
import pandas as pd

from quant.data.providers.base import DataNotAvailableError, MarketDataProvider, SymbolNotFoundError


class MockProvider(MarketDataProvider):
    """Generates synthetic OHLCV data for testing."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        seed: int = 42,
        start_price: float = 100.0,
        daily_vol: float = 0.02,
        trend: float = 0.0001,
    ):
        """
        Initialize mock provider.

        Args:
            symbols: List of symbols to support. Default: ["MOCK1", "MOCK2", "MOCK3"]
            seed: Random seed for reproducibility
            start_price: Initial price for all symbols
            daily_vol: Daily volatility (std dev of returns)
            trend: Daily drift (expected return)
        """
        self._symbols = symbols or ["MOCK1", "MOCK2", "MOCK3"]
        self._seed = seed
        self._start_price = start_price
        self._daily_vol = daily_vol
        self._trend = trend
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, pd.DataFrame] = {}

    def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        if symbol not in self._symbols:
            raise SymbolNotFoundError(f"Symbol {symbol} not available in mock provider")

        if timeframe != "1d":
            raise DataNotAvailableError(f"Mock provider only supports '1d' timeframe, got {timeframe}")

        cache_key = f"{symbol}_{start.date()}_{end.date()}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Generate business days
        dates = pd.bdate_range(start=start, end=end, freq="B")
        if len(dates) == 0:
            raise DataNotAvailableError(f"No business days in range {start} to {end}")

        n = len(dates)
        # Generate returns with trend and volatility
        returns = self._rng.normal(self._trend, self._daily_vol, n)

        # Generate prices
        prices = self._start_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        # Add intraday noise
        intraday_vol = self._daily_vol * 0.3
        high_low_spread = np.abs(self._rng.normal(0, intraday_vol, n))

        close = prices
        high = close * (1 + high_low_spread)
        low = close * (1 - high_low_spread)
        open_ = np.roll(close, 1)
        open_[0] = self._start_price
        # Add small gap noise to open
        open_ = open_ * (1 + self._rng.normal(0, intraday_vol * 0.5, n))

        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        # Volume - lognormal distribution
        volume = self._rng.lognormal(13, 0.5, n).astype(np.int64)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=pd.DatetimeIndex(dates, name="timestamp"),
        )

        # Ensure timezone naive for simplicity (can be localized by caller)
        self._cache[cache_key] = df
        return df.copy()

    def get_available_symbols(self) -> list[str]:
        return self._symbols.copy()

    def get_available_timeframes(self) -> list[str]:
        return ["1d"]

    def generate_multi_symbol_data(
        self,
        start: datetime,
        end: datetime,
        correlation: float = 0.3,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate correlated multi-symbol data.

        Args:
            start: Start date
            end: End date
            correlation: Pairwise correlation between symbols

        Returns:
            Dict mapping symbol -> DataFrame
        """
        n_symbols = len(self._symbols)
        dates = pd.bdate_range(start=start, end=end, freq="B")
        n = len(dates)

        # Create correlation matrix
        corr_matrix = np.full((n_symbols, n_symbols), correlation)
        np.fill_diagonal(corr_matrix, 1.0)

        # Cholesky decomposition for correlated random variables
        chol = np.linalg.cholesky(corr_matrix)

        # Generate independent returns
        independent_returns = self._rng.normal(
            self._trend, self._daily_vol, (n, n_symbols)
        )

        # Apply correlation
        correlated_returns = independent_returns @ chol.T

        result = {}
        for i, symbol in enumerate(self._symbols):
            returns = correlated_returns[:, i]
            prices = self._start_price * np.exp(np.cumsum(returns))

            intraday_vol = self._daily_vol * 0.3
            high_low_spread = np.abs(self._rng.normal(0, intraday_vol, n))

            close = prices
            high = close * (1 + high_low_spread)
            low = close * (1 - high_low_spread)
            open_ = np.roll(close, 1)
            open_[0] = self._start_price
            open_ = open_ * (1 + self._rng.normal(0, intraday_vol * 0.5, n))

            high = np.maximum(high, np.maximum(open_, close))
            low = np.minimum(low, np.minimum(open_, close))

            volume = self._rng.lognormal(13, 0.5, n).astype(np.int64)

            df = pd.DataFrame(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                },
                index=pd.DatetimeIndex(dates, name="timestamp"),
            )
            result[symbol] = df

        return result
