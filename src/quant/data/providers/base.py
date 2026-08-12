"""Abstract market data provider interface."""
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "AAPL", "SPY")
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            timeframe: Bar timeframe (e.g., "1m", "1h", "1d")

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index must be DatetimeIndex (timezone-aware preferred)
        """
        ...

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """Return list of available symbols."""
        ...

    @abstractmethod
    def get_available_timeframes(self) -> list[str]:
        """Return list of supported timeframes."""
        ...

    def health_check(self) -> bool:
        """Check if provider is accessible. Override for real providers."""
        return True


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class SymbolNotFoundError(ProviderError):
    """Raised when symbol is not available from provider."""
    pass


class DataNotAvailableError(ProviderError):
    """Raised when requested data range is not available."""
    pass
