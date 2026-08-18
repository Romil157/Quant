"""Provider factory for configurable market data providers."""
from pathlib import Path

from quant.data.providers.base import MarketDataProvider
from quant.data.providers.mock import MockProvider
from quant.data.providers.parquet import ParquetProvider
from quant.data.providers.yfinance_provider import YFinanceProvider


class ProviderFactory:
    """Factory for creating market data providers from configuration."""

    _providers: dict[str, MarketDataProvider] = {}

    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        **kwargs,
    ) -> MarketDataProvider:
        """Create a provider instance by type."""
        if provider_type == "mock":
            return MockProvider(**kwargs)
        elif provider_type == "parquet":
            data_root = kwargs.pop("data_root", Path("data/raw"))
            return ParquetProvider(data_root=data_root, **kwargs)
        elif provider_type == "yfinance":
            return YFinanceProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @classmethod
    def get_provider(
        cls,
        name: str,
        provider_type: str,
        **kwargs,
    ) -> MarketDataProvider:
        """Get or create a cached provider instance."""
        if name not in cls._providers:
            cls._providers[name] = cls.create_provider(provider_type, **kwargs)
        return cls._providers[name]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear provider cache."""
        cls._providers.clear()
