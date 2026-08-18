"""Unit tests for the YFinanceProvider.

These tests run only when the optional ``yfinance`` package is installed.
The provider is loaded via :func:`pytest.importorskip` so missing-dependency
environments simply skip these tests instead of failing.

We monkeypatch ``yfinance.download`` to avoid hitting the network in CI.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

yfinance = pytest.importorskip("yfinance")  # noqa: N816

from quant.data.providers.yfinance_provider import YFinanceProvider  # noqa: E402


def _fake_ohlcv_frame(symbol: str, n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    base = 100.0 if symbol == "AAPL" else 200.0
    closes = base * (1.001 + 0.0001) ** pd.RangeIndex(n)
    return pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.005,
            "Low": closes * 0.995,
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


def make_provider(monkeypatch, df_to_return):
    """Create a ``YFinanceProvider`` and patch yf.download to return ``df``."""
    df_to_return.index.name = "Date"
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: df_to_return)
    return YFinanceProvider()


def test_yfin_provider_column_normalization(monkeypatch):
    df = _fake_ohlcv_frame("AAPL")
    provider = make_provider(monkeypatch, df)

    result = provider.get_historical_data(
        "AAPL",
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 9),
    )
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "timestamp"
    assert result.index.tz is None
    assert len(result) == 5
    # Slice to (start, end] inclusive — yfinance interval semantics make end+1.
    assert result.index.min() >= pd.Timestamp("2023-01-01")
    assert result.index.max() <= pd.Timestamp("2023-01-09")


def test_yfin_provider_raises_on_empty(monkeypatch):
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    provider = YFinanceProvider()
    from quant.data.providers.base import SymbolNotFoundError

    with pytest.raises(SymbolNotFoundError):
        provider.get_historical_data("UNOBTAINIUM", datetime(2023, 1, 1), datetime(2023, 1, 10))


def test_yfin_provider_invalid_timeframe():
    provider = YFinanceProvider()
    from quant.data.providers.base import DataNotAvailableError

    with pytest.raises(DataNotAvailableError):
        provider.get_historical_data("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 10), timeframe="17d")


def test_yfin_provider_factory_dispatch():
    """The factory must dispatch ``yfinance`` to ``YFinanceProvider``."""
    from quant.data.providers.factory import ProviderFactory
    from quant.data.providers.yfinance_provider import YFinanceProvider

    p = ProviderFactory.create_provider("yfinance")
    assert isinstance(p, YFinanceProvider)


def test_yfin_provider_available_timeframes():
    provider = YFinanceProvider()
    tfs = provider.get_available_timeframes()
    for expected in ("1d", "1h", "1m", "1wk", "1mo"):
        assert expected in tfs
