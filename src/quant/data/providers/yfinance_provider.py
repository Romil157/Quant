"""Yahoo Finance market data provider.

Uses the optional ``yfinance`` package to fetch OHLCV data for equities and
ETFs. The provider is intentionally defensive: column names and indexing are
normalized to match the rest of the platform (lowercase
``open/high/low/close/volume``, ``DatetimeIndex`` named ``timestamp``).

Install the optional dependency to enable this provider::

    uv sync --extra live
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant.data.providers.base import DataNotAvailableError, MarketDataProvider, SymbolNotFoundError

_YFINANCE_TIMEFRAMES = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}


class YFinanceProvider(MarketDataProvider):
    """Fetches OHLCV data via the ``yfinance`` library."""

    def __init__(self, **_kwargs):
        """Additional keyword arguments are ignored so the factory's
        ``data_root=...`` works for parquet-style invocations.

        Raises:
            ImportError: if ``yfinance`` is not installed.
        """
        try:
            import yfinance as yf  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "yfinance is required for YFinanceProvider. "
                "Install the optional 'live' extra: `uv sync --extra live` or "
                "`pip install -e .[live]`."
            ) from e

    def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        if timeframe not in _YFINANCE_TIMEFRAMES:
            raise DataNotAvailableError(f"Unsupported yfinance timeframe: {timeframe!r}")

        import yfinance as yf

        # yfinance's end is exclusive; bump it by one day so callers using an
        # inclusive end (matching !MockProvider/ParquetProvider) get the data
        # they expect.
        yf_end = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        yf_start = start.strftime("%Y-%m-%d")

        try:
            df = yf.download(
                symbol,
                start=yf_start,
                end=yf_end,
                interval=timeframe,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as e:
            raise DataNotAvailableError(f"yfinance download failed for {symbol!r}: {e}") from e

        if df is None or df.empty:
            raise SymbolNotFoundError(f"No yfinance data for symbol {symbol!r} in {yf_start}..{yf_end}")

        # yfinance may return a MultiIndex on columns when a single symbol is
        # requested; collapse to a flat index so output is uniform.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        # Ticker column can appear on multi-symbol calls; drop if present.
        if "Ticker" in df.columns:
            df = df.drop(columns=["Ticker"])

        # Locate the timestamp column (yfinance names it "Date" for 1d or
        # "Datetime" for intraday).
        ts_col = next((c for c in ("Datetime", "Date", "timestamp") if c in df.columns), None)
        if ts_col is None:
            raise DataNotAvailableError("yfinance result did not contain a timestamp column")
        df = df.rename(columns={ts_col: "timestamp"})

        # Normalize the remaining columns to the platform's lowercase contract.
        rename_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        df.index.name = "timestamp"

        # Make timezone-naive so it matches what MockProvider and ParquetProvider emit.
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataNotAvailableError(f"yfinance data missing columns {missing} for {symbol!r}")

        return df[required].sort_index().copy()

    def get_available_symbols(self) -> list[str]:
        # yfinance has no enumeration endpoint; the caller supplies the universe.
        return []

    def get_available_timeframes(self) -> list[str]:
        return list(_YFINANCE_TIMEFRAMES)

    def health_check(self) -> bool:
        try:
            import yfinance as yf  # noqa: F401
            return True
        except ImportError:
            return False
