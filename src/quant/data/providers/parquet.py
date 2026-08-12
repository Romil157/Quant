"""Local Parquet market data provider."""
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.data.providers.base import DataNotAvailableError, MarketDataProvider, SymbolNotFoundError


class ParquetProvider(MarketDataProvider):
    """Reads OHLCV data from local Parquet files.

    Expected directory structure:
        data_root/
            provider_name/
                asset_class/
                    symbol/
                        year/
                            data.parquet
    """

    def __init__(
        self,
        data_root: Path,
        provider: str = "default",
        asset_class: str = "equities",
    ):
        """
        Initialize Parquet provider.

        Args:
            data_root: Root directory containing provider/asset/symbol/year/data.parquet
            provider: Provider subdirectory name
            asset_class: Asset class subdirectory (equities, futures, crypto, forex)
        """
        self._data_root = Path(data_root)
        self._provider = provider
        self._asset_class = asset_class
        self._symbol_cache: dict[str, list[int]] = {}

    def _get_symbol_path(self, symbol: str) -> Path:
        return self._data_root / self._provider / self._asset_class / symbol

    def _get_available_years(self, symbol: str) -> list[int]:
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        symbol_path = self._get_symbol_path(symbol)
        if not symbol_path.exists():
            return []

        years = []
        for year_dir in symbol_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                years.append(int(year_dir.name))

        years.sort()
        self._symbol_cache[symbol] = years
        return years

    def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        if timeframe != "1d":
            raise DataNotAvailableError(f"Parquet provider only supports '1d' timeframe, got {timeframe}")

        years = self._get_available_years(symbol)
        if not years:
            raise SymbolNotFoundError(f"No data found for symbol {symbol}")

        start_year = start.year
        end_year = end.year

        relevant_years = [y for y in years if start_year <= y <= end_year]
        if not relevant_years:
            raise DataNotAvailableError(f"No data for {symbol} in range {start.year}-{end.year}")

        dfs = []
        for year in relevant_years:
            parquet_file = self._get_symbol_path(symbol) / str(year) / "data.parquet"
            if not parquet_file.exists():
                continue

            try:
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
            except Exception as e:
                raise DataNotAvailableError(f"Failed to read {parquet_file}: {e}") from e

        if not dfs:
            raise DataNotAvailableError(f"No readable data files for {symbol} in range")

        combined = pd.concat(dfs).sort_index()

        # Filter to requested date range
        mask = (combined.index >= pd.Timestamp(start)) & (combined.index <= pd.Timestamp(end))
        result = combined.loc[mask]

        if result.empty:
            raise DataNotAvailableError(f"No data for {symbol} in range {start} to {end}")

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in result.columns]
        if missing:
            raise DataNotAvailableError(f"Missing required columns: {missing}")

        return result[required_cols].copy()

    def get_available_symbols(self) -> list[str]:
        asset_path = self._data_root / self._provider / self._asset_class
        if not asset_path.exists():
            return []

        symbols = []
        for symbol_dir in asset_path.iterdir():
            if symbol_dir.is_dir():
                # Check if it has at least one year directory with data.parquet
                has_data = False
                for year_dir in symbol_dir.iterdir():
                    if year_dir.is_dir() and (year_dir / "data.parquet").exists():
                        has_data = True
                        break
                if has_data:
                    symbols.append(symbol_dir.name)

        return sorted(symbols)

    def get_available_timeframes(self) -> list[str]:
        return ["1d"]

    def get_symbol_date_range(self, symbol: str) -> tuple[datetime, datetime] | None:
        """Get the actual date range available for a symbol."""
        years = self._get_available_years(symbol)
        if not years:
            return None

        # Read first and last year to get exact range
        first_file = self._get_symbol_path(symbol) / str(years[0]) / "data.parquet"
        last_file = self._get_symbol_path(symbol) / str(years[-1]) / "data.parquet"

        try:
            first_df = pd.read_parquet(first_file)
            last_df = pd.read_parquet(last_file)
            return (first_df.index[0].to_pydatetime(), last_df.index[-1].to_pydatetime())
        except Exception:
            return None
