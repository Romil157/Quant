"""Data management and ingestion module."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.data.providers.factory import ProviderFactory
from quant.data.validation import validate_ohlc_relationships


def download_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    provider: str = "mock",
    data_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Download data for given symbols using specified provider."""
    if provider == "mock":
        # Allow the MockProvider to actually respond for the requested symbols.
        prov = ProviderFactory.create_provider(provider, symbols=symbols)
    else:
        root = data_root or Path("data/raw")
        prov = ProviderFactory.create_provider(provider, data_root=root)

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = prov.get_historical_data(symbol, start, end)
        result[symbol] = df

    return result


def validate_data(
    symbol: str,
    start_date: datetime | str,
    end_date: datetime | str,
) -> dict[str, Any]:
    """Validate data quality for a symbol."""
    start = datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date

    end = datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    prov = ProviderFactory.create_provider("mock", symbols=[symbol])
    try:
        df = prov.get_historical_data(symbol, start, end)
        res = validate_ohlc_relationships(df)
        return {
            "symbol": symbol,
            "valid": res.is_valid,
            "issues": res.errors,
            "warnings": res.warnings,
            "metrics": res.metrics,
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "valid": False,
            "issues": [str(e)],
            "warnings": [],
            "metrics": {},
        }
