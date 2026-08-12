"""Unit tests for ParquetProvider path traversal security checks."""
from pathlib import Path

import pytest

from quant.data.providers.parquet import ParquetProvider


def test_parquet_provider_rejects_path_traversal(tmp_path: Path):
    """Test that ParquetProvider rejects path traversal symbols."""
    provider = ParquetProvider(data_root=tmp_path)

    invalid_symbols = [
        "../../etc/passwd",
        "..\\..\\Windows",
        "../secret",
        "AAPL/../../etc",
        "AAPL;drop table",
        "AAPL<script>",
        "../../",
    ]

    for symbol in invalid_symbols:
        with pytest.raises(ValueError, match="(Invalid symbol format|Symbol path is outside data_root)"):
            provider._get_symbol_path(symbol)


def test_parquet_provider_accepts_valid_symbols(tmp_path: Path):
    """Test that ParquetProvider accepts valid symbols."""
    provider = ParquetProvider(data_root=tmp_path)

    valid_symbols = ["AAPL", "GOOG_L", "BRK.A", "EUR-USD", "BTC_USDT"]
    for symbol in valid_symbols:
        path = provider._get_symbol_path(symbol)
        assert path.is_relative_to(tmp_path.resolve())
