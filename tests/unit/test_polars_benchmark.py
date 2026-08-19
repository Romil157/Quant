"""Unit tests for polars benchmark module."""
from benchmarks.benchmark_polars import benchmark_pandas_features, benchmark_polars_features


def test_benchmark_pandas_features():
    """Test pandas benchmark execution."""
    elapsed = benchmark_pandas_features(n_bars=1000)
    assert elapsed > 0


def test_benchmark_polars_features():
    """Test polars benchmark execution (runs safely even if polars not installed)."""
    elapsed = benchmark_polars_features(n_bars=1000)
    assert elapsed >= 0
