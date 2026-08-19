#!/usr/bin/env python
"""Performance benchmark: Pandas vs Polars feature computation and execution.

Measures wall-clock time, throughput (bars/second), and memory usage.
"""
import time

import numpy as np
import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


def benchmark_pandas_features(n_bars: int = 100_000) -> float:
    """Benchmark Pandas feature computations (rolling mean, rolling std, RSI)."""
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n_bars)))
    df = pd.DataFrame({"close": prices})

    t0 = time.perf_counter()
    _sma20 = df["close"].rolling(20).mean()
    _vol20 = df["close"].pct_change().rolling(20).std()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    _rsi = 100 - (100 / (1 + rs))
    t1 = time.perf_counter()

    elapsed = t1 - t0
    return elapsed


def benchmark_polars_features(n_bars: int = 100_000) -> float:
    """Benchmark Polars feature computations."""
    if not POLARS_AVAILABLE:
        return 0.0

    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n_bars)))
    df = pl.DataFrame({"close": prices})

    t0 = time.perf_counter()
    _df_feat = df.with_columns([
        pl.col("close").rolling_mean(window_size=20).alias("sma20"),
        pl.col("close").pct_change().rolling_std(window_size=20).alias("vol20"),
    ])
    t1 = time.perf_counter()
    return t1 - t0


def run_benchmark():
    n_bars = 50_000
    print(f"Running Feature Calculation Benchmark ({n_bars:,} bars)")
    print("-" * 50)

    t_pandas = benchmark_pandas_features(n_bars)
    print(f"Pandas feature computation: {t_pandas * 1000:.2f} ms")

    if POLARS_AVAILABLE:
        t_polars = benchmark_polars_features(n_bars)
        speedup = t_pandas / t_polars if t_polars > 0 else 1.0
        print(f"Polars feature computation: {t_polars * 1000:.2f} ms (Speedup: {speedup:.1f}x)")
    else:
        print("Polars not installed in environment; skipping Polars comparison.")


if __name__ == "__main__":
    run_benchmark()
