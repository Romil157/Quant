"""Unit tests for ML feature pipeline."""
import numpy as np
import pandas as pd

from quant.ml.features import (
    FeatureConfig,
    FeaturePipeline,
    create_default_pipeline,
    create_minimal_pipeline,
)


def create_sample_ohlcv(symbols: list[str], days: int = 300) -> dict:
    """Create sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=days, freq="B")

    data = {}
    for i, sym in enumerate(symbols):
        drift = 0.0002 * (i + 1)
        vol = 0.01 * (1 + i * 0.2)
        returns = np.random.normal(drift, vol, days)
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, days))),
            'close': prices,
            'volume': np.random.lognormal(13, 0.5, days).astype(int),
        }, index=dates)
        data[sym] = df

    return data


def test_feature_config():
    """Test feature configuration."""
    config = FeatureConfig(
        use_returns=True,
        return_horizons=[1, 5, 10],
        use_sma=True,
        sma_windows=[20, 50],
        target_horizon=5,
        target_type="returns",
    )

    assert config.use_returns is True
    assert config.return_horizons == [1, 5, 10]
    assert config.target_horizon == 5


def test_minimal_pipeline():
    """Test minimal feature pipeline."""
    data = create_sample_ohlcv(['AAPL'], 100)

    pipeline = create_minimal_pipeline()
    X, y = pipeline.fit_transform(data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 0
    assert 'AAPL_ret_1' in X.columns or 'AAPL_ret_5' in X.columns


def test_default_pipeline():
    """Test default feature pipeline."""
    data = create_sample_ohlcv(['AAPL', 'MSFT'], 200)

    pipeline = create_default_pipeline()
    X, y = pipeline.fit_transform(data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 50  # Should have enough samples after warmup

    # Check cross-asset features
    assert any('corr' in col for col in X.columns) or any('ratio' in col for col in X.columns)


def test_pipeline_transform():
    """Test transform on new data."""
    train_data = create_sample_ohlcv(['AAPL'], 200)
    test_data = create_sample_ohlcv(['AAPL'], 50)

    pipeline = create_minimal_pipeline()
    X_train, y_train = pipeline.fit_transform(train_data)

    X_test = pipeline.transform(test_data)

    assert list(X_test.columns) == list(X_train.columns)


def test_target_types():
    """Test different target types."""
    data = create_sample_ohlcv(['AAPL'], 100)

    for target_type in ["returns", "direction", "volatility"]:
        config = FeatureConfig(target_type=target_type, target_horizon=5)
        pipeline = FeaturePipeline(config)
        X, y = pipeline.fit_transform(data)

        assert y.name.endswith('_5')
        assert len(y.dropna()) > 0


def test_feature_names():
    """Test feature names property."""
    data = create_sample_ohlcv(['AAPL'], 100)

    pipeline = create_minimal_pipeline()
    pipeline.fit_transform(data)

    names = pipeline.feature_names
    assert isinstance(names, list)
    assert len(names) > 0


def test_preprocessing():
    """Test preprocessing options."""
    data = create_sample_ohlcv(['AAPL'], 100)

    # Test robust scaler
    config = FeatureConfig(scaler="robust")
    pipeline = FeaturePipeline(config)
    X, y = pipeline.fit_transform(data)

    # Should be scaled (mean ~ 0, std ~ 1 for robust)
    assert X.std().mean() < 10  # Not exactly 1 due to robust scaling

    # Test no scaler
    config = FeatureConfig(scaler="none")
    pipeline = FeaturePipeline(config)
    X, y = pipeline.fit_transform(data)

    # Should not be scaled
    assert X.std().mean() > 0.01


def test_fill_missing():
    """Test missing value filling."""
    data = create_sample_ohlcv(['AAPL'], 200)

    pipeline = create_minimal_pipeline()
    X, y = pipeline.fit_transform(data)

    # Introduce NaN in the middle (where no natural NaN exists)
    col = X.columns[0]
    X.loc[X.index[50:55], col] = np.nan

    X_filled = pipeline.fill_missing(X)

    # Check that the introduced NaN was filled
    assert not X_filled.loc[X.index[50:55], col].isna().any()

    # Also check that fill_missing doesn't crash
    X_filled2 = pipeline.fill_missing(X_filled)
    assert X_filled2.shape == X.shape


def test_get_feature_importance_df():
    """Test feature importance DataFrame creation."""
    data = create_sample_ohlcv(['AAPL'], 100)

    pipeline = create_minimal_pipeline()
    pipeline.fit_transform(data)

    importance = np.random.rand(len(pipeline.feature_names))
    df = pipeline.get_feature_importance_df(importance)

    assert isinstance(df, pd.DataFrame)
    assert 'feature' in df.columns
    assert 'importance' in df.columns
    assert len(df) == len(pipeline.feature_names)
    assert df['importance'].is_monotonic_decreasing
