"""Unit tests for ML pipeline."""
import numpy as np
import pandas as pd

from quant.ml.features import FeatureConfig
from quant.ml.pipeline import (
    MLPipeline,
    MLPipelineConfig,
    MLPipelineResult,
    compare_models,
    run_ml_experiment,
    walk_forward_backtest,
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


def test_pipeline_config():
    """Test ML pipeline configuration."""
    config = MLPipelineConfig(
        model_name="rf",
        task="regression",
        model_params={"n_estimators": 100},
    )

    assert config.model_name == "rf"
    assert config.task == "regression"
    assert config.model_params["n_estimators"] == 100


def test_run_ml_experiment():
    """Test running ML experiment."""
    data = create_sample_ohlcv(['AAPL'], 500)

    result = run_ml_experiment(
        data=data,
        model_name="linear",
        task="regression",
        test_size=0.3,
    )

    assert isinstance(result, MLPipelineResult)
    assert result.model is not None
    assert result.feature_pipeline is not None
    assert result.test_metrics is not None
    assert 'r2' in result.test_metrics
    assert isinstance(result.feature_importance, pd.DataFrame)
    assert isinstance(result.predictions, pd.DataFrame)


def test_ml_pipeline_class():
    """Test MLPipeline class."""
    data = create_sample_ohlcv(['AAPL'], 500)

    config = MLPipelineConfig(
        model_name="ridge",
        task="regression",
        model_params={"alpha": 1.0},
    )

    pipeline = MLPipeline(config)
    result = pipeline.fit(data)

    assert result.model.is_fitted
    assert pipeline.is_fitted

    # Predict on new data
    preds = pipeline.predict(data)
    assert 'predicted' in preds.columns


def test_compare_models():
    """Test model comparison."""
    data = create_sample_ohlcv(['AAPL'], 500)

    results = compare_models(
        data=data,
        model_names=["linear", "ridge"],
        task="regression",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert 'linear' in results.index
    assert 'ridge' in results.index


def test_walk_forward_backtest():
    """Test walk-forward backtest."""
    data = create_sample_ohlcv(['AAPL'], 300)

    config = MLPipelineConfig(
        model_name="linear",
        task="regression",
    )
    pipeline = MLPipeline(config)
    pipeline.fit(data)

    preds = walk_forward_backtest(
        pipeline,
        data,
        initial_train=100,
        step=30,
        retrain=True,
    )

    assert isinstance(preds, pd.DataFrame)
    assert 'actual' in preds.columns
    assert 'predicted' in preds.columns
    assert len(preds) > 0


def test_classification_pipeline():
    """Test classification pipeline."""
    # Create data with direction target
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, 500)))

    data = {
        'AAPL': pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.lognormal(13, 0.5, 500).astype(int),
        }, index=dates),
    }

    config = FeatureConfig(target_type="direction", target_horizon=5)
    pipeline_config = MLPipelineConfig(
        feature_config=config,
        model_name="logistic",
        task="classification",
        model_params={"max_iter": 1000},
    )

    pipeline = MLPipeline(pipeline_config)
    result = pipeline.fit(data)

    assert 'accuracy' in result.test_metrics
    assert 'f1' in result.test_metrics


def test_pipeline_with_tuning():
    """Test pipeline with hyperparameter tuning."""
    data = create_sample_ohlcv(['AAPL'], 500)

    config = MLPipelineConfig(
        model_name="ridge",
        task="regression",
        tune_hyperparams=True,
        model_param_grid={"alpha": [0.1, 1.0, 10.0]},
        tune_n_iter=3,
    )

    pipeline = MLPipeline(config)
    result = pipeline.fit(data)

    assert result.model.is_fitted
    assert 'alpha' in result.model.config.params


def test_predict_proba():
    """Test predict_proba for classification."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, 500)))

    data = {
        'AAPL': pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.lognormal(13, 0.5, 500).astype(int),
        }, index=dates),
    }

    config = FeatureConfig(target_type="direction", target_horizon=5)
    pipeline_config = MLPipelineConfig(
        feature_config=config,
        model_name="logistic",
        task="classification",
        model_params={"max_iter": 1000},
    )

    pipeline = MLPipeline(pipeline_config)
    pipeline.fit(data)

    probs = pipeline.predict_proba(data)
    assert probs.shape[1] == 2  # Binary classification
    assert np.allclose(probs.sum(axis=1), 1.0)
