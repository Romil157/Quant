"""Integration test for ML benchmark comparison."""
import pandas as pd
import pytest

from quant.data import download_data
from quant.ml.features import create_minimal_pipeline
from quant.ml.models import ModelConfig, RidgeModel


def test_ml_pipeline_feature_generation_and_training():
    """Test feature generation and training pipeline on market data."""
    data = download_data(symbols=["SPY"], start_date="2022-01-01", end_date="2023-06-01", provider="mock")

    pipeline = create_minimal_pipeline()
    X, y = pipeline.fit_transform(data)

    assert not X.empty
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    valid_mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    model = RidgeModel(ModelConfig(name="ridge", model_type="regression"))
    model.fit(X_clean, y_clean)
    preds = model.predict(X_clean)

    assert len(preds) == len(X_clean)
