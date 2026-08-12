"""Unit tests for ML models."""
import numpy as np
import pandas as pd

from quant.ml.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    LinearRegressionModel,
    LogisticRegressionModel,
    ModelConfig,
    RandomForestClsModel,
    RandomForestRegModel,
    create_model,
    ensemble_predict,
    get_model_class,
    tune_hyperparameters,
)


def create_sample_data(n: int = 200, task: str = "regression") -> tuple:
    """Create sample data."""
    np.random.seed(42)
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n),
    })
    if task == "regression":
        y = pd.Series(0.5 * X['f1'] + 0.3 * X['f2'] + np.random.randn(n) * 0.1)
    else:
        y = pd.Series((0.5 * X['f1'] + 0.3 * X['f2'] + np.random.randn(n) * 0.1 > 0).astype(int))
    return X, y


def test_model_config():
    """Test model configuration."""
    config = ModelConfig(
        name="rf",
        model_type="regression",
        params={"n_estimators": 100},
        param_grid={"n_estimators": [50, 100]},
    )

    assert config.name == "rf"
    assert config.model_type == "regression"
    assert config.params["n_estimators"] == 100


def test_get_model_class():
    """Test getting model class."""
    # Regression
    cls = get_model_class("rf", "regression")
    assert cls == RandomForestRegModel

    cls = get_model_class("linear", "regression")
    assert cls == LinearRegressionModel

    # Classification
    cls = get_model_class("rf", "classification")
    assert cls == RandomForestClsModel

    cls = get_model_class("logistic", "classification")
    assert cls == LogisticRegressionModel

    # Unknown
    try:
        get_model_class("unknown", "regression")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_create_model():
    """Test model creation."""
    model = create_model("rf", "regression", {"n_estimators": 50})

    assert isinstance(model, RandomForestRegModel)
    assert model.config.params["n_estimators"] == 50
    assert model.config.model_type == "regression"


def test_model_fit_predict():
    """Test model fit and predict."""
    X, y = create_sample_data(100, "regression")

    model = create_model("linear", "regression")
    model.fit(X, y)

    assert model.is_fitted
    assert len(model.feature_names) == 3

    preds = model.predict(X)
    assert len(preds) == len(X)


def test_model_classification():
    """Test classification model."""
    X, y = create_sample_data(100, "classification")

    model = create_model("logistic", "classification")
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_feature_importance():
    """Test feature importance extraction."""
    X, y = create_sample_data(100, "regression")

    model = create_model("rf", "regression", {"n_estimators": 50, "random_state": 42})
    model.fit(X, y)

    importance = model.get_feature_importance()

    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert len(importance) == 3
    assert importance['importance'].sum() > 0


def test_ridge_model():
    """Test Ridge regression model."""
    X, y = create_sample_data(100, "regression")

    model = create_model("ridge", "regression", {"alpha": 1.0})
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)

    importance = model.get_feature_importance()
    assert len(importance) == 3


def test_tune_hyperparameters():
    """Test hyperparameter tuning."""
    X, y = create_sample_data(150, "regression")

    model = create_model("ridge", "regression")
    tuned = tune_hyperparameters(
        model, X, y,
        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
        cv=3,
        n_iter=4,
        method="random",
    )

    assert tuned.is_fitted
    assert 'alpha' in tuned.config.params


def test_ensemble_predict():
    """Test ensemble prediction."""
    X, y = create_sample_data(100, "regression")

    model1 = create_model("linear", "regression")
    model1.fit(X, y)

    model2 = create_model("ridge", "regression", {"alpha": 1.0})
    model2.fit(X, y)

    preds = ensemble_predict([model1, model2], X, method="mean")

    assert len(preds) == len(X)

    # Weighted
    preds_w = ensemble_predict([model1, model2], X, weights=[0.7, 0.3], method="mean")
    assert len(preds_w) == len(X)


def test_model_set_params():
    """Test setting model parameters."""
    model = create_model("rf", "regression")
    model.set_params(n_estimators=200, max_depth=10)

    assert model.config.params["n_estimators"] == 200
    assert model.config.params["max_depth"] == 10


def test_regression_models_available():
    """Test all regression models are registered."""
    expected = ['linear', 'ridge', 'lasso', 'elasticnet', 'rf', 'gbr', 'et', 'hgbr', 'svr', 'knn', 'mlp']
    for name in expected:
        assert name in REGRESSION_MODELS


def test_classification_models_available():
    """Test all classification models are registered."""
    expected = ['logistic', 'ridge_cls', 'rf', 'gbc', 'et', 'hgbc', 'svc', 'knn', 'mlp']
    for name in expected:
        assert name in CLASSIFICATION_MODELS
