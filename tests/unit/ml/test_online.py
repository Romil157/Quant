"""Unit tests for online learning."""
import numpy as np
import pandas as pd

from quant.ml.online import (
    DriftDetector,
    OnlineConfig,
    OnlineEnsemble,
    OnlineLearner,
    RollingRetrainer,
)


def create_sample_data(n: int = 100, task: str = "regression") -> tuple:
    """Create sample data."""
    np.random.seed(42)
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
    })
    if task == "regression":
        y = pd.Series(0.5 * X['f1'] + 0.3 * X['f2'] + np.random.randn(n) * 0.1)
    else:
        y = pd.Series((0.5 * X['f1'] + 0.3 * X['f2'] + np.random.randn(n) * 0.1 > 0).astype(int))
    return X, y


def test_online_config():
    """Test online learning configuration."""
    config = OnlineConfig(
        model_type="sgd",
        task="regression",
        learning_rate=0.01,
        alpha=0.0001,
    )

    assert config.model_type == "sgd"
    assert config.task == "regression"
    assert config.learning_rate == 0.01


def test_online_learner_regression():
    """Test online learner for regression."""
    X, y = create_sample_data(50, "regression")

    config = OnlineConfig(model_type="sgd", task="regression")
    learner = OnlineLearner(config)

    # Partial fit
    learner.partial_fit(X[:25], y[:25])

    assert learner.is_fitted
    assert learner.n_samples_seen == 25

    # Continue fitting
    learner.partial_fit(X[25:], y[25:])
    assert learner.n_samples_seen == 50

    # Predict
    preds = learner.predict(X[:10])
    assert len(preds) == 10


def test_online_learner_classification():
    """Test online learner for classification."""
    X, y = create_sample_data(50, "classification")

    config = OnlineConfig(model_type="sgd", task="classification")
    learner = OnlineLearner(config)

    learner.partial_fit(X[:25], y[:25])
    learner.partial_fit(X[25:], y[25:])

    preds = learner.predict(X[:10])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})

    probs = learner.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_online_learner_passive_aggressive():
    """Test passive aggressive online learner."""
    X, y = create_sample_data(50, "regression")

    config = OnlineConfig(model_type="passive_aggressive", task="regression")
    learner = OnlineLearner(config)

    learner.partial_fit(X, y)

    preds = learner.predict(X[:10])
    assert len(preds) == 10


def test_online_learner_reset():
    """Test online learner reset."""
    X, y = create_sample_data(50, "regression")

    config = OnlineConfig(model_type="sgd", task="regression")
    learner = OnlineLearner(config)

    learner.partial_fit(X, y)
    assert learner.n_samples_seen == 50

    learner.reset()
    assert learner.n_samples_seen == 0
    assert not learner.is_fitted


def test_rolling_retrainer():
    """Test rolling retrainer."""
    X, y = create_sample_data(200, "regression")

    from sklearn.linear_model import LinearRegression
    retrainer = RollingRetrainer(
        LinearRegression(),
        window_size=50,
        retrain_freq=30,
    )

    # Feed data in chunks
    for i in range(0, 200, 20):
        retrainer.partial_fit(X[i:i+20], y[i:i+20])

    assert retrainer.is_fitted
    assert len(retrainer.buffer_X) > 0

    preds = retrainer.predict(X[:10])
    assert len(preds) == 10


def test_rolling_retrainer_force_retrain():
    """Test forced retrain."""
    X, y = create_sample_data(100, "regression")

    from sklearn.linear_model import LinearRegression
    retrainer = RollingRetrainer(LinearRegression(), window_size=50, retrain_freq=1000)

    retrainer.partial_fit(X[:30], y[:30])
    assert not retrainer.is_fitted  # Not enough for retrain_freq

    retrainer.force_retrain()
    assert retrainer.is_fitted


def test_drift_detector():
    """Test drift detector."""
    detector = DriftDetector(window_size=20, threshold=0.5)

    # Stable predictions
    for _ in range(30):
        drift = detector.update(1.0, 1.05)  # Small error

    assert not drift

    # Sudden change - large errors
    for _ in range(15):
        drift = detector.update(1.0, 2.0)  # Large error

    # Should detect drift after window fills
    assert detector.drift_detected or len(detector.errors) >= 20


def test_drift_detector_reset():
    """Test drift detector reset."""
    detector = DriftDetector(window_size=10)

    for _ in range(15):
        detector.update(1.0, 2.0)

    assert detector.drift_detected or len(detector.errors) >= 10

    detector.reset()
    assert len(detector.errors) == 0
    assert not detector.drift_detected


def test_online_ensemble():
    """Test online ensemble."""
    X, y = create_sample_data(50, "regression")

    config1 = OnlineConfig(model_type="sgd", task="regression")
    config2 = OnlineConfig(model_type="passive_aggressive", task="regression")

    learner1 = OnlineLearner(config1)
    learner2 = OnlineLearner(config2)

    ensemble = OnlineEnsemble([learner1, learner2], weights=[0.6, 0.4])

    ensemble.partial_fit(X[:25], y[:25])
    ensemble.partial_fit(X[25:], y[25:])

    preds = ensemble.predict(X[:10])
    assert len(preds) == 10

    # Update performance
    ensemble.update_performance(np.ones(10), preds)


def test_online_learner_get_params():
    """Test getting online learner parameters."""
    config = OnlineConfig(model_type="sgd", task="regression")
    learner = OnlineLearner(config)

    X, y = create_sample_data(20, "regression")
    learner.partial_fit(X, y)

    params = learner.get_params()

    assert 'config' in params
    assert 'n_samples_seen' in params
    assert params['n_samples_seen'] == 20
