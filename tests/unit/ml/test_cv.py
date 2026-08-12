"""Unit tests for ML cross-validation."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from quant.ml.cv import (
    CVResult,
    CVSummary,
    TimeSeriesCV,
    TimeSeriesCVConfig,
    combinatorial_purged_cv,
    cross_validate,
    evaluate_model,
    purged_kfold_cv,
    walk_forward_predict,
)


def create_sample_data(n: int = 500) -> tuple:
    """Create sample data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    X = pd.DataFrame({
        'f1': np.random.randn(n).cumsum(),
        'f2': np.random.randn(n).cumsum(),
        'f3': np.random.randn(n).cumsum(),
    }, index=dates)
    y = pd.Series(0.5 * X['f1'] + 0.3 * X['f2'] + np.random.randn(n) * 0.1, index=dates, name='target')
    return X, y


def test_cv_config():
    """Test CV configuration."""
    config = TimeSeriesCVConfig(
        n_splits=5,
        train_size=200,
        test_size=63,
        gap=5,
        expanding=False,
    )

    assert config.n_splits == 5
    assert config.train_size == 200
    assert config.test_size == 63
    assert config.gap == 5
    assert config.expanding is False


def test_cv_split():
    """Test CV split generation."""
    X, y = create_sample_data(500)

    config = TimeSeriesCVConfig(
        n_splits=5,
        train_size=100,
        test_size=50,
        gap=5,
        expanding=False,
    )
    cv = TimeSeriesCV(config)

    folds = cv.split(X, y)

    assert len(folds) > 0
    # With rolling window and step=test_size, we get multiple folds
    assert len(folds) <= 8  # Actual number depends on data length

    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert train_idx[-1] < test_idx[0]  # Train before test
        assert test_idx[0] - train_idx[-1] > 5  # Gap respected


def test_cv_expanding():
    """Test expanding window CV."""
    X, y = create_sample_data(300)

    config = TimeSeriesCVConfig(
        n_splits=5,
        test_size=30,
        expanding=True,
    )
    cv = TimeSeriesCV(config)

    folds = cv.split(X, y)

    # First fold train should start at 0
    assert folds[0][0][0] == 0

    # Each fold train window expands
    for i in range(1, len(folds)):
        assert folds[i][0][0] == 0
        assert folds[i][0][-1] > folds[i-1][0][-1]


def test_cv_get_dates():
    """Test getting fold dates."""
    X, y = create_sample_data(200)

    config = TimeSeriesCVConfig(
        train_size=50,
        test_size=20,
        gap=0,
    )
    cv = TimeSeriesCV(config)

    folds = cv.split(X, y)
    train_start, train_end, test_start, test_end = cv.get_fold_dates(X, folds[0])

    assert train_start < train_end
    assert train_end < test_start
    assert test_start < test_end


def test_cross_validate():
    """Test cross_validate function."""
    X, y = create_sample_data(300)

    config = TimeSeriesCVConfig(
        train_size=50,
        test_size=20,
        expanding=True,
    )
    cv = TimeSeriesCV(config)
    model = LinearRegression()

    summary = cross_validate(model, X, y, cv)

    assert isinstance(summary, CVSummary)
    assert len(summary.fold_results) > 0
    assert hasattr(summary, 'mean_test_score')
    assert hasattr(summary, 'std_test_score')
    assert hasattr(summary, 'score_stability')

    for result in summary.fold_results:
        assert isinstance(result, CVResult)
        assert result.train_score is not None
        assert result.test_score is not None
        assert isinstance(result.predictions, pd.Series)
        assert isinstance(result.actuals, pd.Series)


def test_evaluate_model():
    """Test evaluate_model function."""
    X, y = create_sample_data(200)

    # Split
    split = 150
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, task="regression")

    assert 'train_mse' in metrics
    assert 'test_mse' in metrics
    assert 'train_r2' in metrics
    assert 'test_r2' in metrics
    assert 'train_rmse' in metrics
    assert 'test_rmse' in metrics


def test_walk_forward_predict():
    """Test walk_forward_predict function."""
    X, y = create_sample_data(300)

    config = TimeSeriesCVConfig(
        train_size=50,
        test_size=20,
        expanding=True,
    )
    cv = TimeSeriesCV(config)
    model = LinearRegression()

    preds = walk_forward_predict(model, X, y, cv, retrain=True)

    assert isinstance(preds, pd.DataFrame)
    assert 'actual' in preds.columns
    assert 'predicted' in preds.columns
    assert 'fold' in preds.columns
    assert len(preds) > 0


def test_purged_kfold_cv():
    """Test purged K-fold CV."""
    X, y = create_sample_data(500)

    folds = purged_kfold_cv(X, y, n_splits=5, pct_embargo=0.02)

    assert len(folds) == 5

    for train_idx, test_idx in folds:
        assert len(test_idx) > 0
        # Train should not overlap with test (plus embargo)
        if len(train_idx) > 0:
            assert train_idx[-1] < test_idx[0]


def test_combinatorial_purged_cv():
    """Test combinatorial purged CV."""
    X, y = create_sample_data(500)

    folds = combinatorial_purged_cv(X, y, n_splits=5, n_test_splits=2, pct_embargo=0.01)

    # Should have C(5,2) = 10 combinations
    assert len(folds) == 10

    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
