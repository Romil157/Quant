"""Time-series cross-validation and model evaluation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


@dataclass
class TimeSeriesCVConfig:
    """Configuration for time-series cross-validation."""
    n_splits: int = 5
    train_size: int | None = None  # None = expanding
    test_size: int = 63  # ~3 months
    gap: int = 0  # Embargo gap between train and test
    step: int | None = None  # Step between folds, None = test_size
    expanding: bool = True  # Expanding vs rolling window
    anchor: str = "end"  # "end" or "start" - anchor point for rolling


@dataclass
class CVResult:
    """Result of a single CV fold."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_score: float
    test_score: float
    model: Any
    predictions: pd.Series
    actuals: pd.Series


@dataclass
class CVSummary:
    """Aggregated CV results."""
    fold_results: list[CVResult]
    mean_train_score: float
    std_train_score: float
    mean_test_score: float
    std_test_score: float
    mean_fit_time: float
    score_stability: float  # 1 - CV of test scores


class TimeSeriesCV:
    """Time-series cross-validation with walk-forward splits."""

    def __init__(self, config: TimeSeriesCVConfig):
        self.config = config

    def split(self, X: pd.DataFrame, y: pd.Series | None = None) -> list[tuple]:
        """Generate train/test indices for each fold."""
        n = len(X)
        indices = np.arange(n)

        step = self.config.test_size if self.config.step is None else self.config.step

        folds = []

        if self.config.expanding:
            # Expanding window: train grows, test fixed size
            min_train = self.config.train_size or self.config.test_size * 2
            start = min_train
        else:
            # Rolling window: fixed train size
            if self.config.train_size is None:
                raise ValueError("train_size required for rolling window")
            start = self.config.train_size

        for _fold, i in enumerate(range(start, n - self.config.test_size + 1, step)):
            if self.config.expanding:
                train_start_idx = 0
                train_end_idx = i - self.config.gap - 1
            else:
                train_size = self.config.train_size or self.config.test_size * 2
                train_start_idx = i - train_size
                train_end_idx = i - self.config.gap - 1

            if train_end_idx < train_start_idx:
                continue

            test_start_idx = train_end_idx + 1 + self.config.gap
            test_end_idx = test_start_idx + self.config.test_size - 1

            if test_end_idx >= n:
                break

            train_idx = indices[train_start_idx:train_end_idx + 1]
            test_idx = indices[test_start_idx:test_end_idx + 1]

            folds.append((train_idx, test_idx))

        return folds

    def get_fold_dates(self, X: pd.DataFrame, fold: tuple) -> tuple:
        """Get date boundaries for a fold."""
        train_idx, test_idx = fold
        return (
            X.index[train_idx[0]],
            X.index[train_idx[-1]],
            X.index[test_idx[0]],
            X.index[test_idx[-1]],
        )


def cross_validate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesCV,
    scoring: Callable[[np.ndarray, np.ndarray], float] | None = None,
    fit_params: dict[str, Any] | None = None,
    return_models: bool = False,
) -> CVSummary:
    """
    Run time-series cross-validation.

    Args:
        model: sklearn-compatible estimator
        X: Features DataFrame
        y: Target Series
        cv: TimeSeriesCV instance
        scoring: Scoring function (y_true, y_pred) -> float
        fit_params: Additional params for model.fit()
        return_models: Whether to store fitted models

    Returns:
        CVSummary with all fold results
    """
    if scoring is None:
        scoring = r2_score

    folds = cv.split(X, y)
    fit_params = fit_params or {}

    results = []
    fit_times = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_start, train_end, test_start, test_end = cv.get_fold_dates(X, (train_idx, test_idx))

        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Skip if not enough data
        if len(X_train) < 10 or len(X_test) < 5:
            continue

        # Clone and fit model
        fold_model = clone(model)

        import time
        t0 = time.time()
        fold_model.fit(X_train, y_train, **fit_params)
        fit_time = time.time() - t0
        fit_times.append(fit_time)

        # Predict
        y_train_pred = fold_model.predict(X_train)
        y_test_pred = fold_model.predict(X_test)

        # Score
        train_score = scoring(y_train, y_train_pred)
        test_score = scoring(y_test, y_test_pred)

        # Store predictions
        preds = pd.Series(y_test_pred, index=X_test.index, name='pred')
        actuals = y_test.copy()
        actuals.name = 'actual'

        result = CVResult(
            fold=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_score=train_score,
            test_score=test_score,
            model=fold_model if return_models else None,
            predictions=preds,
            actuals=actuals,
        )
        results.append(result)

    if not results:
        raise ValueError("No valid folds produced")

    test_scores = [r.test_score for r in results]
    train_scores = [r.train_score for r in results]

    # Score stability = 1 - coefficient of variation
    score_stability = 1 - (np.std(test_scores) / (np.mean(test_scores) + 1e-8)) if np.mean(test_scores) != 0 else 0

    return CVSummary(
        fold_results=results,
        mean_train_score=float(np.mean(train_scores)),
        std_train_score=float(np.std(train_scores)),
        mean_test_score=float(np.mean(test_scores)),
        std_test_score=float(np.std(test_scores)),
        mean_fit_time=float(np.mean(fit_times)),
        score_stability=float(score_stability),
    )


def evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "regression",
) -> dict:
    """Evaluate model on train and test sets."""
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {}

    if task == "regression":
        metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
        metrics['test_mse'] = mean_squared_error(y_test, y_test_pred)
        metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        metrics['train_r2'] = r2_score(y_train, y_train_pred)
        metrics['test_r2'] = r2_score(y_test, y_test_pred)
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
    elif task == "classification":
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['train_precision'] = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['test_precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['train_recall'] = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['test_recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['train_f1'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    return metrics


def walk_forward_predict(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesCV,
    retrain: bool = True,
) -> pd.DataFrame:
    """
    Generate walk-forward predictions.

    Args:
        model: Base estimator
        X: Features
        y: Target
        cv: TimeSeriesCV config
        retrain: Whether to retrain on each fold (True) or use single model (False)

    Returns:
        DataFrame with columns: actual, predicted, fold
    """
    folds = cv.split(X, y)
    all_preds = []

    if not retrain:
        # Train once on all data before first test
        first_fold = folds[0]
        train_idx = first_fold[0]
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if retrain:
            fold_model = clone(model)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        else:
            fold_model = model

        y_pred = fold_model.predict(X.iloc[test_idx])

        pred_df = pd.DataFrame({
            'actual': y.iloc[test_idx].values,
            'predicted': y_pred,
            'fold': fold_idx,
        }, index=X.index[test_idx])

        all_preds.append(pred_df)

    return pd.concat(all_preds).sort_index()


def purged_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
) -> list[tuple]:
    """
    Purged K-Fold CV (Lopez de Prado).

    Removes embargo periods between train and test to prevent leakage
    from autocorrelation.
    """
    n = len(X)
    indices = np.arange(n)
    embargo = int(n * pct_embargo)

    # Create splits
    fold_size = n // n_splits
    folds = []

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)

        # Purge: remove embargo from train
        train_start = 0
        train_end = test_start - embargo

        if train_end > train_start:
            train_idx = indices[train_start:train_end]
        else:
            train_idx = np.array([], dtype=int)

        test_idx = indices[test_start:test_end]

        folds.append((train_idx, test_idx))

    return folds


def combinatorial_purged_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_test_splits: int = 2,
    pct_embargo: float = 0.01,
) -> list[tuple]:
    """
    Combinatorial Purged K-Fold CV (Lopez de Prado).

    Creates multiple test set combinations for more robust validation.
    """
    from itertools import combinations

    n = len(X)
    np.arange(n)
    embargo = int(n * pct_embargo)
    fold_size = n // n_splits

    # Generate all combinations of test folds
    test_fold_indices = list(range(n_splits))
    test_combos = list(combinations(test_fold_indices, n_test_splits))

    folds = []

    for combo in test_combos:
        # Test indices from selected folds
        test_indices: list[int] = []
        for fold_idx in combo:
            test_start = fold_idx * fold_size
            test_end = min((fold_idx + 1) * fold_size, n)
            test_indices.extend(range(test_start, test_end))

        test_indices_arr = np.array(test_indices)

        # Train indices: everything else minus embargo
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_indices_arr] = False

        # Apply embargo around test sets
        for idx in test_indices_arr:
            start = max(0, idx - embargo)
            end = min(n, idx + embargo + 1)
            train_mask[start:end] = False

        train_indices = np.where(train_mask)[0]

        if len(train_indices) > 0 and len(test_indices_arr) > 0:
            folds.append((train_indices, test_indices_arr))

    return folds
