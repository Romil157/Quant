"""ML Pipeline - end-to-end ML workflow."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quant.ml.cv import CVSummary, TimeSeriesCV, TimeSeriesCVConfig, cross_validate
from quant.ml.features import FeatureConfig, FeaturePipeline
from quant.ml.models import (
    BaseMLModel,
    create_model,
    tune_hyperparameters,
)
from quant.ml.online import OnlineConfig


@dataclass
class MLPipelineConfig:
    """Complete ML pipeline configuration."""
    # Feature engineering
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # Model
    model_name: str = "rf"
    task: str = "regression"  # "regression" or "classification"
    model_params: dict[str, Any] = field(default_factory=dict)
    model_param_grid: dict[str, list] = field(default_factory=dict)

    # Cross-validation
    cv_config: TimeSeriesCVConfig = field(default_factory=TimeSeriesCVConfig)

    # Hyperparameter tuning
    tune_hyperparams: bool = False
    tune_method: str = "random"  # "grid" or "random"
    tune_n_iter: int = 20
    tune_cv: int = 3
    tune_scoring: str = "neg_mean_squared_error"

    # Ensemble
    use_ensemble: bool = False
    ensemble_models: list[str] = field(default_factory=list)
    ensemble_method: str = "mean"  # "mean", "median", "weighted_median"

    # Online learning
    use_online: bool = False
    online_config: OnlineConfig = field(default_factory=OnlineConfig)

    # Training
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class MLPipelineResult:
    """Result of ML pipeline execution."""
    model: BaseMLModel
    feature_pipeline: FeaturePipeline
    cv_summary: CVSummary | None
    test_metrics: dict[str, float]
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame
    config: MLPipelineConfig
    timestamp: datetime


class MLPipeline:
    """End-to-end ML pipeline for financial time series."""

    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self.feature_pipeline: FeaturePipeline | None = None
        self.model: BaseMLModel | None = None
        self.cv_summary: CVSummary | None = None
        self.is_fitted = False

    def fit(
        self,
        data: dict[str, pd.DataFrame],
        tune: bool | None = None,
    ) -> MLPipelineResult:
        """
        Fit the complete pipeline.

        Args:
            data: Dict of symbol -> OHLCV DataFrame
            tune: Whether to tune hyperparameters (overrides config)

        Returns:
            MLPipelineResult with fitted model and metrics
        """
        # Build features
        self.feature_pipeline = FeaturePipeline(self.config.feature_config)
        X, y = self.feature_pipeline.fit_transform(data)

        # Fill missing
        X = self.feature_pipeline.fill_missing(X)
        y = y.loc[X.index]

        # Remove any remaining NaN
        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]


        # Split train/test
        test_size = int(len(X) * self.config.test_size)
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        # Time-series CV
        cv = TimeSeriesCV(self.config.cv_config)
        base_model = create_model(self.config.model_name, self.config.task, self.config.model_params)

        self.cv_summary = cross_validate(
            base_model._get_pipeline(),
            X_train, y_train,
            cv=cv,
            scoring=self._get_scoring_fn(),
        )


        # Hyperparameter tuning
        if tune or (tune is None and self.config.tune_hyperparams):
            base_model = tune_hyperparameters(
                base_model,
                X_train, y_train,
                param_grid=self.config.model_param_grid or self._get_default_param_grid(),
                cv=self.config.tune_cv,
                scoring=self.config.tune_scoring,
                n_iter=self.config.tune_n_iter,
                method=self.config.tune_method,
                random_state=self.config.random_state,
            )

        # Fit final model
        self.model = base_model
        self.model.fit(X_train, y_train)

        # Evaluate on test
        test_metrics = self._evaluate_test(X_test, y_test)

        # Feature importance
        feature_importance = self.model.get_feature_importance()

        # Generate predictions
        y_pred = self.model.predict(X_test)
        predictions = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
        }, index=X_test.index)

        self.is_fitted = True

        return MLPipelineResult(
            model=self.model,
            feature_pipeline=self.feature_pipeline,
            cv_summary=self.cv_summary,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            predictions=predictions,
            config=self.config,
            timestamp=datetime.now(),
        )

    def predict(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate predictions on new data."""
        if not self.is_fitted or self.feature_pipeline is None or self.model is None:
            raise ValueError("Pipeline not fitted")

        X = self.feature_pipeline.transform(data)
        X = self.feature_pipeline.fill_missing(X)

        preds = self.model.predict(X)

        return pd.DataFrame({
            'predicted': preds,
        }, index=X.index)

    def predict_proba(self, data: dict[str, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities (classification)."""
        if not self.is_fitted or self.feature_pipeline is None or self.model is None:
            raise ValueError("Pipeline not fitted")

        X = self.feature_pipeline.transform(data)
        X = self.feature_pipeline.fill_missing(X)

        return self.model.predict_proba(X)

    def _get_scoring_fn(self) -> Callable[[np.ndarray, np.ndarray], float]:
        """Get scoring function for CV."""
        if self.config.task == "regression":
            from sklearn.metrics import r2_score
            return r2_score  # type: ignore
        else:
            from sklearn.metrics import accuracy_score
            return accuracy_score  # type: ignore

    def _evaluate_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Evaluate model on test set."""
        if self.model is None:
            raise ValueError("Model object is None")
        y_pred = self.model.predict(X_test)


        if self.config.task == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            return {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
            }
        else:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }

    def _get_default_param_grid(self) -> dict[str, dict[str, list[Any]]]:
        """Get default parameter grid for model."""
        grids: dict[str, dict[str, list[Any]]] = {
            'rf': {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]},
            'gbr': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
            'ridge': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
            'lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
            'elasticnet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            'svr': {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.2]},
            'mlp': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]},
            'xgb': {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
            'lgb': {'n_estimators': [100, 200], 'num_leaves': [31, 63], 'learning_rate': [0.01, 0.1]},
        }
        return grids.get(self.config.model_name, {})  # type: ignore


def run_ml_experiment(
    data: dict[str, pd.DataFrame],
    model_name: str = "rf",
    task: str = "regression",
    feature_config: FeatureConfig | None = None,
    cv_config: TimeSeriesCVConfig | None = None,
    model_params: dict | None = None,
    param_grid: dict | None = None,
    tune: bool = False,
    test_size: float = 0.2,
) -> MLPipelineResult:
    """Convenience function to run ML experiment."""
    config = MLPipelineConfig(
        feature_config=feature_config or FeatureConfig(),
        model_name=model_name,
        task=task,
        model_params=model_params or {},
        model_param_grid=param_grid or {},
        cv_config=cv_config or TimeSeriesCVConfig(),
        tune_hyperparams=tune,
        test_size=test_size,
    )

    pipeline = MLPipeline(config)
    return pipeline.fit(data, tune=tune)


def compare_models(
    data: dict[str, pd.DataFrame],
    model_names: list[str],
    task: str = "regression",
    feature_config: FeatureConfig | None = None,
    cv_config: TimeSeriesCVConfig | None = None,
) -> pd.DataFrame:
    """Compare multiple models."""
    results = []

    for name in model_names:
        try:
            result = run_ml_experiment(
                data=data,
                model_name=name,
                task=task,
                feature_config=feature_config,
                cv_config=cv_config,
            )
            results.append({
                'model': name,
                'cv_mean': result.cv_summary.mean_test_score if result.cv_summary else None,
                'cv_std': result.cv_summary.std_test_score if result.cv_summary else None,
                **result.test_metrics,
            })
        except Exception as e:
            results.append({'model': name, 'error': str(e)})

    return pd.DataFrame(results).set_index('model')


def walk_forward_backtest(
    pipeline: MLPipeline,
    data: dict[str, pd.DataFrame],
    initial_train: int = 500,
    step: int = 63,
    retrain: bool = True,
) -> pd.DataFrame:
    """
    Run walk-forward backtest of ML pipeline.

    Args:
        pipeline: Fitted MLPipeline
        data: Full dataset
        initial_train: Initial training window
        step: Step size between retraining
        retrain: Whether to retrain at each step

    Returns:
        DataFrame with predictions and actuals
    """
    # Build features for full dataset
    if pipeline.feature_pipeline is None or pipeline.model is None:
        raise ValueError("Pipeline feature_pipeline and model must not be None")
    X, y = pipeline.feature_pipeline._build_features(data), pipeline.feature_pipeline._build_target(data)
    X = pipeline.feature_pipeline.fill_missing(X)
    y = y.loc[X.index]
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    all_predictions = []

    for i in range(initial_train, len(X) - step, step):
        train_end = i
        test_end = min(i + step, len(X))

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        if retrain:
            # Retrain model
            model = create_model(pipeline.config.model_name, pipeline.config.task, pipeline.config.model_params)
            model.fit(X_train, y_train)
        else:
            model = pipeline.model

        if model is None:
            raise ValueError("Model object is None")
        y_pred = model.predict(X_test)


        pred_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
        }, index=X_test.index)

        all_predictions.append(pred_df)

    return pd.concat(all_predictions).sort_index()
