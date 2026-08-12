"""ML model wrappers and utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


@dataclass
class ModelConfig:
    """Configuration for ML model."""
    name: str
    model_type: str  # "regression" or "classification"
    params: dict[str, Any] = field(default_factory=dict)
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    use_scaling: bool = True
    feature_selection: bool = False
    n_features: int = 50


class BaseMLModel(ABC):
    """Base class for ML models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: BaseEstimator | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying sklearn model."""
        pass

    def _get_pipeline(self) -> Pipeline:
        """Get preprocessing + model pipeline."""
        steps = []
        if self.config.use_scaling:
            self.scaler = StandardScaler()
            steps.append(('scaler', self.scaler))
        steps.append(('model', self._create_model()))
        return Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> BaseMLModel:
        """Fit the model."""
        self.feature_names = list(X.columns)
        self.model = self._get_pipeline()
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        # Ensure same columns
        X = X[self.feature_names]
        return self.model.predict(X)  # type: ignore

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support predict_proba")
        X = X[self.feature_names]
        return self.model.predict_proba(X)  # type: ignore

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        model_step = self.model.named_steps['model']


        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
        elif hasattr(model_step, 'coef_'):
            importances = np.abs(model_step.coef_).flatten()
        else:
            return pd.DataFrame({'feature': self.feature_names, 'importance': 0})

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

    def get_params(self) -> dict:
        return self.config.params

    def set_params(self, **params) -> BaseMLModel:
        self.config.params.update(params)
        return self


# Regression Models

class LinearRegressionModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return LinearRegression(**self.config.params)


class RidgeModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return Ridge(**self.config.params)


class LassoModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return Lasso(**self.config.params)


class ElasticNetModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return ElasticNet(**self.config.params)


class RandomForestRegModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return RandomForestRegressor(**self.config.params)


class GradientBoostingRegModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return GradientBoostingRegressor(**self.config.params)


class ExtraTreesRegModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return ExtraTreesRegressor(**self.config.params)


class HistGradientBoostingRegModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return HistGradientBoostingRegressor(**self.config.params)


class SVRModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return SVR(**self.config.params)


class KNNRegModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return KNeighborsRegressor(**self.config.params)


class MLPRegModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return MLPRegressor(**self.config.params)


# Classification Models

class LogisticRegressionModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return LogisticRegression(**self.config.params)


class RidgeClassifierModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return RidgeClassifier(**self.config.params)


class RandomForestClsModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return RandomForestClassifier(**self.config.params)


class GradientBoostingClsModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return GradientBoostingClassifier(**self.config.params)


class ExtraTreesClsModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return ExtraTreesClassifier(**self.config.params)


class HistGradientBoostingClsModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return HistGradientBoostingClassifier(**self.config.params)


class SVCModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return SVC(probability=True, **self.config.params)


class KNNClsModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return KNeighborsClassifier(**self.config.params)


class MLPClsModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        return MLPClassifier(**self.config.params)


# Gradient Boosting Libraries

if HAS_XGBOOST:
    class XGBoostRegModel(BaseMLModel):
        def _create_model(self) -> BaseEstimator:
            return xgb.XGBRegressor(**self.config.params)

    class XGBoostClsModel(BaseMLModel):
        def _create_model(self) -> BaseEstimator:
            return xgb.XGBClassifier(**self.config.params)

if HAS_LIGHTGBM:
    class LightGBMRegModel(BaseMLModel):
        def _create_model(self) -> BaseEstimator:
            return lgb.LGBMRegressor(**self.config.params)

    class LightGBMClsModel(BaseMLModel):
        def _create_model(self) -> BaseEstimator:
            return lgb.LGBMClassifier(**self.config.params)

if HAS_CATBOOST:
    class CatBoostRegModel(BaseMLModel):
        def _create_model(self) -> BaseEstimator:
            return cb.CatBoostRegressor(verbose=False, **self.config.params)

    class CatBoostClsModel(BaseMLModel):
        def _create_model(self) -> BaseEstimator:
            return cb.CatBoostClassifier(verbose=False, **self.config.params)


# Model Registry

REGRESSION_MODELS = {
    'linear': LinearRegressionModel,
    'ridge': RidgeModel,
    'lasso': LassoModel,
    'elasticnet': ElasticNetModel,
    'rf': RandomForestRegModel,
    'gbr': GradientBoostingRegModel,
    'et': ExtraTreesRegModel,
    'hgbr': HistGradientBoostingRegModel,
    'svr': SVRModel,
    'knn': KNNRegModel,
    'mlp': MLPRegModel,
}

CLASSIFICATION_MODELS = {
    'logistic': LogisticRegressionModel,
    'ridge_cls': RidgeClassifierModel,
    'rf': RandomForestClsModel,
    'gbc': GradientBoostingClsModel,
    'et': ExtraTreesClsModel,
    'hgbc': HistGradientBoostingClsModel,
    'svc': SVCModel,
    'knn': KNNClsModel,
    'mlp': MLPClsModel,
}

if HAS_XGBOOST:
    REGRESSION_MODELS['xgb'] = XGBoostRegModel
    CLASSIFICATION_MODELS['xgb'] = XGBoostClsModel

if HAS_LIGHTGBM:
    REGRESSION_MODELS['lgb'] = LightGBMRegModel
    CLASSIFICATION_MODELS['lgb'] = LightGBMClsModel

if HAS_CATBOOST:
    REGRESSION_MODELS['cat'] = CatBoostRegModel
    CLASSIFICATION_MODELS['cat'] = CatBoostClsModel


def get_model_class(name: str, task: str) -> type:
    """Get model class by name and task."""
    if task == "regression":
        if name not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model: {name}. Available: {list(REGRESSION_MODELS.keys())}")
        return REGRESSION_MODELS[name]
    else:
        if name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model: {name}. Available: {list(CLASSIFICATION_MODELS.keys())}")
        return CLASSIFICATION_MODELS[name]


def create_model(name: str, task: str, params: dict | None = None) -> BaseMLModel:
    """Create model instance."""
    model_class = get_model_class(name, task)
    config = ModelConfig(name=name, model_type=task, params=params or {})
    return model_class(config)  # type: ignore


def tune_hyperparameters(
    model: BaseMLModel,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict[str, Any],
    cv: int = 3,
    scoring: str = 'neg_mean_squared_error',
    n_iter: int = 20,
    random_state: int = 42,
    method: str = 'random',  # 'grid' or 'random'
) -> BaseMLModel:
    """Tune hyperparameters using cross-validation."""
    pipeline = model._get_pipeline()

    # Prefix params for pipeline
    prefixed_grid = {f'model__{k}': v for k, v in param_grid.items()}

    if method == 'grid':
        search = GridSearchCV(
            pipeline, prefixed_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=0
        )
    else:
        search = RandomizedSearchCV(
            pipeline, prefixed_grid, cv=cv, scoring=scoring,
            n_iter=n_iter, n_jobs=-1, verbose=0, random_state=random_state
        )

    search.fit(X, y)

    # Update model with best params
    best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    model.config.params.update(best_params)
    model.model = search.best_estimator_
    model.is_fitted = True

    return model


def ensemble_predict(
    models: list[BaseMLModel],
    X: pd.DataFrame,
    weights: list[float] | None = None,
    method: str = 'mean',
) -> np.ndarray:
    """Ensemble predictions from multiple models."""
    preds_list: list[np.ndarray] = []
    for model in models:
        preds_list.append(model.predict(X))

    preds = np.column_stack(preds_list)

    weights_arr = np.ones(len(models)) / len(models) if weights is None else np.array(weights)

    if method == 'mean':
        return np.average(preds, axis=1, weights=weights_arr)  # type: ignore
    elif method == 'median':
        return np.median(preds, axis=1)  # type: ignore
    elif method == 'weighted_median':
        # Approximate weighted median
        sorted_idx = np.argsort(preds, axis=1)
        cumsum = np.cumsum(weights_arr[sorted_idx], axis=1)
        median_idx = np.argmax(cumsum >= 0.5, axis=1)
        return preds[np.arange(len(preds)), sorted_idx[np.arange(len(preds)), median_idx]]  # type: ignore
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def stack_models(
    base_models: list[BaseMLModel],
    meta_model: BaseMLModel,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> BaseMLModel:
    """Stack models using cross-validated predictions as meta-features."""
    from sklearn.model_selection import KFold

    n_samples = len(X)
    meta_features = np.zeros((n_samples, len(base_models)))

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for i, model in enumerate(base_models):
        fold_preds = np.zeros(n_samples)
        for train_idx, val_idx in kf.split(X):
            fold_model = clone(model._get_pipeline())
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_preds[val_idx] = fold_model.predict(X.iloc[val_idx])
        meta_features[:, i] = fold_preds

    # Train meta-model on meta-features
    meta_model.fit(pd.DataFrame(meta_features, index=X.index), y)

    # Refit base models on full data
    for model in base_models:
        model.fit(X, y)

    # Create stacked predictor
    class StackedModel(BaseMLModel):
        def __init__(self, base_models: list[BaseMLModel], meta_model: BaseMLModel):
            super().__init__(ModelConfig(name="stacked", model_type=meta_model.config.model_type))
            self.base_models = base_models
            self.meta_model = meta_model
            self.is_fitted = True
            self.feature_names = meta_model.feature_names

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            base_preds = np.column_stack([m.predict(X) for m in self.base_models])
            return self.meta_model.predict(pd.DataFrame(base_preds, index=X.index))

        def _create_model(self) -> BaseEstimator:
            return self.meta_model._create_model()

    return StackedModel(base_models, meta_model)
