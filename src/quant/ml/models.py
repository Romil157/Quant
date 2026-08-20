"""ML model wrappers and utilities."""
from __future__ import annotations

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
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
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
    model_type: str = "regression"  # "regression" or "classification"
    params: dict[str, Any] = field(default_factory=dict)
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    use_scaling: bool = True
    feature_selection: bool = False
    n_features: int = 50


class BaseMLModel:
    """Base class for ML model wrappers."""

    estimator_cls: type[BaseEstimator] | None = None

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Pipeline | BaseEstimator | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    def _create_model(self) -> BaseEstimator:
        """Create the underlying sklearn estimator."""
        if self.estimator_cls is None:
            raise NotImplementedError("Model class must define estimator_cls or _create_model()")
        return self.estimator_cls(**self.config.params)

    def _get_pipeline(self) -> Pipeline:
        """Get preprocessing + estimator pipeline."""
        steps: list[tuple[str, Any]] = []
        if self.config.use_scaling:
            self.scaler = StandardScaler()
            steps.append(("scaler", self.scaler))
        steps.append(("model", self._create_model()))
        return Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> BaseMLModel:
        """Fit preprocessing pipeline and model on feature matrix and target."""
        self.feature_names = list(X.columns)
        self.model = self._get_pipeline()
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions from feature matrix."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X[self.feature_names])  # type: ignore

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model does not support predict_proba")
        return self.model.predict_proba(X[self.feature_names])  # type: ignore

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importances or linear coefficients."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")

        estimator = self.model.named_steps["model"] if isinstance(self.model, Pipeline) else self.model
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_).flatten()
        else:
            return pd.DataFrame({"feature": self.feature_names, "importance": 0.0})

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    def get_params(self) -> dict:
        return self.config.params

    def set_params(self, **params) -> BaseMLModel:
        self.config.params.update(params)
        return self


# Regression Models
class LinearRegressionModel(BaseMLModel):
    estimator_cls = LinearRegression


class RidgeModel(BaseMLModel):
    estimator_cls = Ridge


class LassoModel(BaseMLModel):
    estimator_cls = Lasso


class ElasticNetModel(BaseMLModel):
    estimator_cls = ElasticNet


class RandomForestRegModel(BaseMLModel):
    estimator_cls = RandomForestRegressor


class GradientBoostingRegModel(BaseMLModel):
    estimator_cls = GradientBoostingRegressor


class ExtraTreesRegModel(BaseMLModel):
    estimator_cls = ExtraTreesRegressor


class HistGradientBoostingRegModel(BaseMLModel):
    estimator_cls = HistGradientBoostingRegressor


class SVRModel(BaseMLModel):
    estimator_cls = SVR


class KNNRegModel(BaseMLModel):
    estimator_cls = KNeighborsRegressor


class MLPRegModel(BaseMLModel):
    estimator_cls = MLPRegressor


# Classification Models
class LogisticRegressionModel(BaseMLModel):
    estimator_cls = LogisticRegression


class RidgeClassifierModel(BaseMLModel):
    estimator_cls = RidgeClassifier


class RandomForestClsModel(BaseMLModel):
    estimator_cls = RandomForestClassifier


class GradientBoostingClsModel(BaseMLModel):
    estimator_cls = GradientBoostingClassifier


class ExtraTreesClsModel(BaseMLModel):
    estimator_cls = ExtraTreesClassifier


class HistGradientBoostingClsModel(BaseMLModel):
    estimator_cls = HistGradientBoostingClassifier


class SVCModel(BaseMLModel):
    def _create_model(self) -> BaseEstimator:
        params = dict(self.config.params)
        params.setdefault("probability", True)
        return SVC(**params)


class KNNClsModel(BaseMLModel):
    estimator_cls = KNeighborsClassifier


class MLPClsModel(BaseMLModel):
    estimator_cls = MLPClassifier


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


REGRESSION_MODELS: dict[str, type[BaseMLModel]] = {
    "linear": LinearRegressionModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "elasticnet": ElasticNetModel,
    "rf": RandomForestRegModel,
    "gbr": GradientBoostingRegModel,
    "et": ExtraTreesRegModel,
    "hgbr": HistGradientBoostingRegModel,
    "svr": SVRModel,
    "knn": KNNRegModel,
    "mlp": MLPRegModel,
}

CLASSIFICATION_MODELS: dict[str, type[BaseMLModel]] = {
    "logistic": LogisticRegressionModel,
    "ridge_cls": RidgeClassifierModel,
    "rf": RandomForestClsModel,
    "gbc": GradientBoostingClsModel,
    "et": ExtraTreesClsModel,
    "hgbc": HistGradientBoostingClsModel,
    "svc": SVCModel,
    "knn": KNNClsModel,
    "mlp": MLPClsModel,
}

if HAS_XGBOOST:
    REGRESSION_MODELS["xgb"] = XGBoostRegModel
    CLASSIFICATION_MODELS["xgb"] = XGBoostClsModel

if HAS_LIGHTGBM:
    REGRESSION_MODELS["lgb"] = LightGBMRegModel
    CLASSIFICATION_MODELS["lgb"] = LightGBMClsModel

if HAS_CATBOOST:
    REGRESSION_MODELS["cat"] = CatBoostRegModel
    CLASSIFICATION_MODELS["cat"] = CatBoostClsModel


def get_model_class(name: str, task: str) -> type[BaseMLModel]:
    """Get model class by name and task."""
    registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
    if name not in registry:
        raise ValueError(f"Unknown {task} model: {name}. Available: {sorted(registry.keys())}")
    return registry[name]


def create_model(name: str, task: str, params: dict | None = None) -> BaseMLModel:
    """Create model instance with configuration."""
    model_class = get_model_class(name, task)
    config = ModelConfig(name=name, model_type=task, params=params or {})
    return model_class(config)


def tune_hyperparameters(
    model: BaseMLModel,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict[str, Any],
    cv: int = 3,
    scoring: str = "neg_mean_squared_error",
    n_iter: int = 20,
    random_state: int = 42,
    method: str = "random",
) -> BaseMLModel:
    """Tune hyperparameters using cross-validation."""
    pipeline = model._get_pipeline()
    prefixed_grid = {f"model__{k}": v for k, v in param_grid.items()}

    if method == "grid":
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
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    model.config.params.update(best_params)
    model.model = search.best_estimator_
    model.is_fitted = True
    return model


def ensemble_predict(
    models: list[BaseMLModel],
    X: pd.DataFrame,
    weights: list[float] | None = None,
    method: str = "mean",
) -> np.ndarray:
    """Ensemble predictions from multiple models."""
    preds = np.column_stack([model.predict(X) for model in models])
    weights_arr = np.ones(len(models)) / len(models) if weights is None else np.array(weights)

    if method == "mean":
        return np.asarray(np.average(preds, axis=1, weights=weights_arr))
    elif method == "median":
        return np.asarray(np.median(preds, axis=1))
    elif method == "weighted_median":
        sorted_idx = np.argsort(preds, axis=1)
        cumsum = np.cumsum(weights_arr[sorted_idx], axis=1)
        median_idx = np.argmax(cumsum >= 0.5, axis=1)
        return np.asarray(preds[np.arange(len(preds)), sorted_idx[np.arange(len(preds)), median_idx]])
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

    meta_model.fit(pd.DataFrame(meta_features, index=X.index), y)

    for model in base_models:
        model.fit(X, y)

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
