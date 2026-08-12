"""Online/incremental learning for ML models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.preprocessing import StandardScaler

try:
    from river import compose, ensemble, linear_model, metrics, preprocessing, stream
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False
    linear_model = ensemble = preprocessing = compose = metrics = stream = None


@dataclass
class OnlineConfig:
    """Configuration for online learning."""
    model_type: str = "sgd"  # "sgd", "passive_aggressive", "river_linear", "river_ensemble"
    task: str = "regression"  # "regression" or "classification"
    learning_rate: float = 0.01
    alpha: float = 0.0001  # L2 regularization
    loss: str = ""  # SGD loss (auto-set based on task)
    penalty: str = "l2"
    max_iter: int = 1
    tol: float = 1e-3
    shuffle: bool = True
    random_state: int = 42
    n_estimators: int = 10  # For ensemble
    window_size: int = 1000  # For rolling retrain

    def __post_init__(self):
        if not self.loss:
            self.loss = "log_loss" if self.task == "classification" else "squared_error"


class OnlineLearner:
    """Online/incremental learning wrapper."""

    def __init__(self, config: OnlineConfig):
        self.config = config
        self.model: Any = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_samples_seen = 0
        self.feature_names: list[str] = []
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the online model."""
        if self.config.model_type == "sgd":
            if self.config.task == "regression":
                self.model = SGDRegressor(
                    loss=self.config.loss,
                    penalty=self.config.penalty,
                    alpha=self.config.alpha,
                    learning_rate='adaptive',
                    eta0=self.config.learning_rate,
                    max_iter=self.config.max_iter,
                    tol=self.config.tol,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state,
                    warm_start=True,
                )
            else:
                self.model = SGDClassifier(
                    loss=self.config.loss,
                    penalty=self.config.penalty,
                    alpha=self.config.alpha,
                    learning_rate='adaptive',
                    eta0=self.config.learning_rate,
                    max_iter=self.config.max_iter,
                    tol=self.config.tol,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state,
                    warm_start=True,
                )

        elif self.config.model_type == "passive_aggressive":
            if self.config.task == "regression":
                self.model = PassiveAggressiveRegressor(
                    C=1.0 / self.config.alpha,
                    max_iter=self.config.max_iter,
                    tol=self.config.tol,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state,
                    warm_start=True,
                )
            else:
                self.model = PassiveAggressiveClassifier(
                    C=1.0 / self.config.alpha,
                    max_iter=self.config.max_iter,
                    tol=self.config.tol,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state,
                    warm_start=True,
                )

        elif self.config.model_type == "river_linear":
            if self.config.task == "regression":
                self.model = linear_model.LinearRegression(
                    intercept_lr=self.config.learning_rate,
                    optimizer=None,
                )
            else:
                self.model = linear_model.LogisticRegression(
                    optimizer=None,
                )

        elif self.config.model_type == "river_ensemble":
            if self.config.task == "regression":
                self.model = ensemble.AdaGradRegressor(
                    n_estimators=self.config.n_estimators,
                )
            else:
                self.model = ensemble.AdaGradClassifier(
                    n_estimators=self.config.n_estimators,
                )

        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> OnlineLearner:
        """Incrementally fit the model on new data."""
        # Handle feature names
        if not self.feature_names:
            self.feature_names = list(X.columns)
        else:
            # Ensure same columns
            X = X[self.feature_names]

        # Scale features
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)

        y_values = y.values

        if self.config.model_type in ["river_linear", "river_ensemble"]:
            # River models expect dict-like input
            for xi, yi in zip(X_scaled, y_values, strict=False):
                x_dict = {f"f{i}": val for i, val in enumerate(xi)}
                if self.config.task == "regression":
                    self.model.learn_one(x_dict, yi)
                else:
                    self.model.learn_one(x_dict, int(yi))
        else:
            # sklearn incremental models
            if self.config.task == "classification":
                classes = np.unique(y_values)
                self.model.partial_fit(X_scaled, y_values, classes=classes)
            else:
                self.model.partial_fit(X_scaled, y_values)

        self.n_samples_seen += len(X)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call partial_fit first.")

        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)

        if self.config.model_type in ["river_linear", "river_ensemble"]:
            preds = []
            for xi in X_scaled:
                x_dict = {f"f{i}": val for i, val in enumerate(xi)}
                pred = self.model.predict_one(x_dict)
                preds.append(pred if pred is not None else 0)
            return np.array(preds)  # type: ignore
        else:
            assert self.model is not None
            return self.model.predict(X_scaled)  # type: ignore

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.config.task != "classification":
            raise ValueError("predict_proba only for classification")

        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)

        if self.config.model_type in ["river_linear", "river_ensemble"]:
            probs = []
            for xi in X_scaled:
                x_dict = {f"f{i}": val for i, val in enumerate(xi)}
                prob = self.model.predict_proba_one(x_dict)
                probs.append(prob if prob is not None else {0: 0.5, 1: 0.5})
            # Convert to array
            classes = sorted(probs[0].keys())
            return np.array([[p.get(c, 0) for c in classes] for p in probs])  # type: ignore
        else:
            assert self.model is not None
            return self.model.predict_proba(X_scaled)  # type: ignore

    def get_params(self) -> dict:
        return {
            'config': self.config.__dict__,
            'n_samples_seen': self.n_samples_seen,
            'feature_names': self.feature_names,
        }

    def reset(self) -> OnlineLearner:
        """Reset model to initial state."""
        self._init_model()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_samples_seen = 0
        self.feature_names = []
        return self


class RollingRetrainer:
    """Periodically retrain model on rolling window."""

    def __init__(
        self,
        base_model: BaseEstimator,
        window_size: int = 1000,
        retrain_freq: int = 100,
        scaler: bool = True,
    ):
        self.base_model = base_model
        self.window_size = window_size
        self.retrain_freq = retrain_freq
        self.use_scaler = scaler

        self.model: BaseEstimator | None = None
        self.scaler = StandardScaler() if scaler else None
        self.buffer_X: list[pd.DataFrame] = []
        self.buffer_y: list[pd.Series] = []
        self.samples_since_retrain = 0
        self.feature_names: list[str] = []
        self.is_fitted = False

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> RollingRetrainer:
        """Add data to buffer and retrain if needed."""
        # Store feature names
        if not self.feature_names:
            self.feature_names = list(X.columns)
        else:
            X = X[self.feature_names]

        # Add to buffer
        self.buffer_X.append(X)
        self.buffer_y.append(y)
        self.samples_since_retrain += len(X)

        # Trim buffer to window size
        total_samples = sum(len(df) for df in self.buffer_X)
        while total_samples > self.window_size:
            # Remove from oldest
            oldest = self.buffer_X[0]
            oldest_y = self.buffer_y[0]
            if len(oldest) <= total_samples - self.window_size:
                total_samples -= len(oldest)
                self.buffer_X.pop(0)
                self.buffer_y.pop(0)
            else:
                # Trim oldest
                trim = total_samples - self.window_size
                self.buffer_X[0] = oldest.iloc[trim:]
                self.buffer_y[0] = oldest_y.iloc[trim:]
                total_samples = self.window_size

        # Retrain if frequency reached
        if self.samples_since_retrain >= self.retrain_freq:
            self._retrain()
            self.samples_since_retrain = 0

        return self

    def _retrain(self) -> None:
        """Retrain model on current buffer."""
        if not self.buffer_X:
            return

        X_all = pd.concat(self.buffer_X)
        y_all = pd.concat(self.buffer_y)

        # Remove NaN
        valid = X_all.notna().all(axis=1) & y_all.notna()
        X_all = X_all[valid]
        y_all = y_all[valid]

        if len(X_all) < 10:
            return

        # Clone base model
        self.model = clone(self.base_model)

        # Scale
        if self.use_scaler:
            assert self.scaler is not None
            X_scaled = self.scaler.fit_transform(X_all)
        else:
            X_scaled = X_all

        self.model.fit(X_scaled, y_all)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using current model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        assert self.model is not None
        X = X[self.feature_names]

        if self.use_scaler:
            assert self.scaler is not None
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        return self.model.predict(X_scaled)  # type: ignore

    def force_retrain(self) -> RollingRetrainer:
        """Force immediate retrain."""
        self._retrain()
        self.samples_since_retrain = 0
        return self


class DriftDetector:
    """Concept drift detection for online learning."""

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.05,
        metric: str = "mse",
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.metric = metric

        self.errors: list[float] = []
        self.baseline_error: float | None = None
        self.drift_detected = False

    def update(self, y_true: float, y_pred: float) -> bool:
        """Update with new prediction error."""
        if self.metric == "mse":
            error = (y_true - y_pred) ** 2
        elif self.metric == "mae":
            error = abs(y_true - y_pred)
        else:
            error = (y_true - y_pred) ** 2

        self.errors.append(error)

        # Maintain window
        if len(self.errors) > self.window_size:
            self.errors.pop(0)

        # Check for drift after warmup
        if len(self.errors) >= self.window_size:
            current_error = np.mean(self.errors[-self.window_size//2:])
            reference_error = np.mean(self.errors[:self.window_size//2])

            if reference_error > 0:
                relative_change = (current_error - reference_error) / reference_error
                self.drift_detected = bool(relative_change > self.threshold)
            else:
                self.drift_detected = False

        return self.drift_detected

    def reset(self) -> None:
        """Reset drift detector."""
        self.errors = []
        self.baseline_error = None
        self.drift_detected = False


class OnlineEnsemble:
    """Ensemble of online learners with drift adaptation."""

    def __init__(
        self,
        models: list[OnlineLearner],
        drift_detector: DriftDetector | None = None,
        weights: list[float] | None = None,
    ):
        self.models = models
        self.drift_detector = drift_detector or DriftDetector()
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.predictions_history: list[dict] = []

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> OnlineEnsemble:
        """Fit all models."""
        for model in self.models:
            model.partial_fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted ensemble prediction."""
        preds = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            preds.append(pred * self.weights[i])

        return np.sum(preds, axis=0)

    def update_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update drift detector and potentially retrain."""
        for yt, yp in zip(y_true, y_pred, strict=False):
            if self.drift_detector.update(yt, yp):
                # Drift detected - retrain worst model
                self._replace_worst_model()
                break

    def _replace_worst_model(self) -> None:
        """Replace worst performing model with fresh one."""
        # For now, just reset the first model
        # In practice, would track individual model performance
        if self.models:
            self.models[0].reset()


def create_online_pipeline(
    feature_pipeline,
    online_config: OnlineConfig,
) -> tuple[Any, OnlineLearner]:
    """Create combined feature + online learning pipeline."""
    # This would combine feature extraction with online learning
    # For now, return the online learner separately
    learner = OnlineLearner(online_config)
    return feature_pipeline, learner
