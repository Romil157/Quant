"""ML feature engineering pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from quant.features import (
    atr,
    bollinger_bands,
    breakout_levels,
    ema,
    macd,
    momentum,
    rsi,
    sma,
    z_score,
)
from quant.features import (
    ewma_volatility as ewma_vol,
)
from quant.features import (
    garman_klass_volatility as garman_klass_vol,
)
from quant.features import (
    parkinson_volatility as parkinson_vol,
)
from quant.features import (
    realized_volatility as realized_vol,
)
from quant.features.statistical import half_life, hurst_exponent, rolling_correlation


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Price-based features
    use_returns: bool = True
    use_log_returns: bool = True
    return_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 21])

    # Technical indicators
    use_sma: bool = True
    sma_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    use_ema: bool = True
    ema_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    use_rsi: bool = True
    rsi_windows: list[int] = field(default_factory=lambda: [7, 14, 21])
    use_macd: bool = True
    macd_params: list[tuple] = field(default_factory=lambda: [(12, 26, 9)])
    use_bollinger: bool = True
    bb_windows: list[int] = field(default_factory=lambda: [20])
    use_atr: bool = True
    atr_windows: list[int] = field(default_factory=lambda: [14, 21])
    use_momentum: bool = True
    momentum_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 63, 126])
    use_zscore: bool = True
    zscore_windows: list[int] = field(default_factory=lambda: [20, 63])
    use_breakout: bool = True
    breakout_windows: list[int] = field(default_factory=lambda: [20, 55])

    # Volatility features
    use_realized_vol: bool = True
    vol_windows: list[int] = field(default_factory=lambda: [10, 20, 63])
    use_garman_klass: bool = True
    use_parkinson: bool = True
    use_ewma_vol: bool = True
    ewma_lambdas: list[float] = field(default_factory=lambda: [0.94, 0.97])

    # Statistical features
    use_hurst: bool = True
    hurst_windows: list[int] = field(default_factory=lambda: [100, 252])
    use_half_life: bool = True
    hl_windows: list[int] = field(default_factory=lambda: [20, 63])
    use_rolling_corr: bool = True
    corr_windows: list[int] = field(default_factory=lambda: [20, 63])

    # Microstructure (if volume data available)
    use_volume_features: bool = True
    volume_windows: list[int] = field(default_factory=lambda: [5, 20])

    # Target
    target_horizon: int = 5
    target_type: str = "returns"  # "returns", "direction", "volatility"

    # Preprocessing
    scaler: str = "robust"  # "standard", "robust", "minmax", "none"
    clip_outliers: float = 5.0  # Clip at N std devs
    fill_method: str = "ffill"  # "ffill", "bfill", "interpolate", "drop"


class FeaturePipeline:
    """Complete feature engineering pipeline for ML."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.scaler: BaseEstimator | None = None
        self._feature_names: list[str] = []
        self._fitted = False

    def fit_transform(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
        """Fit pipeline and transform data."""
        features = self._build_features(data)
        target = self._build_target(data)

        # Align
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]

        # Fill missing values
        features = self.fill_missing(features)
        target = target.loc[features.index]

        # Preprocess
        features = self._preprocess(features, fit=True)

        self._feature_names = list(features.columns)
        self._fitted = True

        return features, target

    def transform(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        if not self._fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")

        features = self._build_features(data)
        features = self.fill_missing(features)
        features = self._preprocess(features, fit=False)

        # Ensure same columns
        missing = set(self._feature_names) - set(features.columns)
        for col in missing:
            features[col] = np.nan

        return features[self._feature_names]

    def _build_features(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build all features from price data."""
        all_features = []

        for symbol, df in data.items():
            sym_features = pd.DataFrame(index=df.index)
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume'] if 'volume' in df.columns else None

            # Returns
            if self.config.use_returns:
                for h in self.config.return_horizons:
                    sym_features[f'{symbol}_ret_{h}'] = close.pct_change(h)

            if self.config.use_log_returns:
                for h in self.config.return_horizons:
                    sym_features[f'{symbol}_logret_{h}'] = np.log(close / close.shift(h))

            # SMA
            if self.config.use_sma:
                for w in self.config.sma_windows:
                    sma_vals = sma(close, w)
                    sym_features[f'{symbol}_sma_{w}'] = sma_vals
                    sym_features[f'{symbol}_sma_ratio_{w}'] = close / sma_vals - 1

            # EMA
            if self.config.use_ema:
                for w in self.config.ema_windows:
                    ema_vals = ema(close, w)
                    sym_features[f'{symbol}_ema_{w}'] = ema_vals
                    sym_features[f'{symbol}_ema_ratio_{w}'] = close / ema_vals - 1

            # RSI
            if self.config.use_rsi:
                for w in self.config.rsi_windows:
                    sym_features[f'{symbol}_rsi_{w}'] = rsi(close, w)

            # MACD
            if self.config.use_macd:
                for fast, slow, signal in self.config.macd_params:
                    macd_line, signal_line, hist = macd(close, fast, slow, signal)
                    sym_features[f'{symbol}_macd_{fast}_{slow}'] = macd_line
                    sym_features[f'{symbol}_macd_signal_{fast}_{slow}'] = signal_line
                    sym_features[f'{symbol}_macd_hist_{fast}_{slow}'] = hist

            # Bollinger Bands
            if self.config.use_bollinger:
                for w in self.config.bb_windows:
                    upper, middle, lower = bollinger_bands(close, w)
                    sym_features[f'{symbol}_bb_upper_{w}'] = upper
                    sym_features[f'{symbol}_bb_lower_{w}'] = lower
                    sym_features[f'{symbol}_bb_width_{w}'] = (upper - lower) / middle
                    sym_features[f'{symbol}_bb_pos_{w}'] = (close - lower) / (upper - lower)

            # ATR
            if self.config.use_atr:
                for w in self.config.atr_windows:
                    sym_features[f'{symbol}_atr_{w}'] = atr(high, low, close, w)
                    sym_features[f'{symbol}_atr_ratio_{w}'] = sym_features[f'{symbol}_atr_{w}'] / close

            # Momentum
            if self.config.use_momentum:
                for w in self.config.momentum_windows:
                    sym_features[f'{symbol}_mom_{w}'] = momentum(close, w)

            # Z-Score
            if self.config.use_zscore:
                for w in self.config.zscore_windows:
                    sym_features[f'{symbol}_zscore_{w}'] = z_score(close, w)

            # Breakout Levels
            if self.config.use_breakout:
                for w in self.config.breakout_windows:
                    upper, lower = breakout_levels(high, low, w)
                    sym_features[f'{symbol}_breakout_upper_{w}'] = upper
                    sym_features[f'{symbol}_breakout_lower_{w}'] = lower
                    sym_features[f'{symbol}_breakout_pos_{w}'] = (close - lower) / (upper - lower)

            # Volatility
            if self.config.use_realized_vol:
                for w in self.config.vol_windows:
                    sym_features[f'{symbol}_rv_{w}'] = realized_vol(close, w)

            if self.config.use_garman_klass:
                for w in self.config.vol_windows:
                    sym_features[f'{symbol}_gk_vol_{w}'] = garman_klass_vol(
                        df['open'] if 'open' in df.columns else close,
                        high, low, close, w
                    )

            if self.config.use_parkinson:
                for w in self.config.vol_windows:
                    sym_features[f'{symbol}_pk_vol_{w}'] = parkinson_vol(high, low, w)

            if self.config.use_ewma_vol:
                for lam in self.config.ewma_lambdas:
                    # Convert lambda to span: span = 1 / (1 - lambda)
                    span = int(1 / (1 - lam))
                    sym_features[f'{symbol}_ewma_vol_{lam}'] = ewma_vol(close, span)

            # Statistical
            if self.config.use_hurst:
                for w in self.config.hurst_windows:
                    sym_features[f'{symbol}_hurst_{w}'] = close.rolling(w).apply(
                        lambda x: hurst_exponent(x) if len(x) == w else np.nan, raw=True
                    )

            if self.config.use_half_life:
                for w in self.config.hl_windows:
                    sym_features[f'{symbol}_halflife_{w}'] = close.rolling(w).apply(
                        lambda x: half_life(x) if len(x) == w else np.nan, raw=True
                    )

            # Volume features
            if self.config.use_volume_features and volume is not None:
                for w in self.config.volume_windows:
                    sym_features[f'{symbol}_vol_sma_{w}'] = volume.rolling(w).mean()
                    sym_features[f'{symbol}_vol_ratio_{w}'] = volume / sym_features[f'{symbol}_vol_sma_{w}']

            all_features.append(sym_features)

        # Cross-asset features
        if len(data) > 1:
            all_features.append(self._build_cross_asset_features(data))

        # Combine
        features = pd.concat(all_features, axis=1)
        return features

    def _build_cross_asset_features(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build cross-asset features (correlations, ratios, etc.)."""
        symbols = list(data.keys())
        features = pd.DataFrame(index=list(data.values())[0].index)

        # Rolling correlations
        if self.config.use_rolling_corr:
            for w in self.config.corr_windows:
                for i, sym1 in enumerate(symbols):
                    for sym2 in symbols[i+1:]:
                        corr = rolling_correlation(data[sym1]['close'], data[sym2]['close'], w)
                        features[f'{sym1}_{sym2}_corr_{w}'] = corr

        # Relative strength (ratio)
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                ratio = data[sym1]['close'] / data[sym2]['close']
                features[f'{sym1}_{sym2}_ratio'] = ratio
                features[f'{sym1}_{sym2}_ratio_z'] = z_score(ratio, 63)

        return features

    def _build_target(self, data: dict[str, pd.DataFrame]) -> pd.Series:
        """Build target variable."""
        # Use first symbol as default target
        symbol = list(data.keys())[0]
        close = data[symbol]['close']
        h = self.config.target_horizon

        if self.config.target_type == "returns":
            target = close.pct_change(h).shift(-h)
            target.name = f'{symbol}_target_ret_{h}'
        elif self.config.target_type == "direction":
            target = (close.pct_change(h).shift(-h) > 0).astype(int)
            target.name = f'{symbol}_target_dir_{h}'
        elif self.config.target_type == "volatility":
            target = realized_vol(close, h).shift(-h)
            target.name = f'{symbol}_target_vol_{h}'
        else:
            raise ValueError(f"Unknown target_type: {self.config.target_type}")

        return target

    def _preprocess(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Preprocess features: clip, scale, fill."""
        # Clip outliers
        if self.config.clip_outliers > 0:
            for col in features.columns:
                std = features[col].std()
                mean = features[col].mean()
                if std > 0:
                    clip_val = self.config.clip_outliers * std
                    features[col] = features[col].clip(mean - clip_val, mean + clip_val)

        # Scale
        if self.config.scaler != "none":
            if fit:
                if self.config.scaler == "standard":
                    self.scaler = StandardScaler()
                elif self.config.scaler == "robust":
                    self.scaler = RobustScaler()
                elif self.config.scaler == "minmax":
                    self.scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown scaler: {self.config.scaler}")

                # Fit on non-NaN values
                valid_mask = features.notna().any(axis=1)
                if valid_mask.any():
                    # Use only rows with at least some valid data for fitting
                    fit_data = features[valid_mask].dropna(axis=1, how='all')
                    if len(fit_data) > 0:
                        self.scaler.fit(fit_data)
                        # Transform all features - need to align columns
                        features_scaled = pd.DataFrame(
                            index=features.index,
                            columns=features.columns,
                            dtype=float
                        )
                        # Only transform columns that were in fit_data
                        common_cols = [c for c in features.columns if c in fit_data.columns]
                        if common_cols:
                            scaled_values = self.scaler.transform(features[common_cols])
                            features_scaled[common_cols] = scaled_values
                        return features_scaled
            elif self.scaler is not None:
                features_scaled = pd.DataFrame(
                    index=features.index,
                    columns=features.columns,
                    dtype=float
                )
                common_cols = [c for c in features.columns if c in self.scaler.feature_names_in_]
                if common_cols:
                    scaled_values = self.scaler.transform(features[common_cols])
                    features_scaled[common_cols] = scaled_values
                return features_scaled

        return features

    def fill_missing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values."""
        if self.config.fill_method == "ffill":
            return features.ffill().bfill()
        elif self.config.fill_method == "bfill":
            return features.bfill().ffill()
        elif self.config.fill_method == "interpolate":
            return features.interpolate(method='time').ffill().bfill()
        elif self.config.fill_method == "drop":
            return features.dropna()
        return features

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names.copy()

    def get_feature_importance_df(self, importance: np.ndarray) -> pd.DataFrame:
        """Convert feature importance array to DataFrame."""
        return pd.DataFrame({
            'feature': self._feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


def create_default_pipeline() -> FeaturePipeline:
    """Create default feature pipeline."""
    return FeaturePipeline(FeatureConfig())


def create_minimal_pipeline() -> FeaturePipeline:
    """Create minimal feature pipeline for quick testing."""
    config = FeatureConfig(
        use_returns=True,
        return_horizons=[1, 5],
        use_sma=True,
        sma_windows=[20, 50],
        use_rsi=True,
        rsi_windows=[14],
        use_momentum=True,
        momentum_windows=[20],
        use_realized_vol=True,
        vol_windows=[20],
        target_horizon=5,
        target_type="returns",
        scaler="robust",
    )
    return FeaturePipeline(config)
