import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from models.forecast_utils import (
    build_feature_frame,
    build_intervals,
    build_time_features,
    clip_forecast,
    estimate_ttf as shared_estimate_ttf,
    compute_history_stats,
    make_model_evaluator,
    project_trend,
    prepare_data as shared_prepare_data,
)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from models.optuna_optimizer import optimize_svm, is_optuna_available
    OPTUNA_AVAILABLE = is_optuna_available()
except ImportError:
    OPTUNA_AVAILABLE = False

@dataclass
class SVMConfig:
    freq: str = "1H"
    kernel: str = "rbf"
    c: float = 15.0
    epsilon: float = 0.08
    gamma: str = "scale"
    degree: int = 3
    test_size: float = 0.2
    random_state: int = 42
    max_train_samples: int = 5000

    # To blend SVM with linear trend for stability
    trend_blend_weight: float = 0.35  # 0 = pure SVM, 1 = pure trend

    # Optuna settings
    use_optuna: bool = False
    optuna_n_trials: int = 50
    optuna_timeout: Optional[int] = 300
    optuna_metric: str = "rmse"

    def model_params(self) -> dict:
        params = {
            "kernel": self.kernel,
            "C": self.c,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
        }
        if self.kernel == "poly":
            params["degree"] = self.degree
        return params

    def update_from_optuna(self, best_params: Dict[str, Any]) -> "SVMConfig":
        # For Optuna
        return SVMConfig(
            freq=self.freq,
            kernel=best_params.get("kernel", self.kernel),
            c=best_params.get("c", self.c),
            epsilon=best_params.get("epsilon", self.epsilon),
            gamma=best_params.get("gamma", self.gamma),
            degree=best_params.get("degree", self.degree),
            test_size=self.test_size,
            random_state=self.random_state,
            max_train_samples=best_params.get("max_train_samples", self.max_train_samples),
            trend_blend_weight=self.trend_blend_weight,
            use_optuna=False,
            optuna_n_trials=self.optuna_n_trials,
            optuna_timeout=self.optuna_timeout,
            optuna_metric=self.optuna_metric,
        )

def _build_svm_model(config: SVMConfig) -> TransformedTargetRegressor:
    # SVR pipeline with scaling

    regressor = make_pipeline(
        StandardScaler(),
        SVR(**config.model_params()),
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )

def _subsample_for_svm(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Evenly subsample long histories
    if len(x_values) <= max_samples:
        return x_values, y_values

    indices = np.linspace(0, len(x_values) - 1, max_samples, dtype=int)
    return x_values[indices], y_values[indices]

def train_svm_direct(
    df_prepared: pd.DataFrame,
    forecast_steps: int,
    freq: str,
    config: Optional[SVMConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, Any]:
    # Train SVM

    config = config or SVMConfig()
    stats = compute_history_stats(df_prepared)

    # Historical features
    ds = pd.DatetimeIndex(df_prepared["ds"])
    x_train = build_time_features(ds)
    x_train["time_idx"] = np.arange(len(x_train)) / len(x_train)
    y_train = df_prepared["y"].values

    # Keep SVM training bounded for long histories
    x_np, y_np = _subsample_for_svm(
        x_train.to_numpy(dtype=float),
        y_train.astype(float),
        max_samples=config.max_train_samples,
    )

    model = _build_svm_model(config)
    model.fit(x_np, y_np)

    y_pred_train = model.predict(x_np)
    residual_std = float(np.std(y_np - y_pred_train))
    residual_std = max(residual_std, stats["std"] * 0.1)

    # Future timestamps
    freq_offset = to_offset(freq)
    last_timestamp = df_prepared["ds"].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + freq_offset,
        periods=forecast_steps,
        freq=freq,
    )

    x_future = build_time_features(future_timestamps)
    x_future["time_idx"] = 1.0 + np.arange(len(x_future)) / len(x_train)

    svm_forecast = model.predict(x_future.to_numpy(dtype=float))

    # Stable trend baseline for long horizons
    trend_forecast = project_trend(stats, forecast_steps)

    # Blend model forecast with trend
    horizon_weight = np.linspace(0, config.trend_blend_weight, forecast_steps)
    forecast_values = (1 - horizon_weight) * svm_forecast + horizon_weight * trend_forecast

    # Soft value bounds
    forecast_values = clip_forecast(forecast_values, stats)

    forecast_series = pd.Series(forecast_values, index=future_timestamps, name="forecast")

    # Confidence intervals
    lower_series, upper_series = build_intervals(
        forecast_values,
        future_timestamps,
        residual_std,
    )

    return forecast_series, lower_series, upper_series, model

def evaluate_svm_direct(
    df_prepared: pd.DataFrame,
    config: Optional[SVMConfig] = None,
) -> Tuple[float, float, float, np.ndarray, float]:
    # Evaluate SVM using a time-based split.
    
    config = config or SVMConfig()

    split_idx = int(len(df_prepared) * (1 - config.test_size))
    train_df = df_prepared.iloc[:split_idx]
    test_df = df_prepared.iloc[split_idx:]

    ds_train = pd.DatetimeIndex(train_df["ds"])
    x_train = build_time_features(ds_train)
    x_train["time_idx"] = np.arange(len(x_train)) / len(x_train)
    y_train = train_df["y"].values

    x_train_np, y_train_np = _subsample_for_svm(
        x_train.to_numpy(dtype=float),
        y_train.astype(float),
        max_samples=config.max_train_samples,
    )

    model = _build_svm_model(config)
    model.fit(x_train_np, y_train_np)

    ds_test = pd.DatetimeIndex(test_df["ds"])
    x_test = build_time_features(ds_test)
    x_test["time_idx"] = 1.0 + np.arange(len(x_test)) / max(1, len(train_df))
    y_test = test_df["y"].values

    y_pred = model.predict(x_test.to_numpy(dtype=float))

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    residual_std = float(np.std(y_test - y_pred))

    return mae, rmse, mape, y_pred, residual_std

prepare_data = shared_prepare_data
evaluate_model = make_model_evaluator(
    "SVM",
    evaluate_svm_direct,
    SVMConfig,
)

def train_model(
    df_prepared: pd.DataFrame,
    forecast_steps: int,
    freq: str,
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.1,
    interval_width: float = 0.90,
    daily_seasonality: bool = "auto",
    weekly_seasonality: bool = "auto",
    n_changepoints: int = 6,
    changepoint_range: float = 0.9,
    use_optuna: bool = False,
    optuna_n_trials: int = 50,
    optuna_timeout: Optional[int] = 300,
) -> Tuple[pd.Series, pd.Series, pd.Series, Any]:
    # Train SVM 

    config = SVMConfig(
        freq=freq,
        use_optuna=use_optuna,
        optuna_n_trials=optuna_n_trials,
        optuna_timeout=optuna_timeout,
    )

    try:
        # Run Optuna optimization if enabled
        if use_optuna and OPTUNA_AVAILABLE:
            print("\nRunning hyperparameter optimization...")
            feature_df = build_feature_frame(df_prepared, build_time_features)

            opt_result = optimize_svm(
                feature_df=feature_df,
                n_trials=optuna_n_trials,
                timeout=optuna_timeout,
                metric=config.optuna_metric,
                test_size=config.test_size,
                random_state=config.random_state,
                show_progress=True,
                verbose=True,
            )
            config = config.update_from_optuna(opt_result.best_params)
            print(" - Using optimized parameters")
        elif use_optuna and not OPTUNA_AVAILABLE:
            print("\nOptuna not available. Using default SVM parameters.")

        forecast_series, lower_series, upper_series, model = train_svm_direct(
            df_prepared, forecast_steps, freq, config
        )

        if forecast_series is not None:
            print("\nSVM training completed.")
            print(f" - Forecast range: {forecast_series.min():.2f} to {forecast_series.max():.2f}")
            print(f" - Uncertainty range: ±{(upper_series - lower_series).mean()/2:.2f}")

        return forecast_series, lower_series, upper_series, model

    except Exception as e:
        print(f"\nSVM model training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

estimate_ttf = shared_estimate_ttf