from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from models.prophet_model import estimate_ttf as prophet_estimate_ttf
from models.prophet_model import prepare_data as prophet_prepare_data
from utils.eval import evaluate_model as run_model_evaluation

def build_time_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=timestamps)
    df["hour"] = timestamps.hour
    df["sin_hour"] = np.sin(2 * np.pi * timestamps.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * timestamps.hour / 24)
    df["day_of_week"] = timestamps.dayofweek
    df["sin_dow"] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
    df["cos_dow"] = np.cos(2 * np.pi * timestamps.dayofweek / 7)
    hour_of_week = timestamps.dayofweek * 24 + timestamps.hour
    df["sin_week"] = np.sin(2 * np.pi * hour_of_week / 168)
    df["cos_week"] = np.cos(2 * np.pi * hour_of_week / 168)
    df["day_of_month"] = timestamps.day
    df["sin_month"] = np.sin(2 * np.pi * timestamps.day / 30.5)
    df["cos_month"] = np.cos(2 * np.pi * timestamps.day / 30.5)
    df["month"] = timestamps.month
    df["sin_year"] = np.sin(2 * np.pi * timestamps.month / 12)
    df["cos_year"] = np.cos(2 * np.pi * timestamps.month / 12)
    df["is_weekend"] = (timestamps.dayofweek >= 5).astype(int)
    return df

def compute_history_stats(
    df_prepared: pd.DataFrame,
    include_calendar_patterns: bool = False,
) -> Dict[str, Any]:
    y = df_prepared["y"].values
    stats: Dict[str, Any] = {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
    }

    if include_calendar_patterns:
        ds = pd.DatetimeIndex(df_prepared["ds"])
        hourly = pd.Series(y, index=ds).groupby(ds.hour).mean()
        stats["hourly_pattern"] = hourly.to_dict()
        dow = pd.Series(y, index=ds).groupby(ds.dayofweek).mean()
        stats["dow_pattern"] = dow.to_dict()

    recent_points = 7 * 24
    if len(y) > recent_points:
        recent_y = y[-recent_points:]
        x = np.arange(len(recent_y))
        slope, _ = np.polyfit(x, recent_y, 1)
        stats["trend_slope"] = float(slope)
        stats["trend_intercept"] = float(recent_y[-1])
    else:
        stats["trend_slope"] = 0.0
        stats["trend_intercept"] = float(y[-1])

    return stats

def project_trend(stats: Dict[str, Any], forecast_steps: int) -> np.ndarray:
    trend_forecast = np.array(
        [stats["trend_intercept"] + stats["trend_slope"] * (i + 1) for i in range(forecast_steps)],
        dtype=float,
    )
    damping = np.power(0.99, np.arange(forecast_steps))
    return stats["mean"] + (trend_forecast - stats["mean"]) * damping

def clip_forecast(
    forecast_values: np.ndarray,
    stats: Dict[str, Any],
    padding_ratio: float = 0.2,
) -> np.ndarray:
    value_range = stats["max"] - stats["min"]
    soft_min = stats["min"] - value_range * padding_ratio
    soft_max = stats["max"] + value_range * padding_ratio
    return np.clip(forecast_values, soft_min, soft_max)

def build_intervals(
    forecast_values: np.ndarray,
    future_timestamps: pd.DatetimeIndex,
    residual_std: float,
) -> Tuple[pd.Series, pd.Series]:
    forecast_steps = len(forecast_values)
    time_factor = np.sqrt(np.arange(1, forecast_steps + 1) / forecast_steps)
    delta = 1.96 * residual_std * (1 + time_factor)
    lower_series = pd.Series(forecast_values - delta, index=future_timestamps, name="lower")
    upper_series = pd.Series(forecast_values + delta, index=future_timestamps, name="upper")
    return lower_series, upper_series

def build_feature_frame(
    df_prepared: pd.DataFrame,
    feature_builder: Callable[[pd.DatetimeIndex], pd.DataFrame] = build_time_features,
) -> pd.DataFrame:
    ds = pd.DatetimeIndex(df_prepared["ds"])
    feature_df = feature_builder(ds)
    feature_df["time_idx"] = np.arange(len(feature_df)) / max(1, len(feature_df))
    feature_df["y"] = df_prepared["y"].values
    feature_df["ds"] = df_prepared["ds"].values
    return feature_df

def train_tree_forecaster(
    df_prepared: pd.DataFrame,
    forecast_steps: int,
    freq: str,
    config: Any,
    estimator_factory: Callable[[Any], Any],
    feature_builder: Callable[[pd.DatetimeIndex], pd.DataFrame] = build_time_features,
    include_calendar_patterns: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, Any]:
    stats = compute_history_stats(
        df_prepared,
        include_calendar_patterns=include_calendar_patterns,
    )
    ds = pd.DatetimeIndex(df_prepared["ds"])
    x_train = feature_builder(ds)
    x_train["time_idx"] = np.arange(len(x_train)) / max(1, len(x_train))
    y_train = df_prepared["y"].values

    model = estimator_factory(config)
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    residual_std = float(np.std(y_train - y_pred_train))
    residual_std = max(residual_std, stats["std"] * 0.1)

    freq_offset = pd.tseries.frequencies.to_offset(freq)
    last_timestamp = df_prepared["ds"].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + freq_offset,
        periods=forecast_steps,
        freq=freq,
    )

    x_future = feature_builder(future_timestamps)
    x_future["time_idx"] = 1.0 + np.arange(len(x_future)) / max(1, len(x_train))
    model_forecast = model.predict(x_future)

    trend_forecast = project_trend(stats, forecast_steps)
    horizon_weight = np.linspace(0, config.trend_blend_weight, forecast_steps)
    forecast_values = (1 - horizon_weight) * model_forecast + horizon_weight * trend_forecast
    forecast_values = clip_forecast(forecast_values, stats)

    forecast_series = pd.Series(forecast_values, index=future_timestamps, name="forecast")
    lower_series, upper_series = build_intervals(
        forecast_values,
        future_timestamps,
        residual_std,
    )

    return forecast_series, lower_series, upper_series, model

def evaluate_tree_forecaster(
    df_prepared: pd.DataFrame,
    config: Any,
    estimator_factory: Callable[[Any], Any],
    feature_builder: Callable[[pd.DatetimeIndex], pd.DataFrame] = build_time_features,
) -> Tuple[float, float, float, np.ndarray, float]:
    split_idx = int(len(df_prepared) * (1 - config.test_size))
    train_df = df_prepared.iloc[:split_idx]
    test_df = df_prepared.iloc[split_idx:]

    ds_train = pd.DatetimeIndex(train_df["ds"])
    x_train = feature_builder(ds_train)
    x_train["time_idx"] = np.arange(len(x_train)) / max(1, len(x_train))
    y_train = train_df["y"].values

    model = estimator_factory(config)
    model.fit(x_train, y_train)

    ds_test = pd.DatetimeIndex(test_df["ds"])
    x_test = feature_builder(ds_test)
    x_test["time_idx"] = 1.0 + np.arange(len(x_test)) / max(1, len(x_train))
    y_test = test_df["y"].values

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    residual_std = float(np.std(y_test - y_pred))

    return mae, rmse, mape, y_pred, residual_std


def prepare_data(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    value_col: str = "Value",
    freq: str = "1H",
    auto_resample: bool = True,
) -> Tuple[pd.DataFrame, bool]:
    return prophet_prepare_data(
        df,
        timestamp_col=timestamp_col,
        value_col=value_col,
        freq=freq,
        auto_resample=auto_resample,
    )

def make_model_evaluator(
    model_name: str,
    direct_evaluator: Callable[[pd.DataFrame, Optional[Any]], Tuple[Any, ...]],
    config_factory: Callable[[], Any],
) -> Callable[..., tuple]:
    def evaluate_model(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.1,
        n_changepoints: int = 6,
        changepoint_range: float = 0.9,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
    ) -> tuple:
        return run_model_evaluation(
            train_data=train_data,
            test_data=test_data,
            model_name=model_name,
            direct_evaluator=direct_evaluator,
            evaluator_config=config_factory(),
        )

    evaluate_model.__name__ = "evaluate_model"
    return evaluate_model

def estimate_ttf(
    forecast: pd.Series,
    df_clean: pd.DataFrame,
    freq: str = "1H",
) -> dict:
    return prophet_estimate_ttf(forecast=forecast, df_clean=df_clean, freq=freq)