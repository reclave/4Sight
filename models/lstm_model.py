import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from models.forecast_utils import (
    build_intervals,
    clip_forecast,
    estimate_ttf as shared_estimate_ttf,
    compute_history_stats,
    make_model_evaluator,
    project_trend,
    prepare_data as shared_prepare_data,
)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

@dataclass
class LSTMConfig:
    freq: str = "1H"
    lookback: int = 72 # Hours of history per sample
    units: int = 64  # Hidden units
    layers: int = 2  # Number of stacked LSTM layers
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 8  # early-stopping patience
    random_state: int = 42
    test_size: float = 0.2

    # Blend LSTM with linear trend for stability
    trend_blend_weight: float = 0.3  # 0 = pure LSTM, 1 = pure trend

def _build_sequences(
    values: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    # Learning sequence of (samples, lookback, 1) -> (samples,)
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback : i])
        y.append(values[i])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)

def _build_model(config: LSTMConfig, lookback: int) -> "Sequential":
    # For stacked LSTM model
    model = Sequential()
    for i in range(config.layers):
        return_seq = i < config.layers - 1
        if i == 0:
            model.add(
                KerasLSTM(
                    config.units,
                    return_sequences=return_seq,
                    input_shape=(lookback, 1),
                )
            )
        else:
            model.add(KerasLSTM(config.units, return_sequences=return_seq))
        model.add(Dropout(config.dropout))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model

def train_lstm_direct(
    df_prepared: pd.DataFrame,
    forecast_steps: int,
    freq: str,
    config: Optional[LSTMConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, Any]:
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for the LSTM model"
        )

    config = config or LSTMConfig()
    stats = compute_history_stats(df_prepared)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_prepared["y"].values.reshape(-1, 1)).flatten()

    X_train, y_train = _build_sequences(scaled, config.lookback)

    # Build and train
    tf.random.set_seed(config.random_state)
    np.random.seed(config.random_state)

    model = _build_model(config, config.lookback)
    es = EarlyStopping(
        monitor="loss", patience=config.patience, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[es],
        verbose=0,
    )

    # Residual std (for confidence intervals) for train set
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    residual_std_scaled = float(np.std(y_train - y_pred_train))

    # Auto-regressive forecast
    last_window = scaled[-config.lookback :].copy()
    lstm_scaled: list[float] = []
    for _ in range(forecast_steps):
        inp = last_window.reshape(1, config.lookback, 1)
        pred = float(model.predict(inp, verbose=0).flatten()[0])
        lstm_scaled.append(pred)
        last_window = np.append(last_window[1:], pred)

    # Inverse-scale
    lstm_forecast = scaler.inverse_transform(
        np.array(lstm_scaled).reshape(-1, 1)
    ).flatten()

    # Residual std back to original scale
    residual_std = residual_std_scaled * (scaler.data_max_[0] - scaler.data_min_[0])
    residual_std = max(residual_std, stats["std"] * 0.1)

    # Linear trend extrapolation (stable baseline)
    trend_forecast = project_trend(stats, forecast_steps)

    # Blend model with trend
    horizon_weight = np.linspace(0, config.trend_blend_weight, forecast_steps)
    forecast_values = (1 - horizon_weight) * lstm_forecast + horizon_weight * trend_forecast

    # Soft bounds
    forecast_values = clip_forecast(forecast_values, stats)

    # Future timestamps
    freq_offset = to_offset(freq)
    last_timestamp = df_prepared["ds"].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + freq_offset, periods=forecast_steps, freq=freq
    )

    forecast_series = pd.Series(forecast_values, index=future_timestamps, name="forecast")

    # Confidence intervals
    lower_series, upper_series = build_intervals(
        forecast_values,
        future_timestamps,
        residual_std,
    )

    return forecast_series, lower_series, upper_series, model

def evaluate_lstm_direct(
    df_prepared: pd.DataFrame,
    config: Optional[LSTMConfig] = None,
) -> Tuple[float, float, float, np.ndarray, float]:
    # Evaluate model using train/test split
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for the LSTM model")

    config = config or LSTMConfig()

    split_idx = int(len(df_prepared) * (1 - config.test_size))
    train_df = df_prepared.iloc[:split_idx]
    test_df = df_prepared.iloc[split_idx:]

    # Scale on training data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df["y"].values.reshape(-1, 1)).flatten()

    X_train, y_train = _build_sequences(train_scaled, config.lookback)

    tf.random.set_seed(config.random_state)
    np.random.seed(config.random_state)

    model = _build_model(config, config.lookback)
    es = EarlyStopping(monitor="loss", patience=config.patience, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=config.epochs, batch_size=config.batch_size,
        callbacks=[es], verbose=0,
    )

    # Predict test set auto-regressively
    all_scaled = scaler.transform(df_prepared["y"].values.reshape(-1, 1)).flatten()
    window = all_scaled[split_idx - config.lookback : split_idx].copy()

    preds_scaled: list[float] = []
    for i in range(len(test_df)):
        inp = window.reshape(1, config.lookback, 1)
        pred = float(model.predict(inp, verbose=0).flatten()[0])
        preds_scaled.append(pred)
        # Use actual next value when available
        if split_idx + i < len(all_scaled):
            window = np.append(window[1:], all_scaled[split_idx + i])
        else:
            window = np.append(window[1:], pred)

    y_pred = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    y_test = test_df["y"].values

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    residual_std = float(np.std(y_test - y_pred))

    return mae, rmse, mape, y_pred, residual_std

prepare_data = shared_prepare_data
evaluate_model = make_model_evaluator(
    "LSTM",
    evaluate_lstm_direct,
    LSTMConfig,
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
    config = LSTMConfig(freq=freq)

    try:
        forecast_series, lower_series, upper_series, model = train_lstm_direct(
            df_prepared, forecast_steps, freq, config
        )

        if forecast_series is not None:
            print(f"\nLSTM training completed.")
            print(f"- Forecast range: {forecast_series.min():.2f} to {forecast_series.max():.2f}")
            print(f"- Uncertainty range: ±{(upper_series - lower_series).mean()/2:.2f}")

        return forecast_series, lower_series, upper_series, model

    except Exception as e:
        print(f"\nLSTM model training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

estimate_ttf = shared_estimate_ttf