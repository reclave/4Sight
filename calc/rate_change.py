import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import timedelta

def calculate_rate_change(df: pd.DataFrame, window_days: int, column: str = 'Value',
                              timestamp_col: str = 'Timestamp', convert_to: str = 'per_day'):
    end_time = df[timestamp_col].max()
    start_time = end_time - timedelta(days=window_days)

    df_window = df[(df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)].copy()

    if df_window.empty or len(df_window) < 10:
        raise ValueError(f"Not enough data for {window_days}-day window.")

    df_window['time_seconds'] = (df_window[timestamp_col] - start_time).dt.total_seconds()
    X = df_window['time_seconds'].values.reshape(-1, 1)
    y = df_window[column].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0][0]  # per second

    if convert_to == 'per_day':
        slope *= 86400
    elif convert_to == 'per_hour':
        slope *= 3600
    elif convert_to == 'per_second':
        pass
    else:
        raise ValueError("convert_to must be 'per_day', 'per_hour', or 'per_second'")

    return slope, df_window

def compare_multiple_periods(df: pd.DataFrame, periods: list = [365, 180, 90, 60, 30, 14], convert_to: str = 'per_day'):
    # Compare ROC for different periods with unit conversion
    slopes = {}
    for days in periods:
        try:
            slope, _ = calculate_rate_change(df, window_days=days, convert_to=convert_to)
            slopes[f"{days}d"] = slope
        except Exception as e:
            slopes[f"{days}d"] = f"Warning: {str(e)}"
    return slopes