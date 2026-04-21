import pandas as pd
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any
from prophet import Prophet
from calc.sigma import calculate_sigma_median
from calc.rate_change import calculate_rate_change
from utils.eval import performance_score, evaluate_model, get_model_diagnostics, cross_validate_model
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from models.optuna_optimizer import optimize_prophet, is_optuna_available, OptimizationResult
    OPTUNA_AVAILABLE = is_optuna_available()
except ImportError:
    OPTUNA_AVAILABLE = False
    OptimizationResult = None

def prepare_data(df: pd.DataFrame, timestamp_col: str = 'Timestamp', value_col: str = 'Value',
                 freq: str = '1H', auto_resample: bool = True):
    df_copy = df.copy()
    
    if auto_resample:
        # Resample to regular intervals and forward fill missing values
        df_resampled = df_copy.set_index(timestamp_col)[value_col].resample(freq).mean().ffill() # Forward fill missing values
        df_prepared = df_resampled.reset_index()
        resampled = True
    else:
        # Use data as-is, just rename columns
        df_prepared = df_copy[[timestamp_col, value_col]].copy()
        resampled = False
    
    # Prophet requires 'ds' and 'y' column names
    df_prepared.columns = ['ds', 'y']
    
    # Remove any missing values
    df_prepared = df_prepared.dropna()
    
    if not df_prepared.empty:
        start_time = df_prepared['ds'].min()
        end_time = df_prepared['ds'].max()
        print(f"\nPrepared {len(df_prepared)} data points for forecasting from {start_time} to {end_time}")
    else:
        print("\nPrepared 0 data points for forecasting (empty)")
    return df_prepared, resampled


def train_model(df_prepared: pd.DataFrame,
                  forecast_steps: int,
                  freq: str,
                  seasonality_mode: str = 'additive',
                  changepoint_prior_scale: float = 0.1,
                  interval_width: float = 0.90,
                  daily_seasonality: bool = 'auto',
                  weekly_seasonality: bool = 'auto',
                  n_changepoints: int = 6,
                  changepoint_range: float = 0.9,
                  use_optuna: bool = False,
                  optuna_n_trials: int = 30,
                  optuna_timeout: Optional[int] = 300) -> tuple:
    """
    Train model and return forecast with uncertainty intervals.
    
    Args:
        use_optuna: Enable Optuna hyperparameter optimization
        optuna_n_trials: Number of Optuna trials
        optuna_timeout: Optuna timeout in seconds
    """
    try:
        if use_optuna and OPTUNA_AVAILABLE:
            print("\nRunning hyperparameter optimization...")
            opt_result = optimize_prophet(
                df_prepared=df_prepared,
                n_trials=optuna_n_trials,
                timeout=optuna_timeout,
                metric="rmse",
                test_size=0.2,
                show_progress=True,
                verbose=True,
            )
            # Use optimized parameters
            best_params = opt_result.best_params
            seasonality_mode = best_params.get("seasonality_mode", seasonality_mode)
            changepoint_prior_scale = best_params.get("changepoint_prior_scale", changepoint_prior_scale)
            n_changepoints = best_params.get("n_changepoints", n_changepoints)
            changepoint_range = best_params.get("changepoint_range", changepoint_range)
            daily_seasonality = best_params.get("daily_seasonality", daily_seasonality)
            weekly_seasonality = best_params.get("weekly_seasonality", weekly_seasonality)
            seasonality_prior_scale = best_params.get("seasonality_prior_scale", 10.0)
            print(f" - Using optimized parameters")
        elif use_optuna and not OPTUNA_AVAILABLE:
            print("\nOptuna not available. Using default Prophet parameters.")
            seasonality_prior_scale = 10.0
        else:
            seasonality_prior_scale = 10.0
        
        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=False,
            uncertainty_samples=200,   
            mcmc_samples=0,           
            interval_width=interval_width
        )

        # Custom seasonality for monthly patterns
        min_points_for_monthly = max(500, int(len(df_prepared) * 0.3))
        if len(df_prepared) > min_points_for_monthly:
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=8,
                prior_scale=10
            )
        
        print("\nTraining model...")
        start_time = time.time()
        model.fit(df_prepared)
        elapsed = time.time() - start_time
        print(f"\nModel training completed in {elapsed:.2f} seconds.")
        
        # Future dataframe
        future = model.make_future_dataframe(periods=forecast_steps, freq=freq)
        
        print(f"\nGenerating {forecast_steps} step forecast...")
        forecast = model.predict(future)
        
        # Extract future values
        forecast_future = forecast.tail(forecast_steps)
        
        # Time index for forecast
        forecast_index = pd.date_range(
            start=df_prepared['ds'].iloc[-1] + pd.Timedelta(freq), 
            periods=forecast_steps, 
            freq=freq
        )
        
        # Extract future values as series with index
        forecast_mean = pd.Series(
            forecast_future['yhat'].values, 
            index=forecast_index, 
            name='forecast'
        )
        lower_bound = pd.Series(
            forecast_future['yhat_lower'].values, 
            index=forecast_index, 
            name='lower'
        )
        upper_bound = pd.Series(
            forecast_future['yhat_upper'].values, 
            index=forecast_index, 
            name='upper'
        )
        
        print(f"\nModel training completed.")
        print(f" - Forecast range: {forecast_mean.min():.2f} to {forecast_mean.max():.2f}")
        print(f" - Uncertainty range: ±{(upper_bound - lower_bound).mean()/2:.2f}")
        
        return forecast_mean, lower_bound, upper_bound, model
        
    except Exception as e:
        print(f"\nModel training failed: {e}")
        return None, None, None, None

def estimate_ttf(forecast: pd.Series,
                                df_clean: pd.DataFrame,
                                # threshold_high: float = 83,
                                # threshold_low: float = 32,
                                freq: str = '1H') -> dict:
    # Estimate time to first breach (TTF) for all thresholds
   
    timestamps = forecast.index
    
    # Calculate sigma statistics from historical data
    _, sigma_stats = calculate_sigma_median(df_clean)
    
    all_thresholds = {
        #"TTF High Alarm": threshold_high,
        # "TTF Low Alarm": threshold_low,
        "Positive 1σ": sigma_stats["1σ+"],
        "Positive 2σ": sigma_stats["2σ+"],
        "Positive 3σ": sigma_stats["3σ+"],
        "Negative 1σ": sigma_stats["1σ-"],
        "Negative 2σ": sigma_stats["2σ-"],
        "Negative 3σ": sigma_stats["3σ-"],
    }
    
    ttf_result = {}
    
    # Find first breach time for each threshold
    for label, threshold in all_thresholds.items():
        if "Pos" in label or "High" in label:
            breach_mask = forecast >= threshold
        else:
            breach_mask = forecast <= threshold
        
        # Find first breach timestamp
        breach_times = timestamps[breach_mask]
        ttf_result[label] = breach_times[0] if not breach_times.empty else None
    
    print("\nTime to Failure (TTF) Predictions")
    for label, breach_time in ttf_result.items():
        if breach_time:
            time_to_breach = breach_time - pd.Timestamp.now()
            print(f"   - {label}: {breach_time.strftime('%Y-%m-%d %H:%M')} ({time_to_breach})")
        else:
            print(f"   - {label}: None")
    
    return ttf_result

def estimate_ttf_roc(df: pd.DataFrame, timestamp_col: str = 'Timestamp', value_col: str = 'Value', 
                     periods: list = [14, 30, 60]) -> dict:
    # Estimate TTF based on Rate of Change (ROC) for different periods
    # Calculates when the linear trend will hit 5 sigma threshold without forecast horizon limits.
    # Statistical inference not used in the application (helper).

    print("\nCalculating TTF ROC (Rate of Change) predictions...")
    
    # Calculate sigma thresholds
    df_marked, thresholds = calculate_sigma_median(df, column=value_col)
    five_sigma_upper = thresholds['5σ+']
    five_sigma_lower = thresholds['5σ-']
    
    current_time = df[timestamp_col].max()
    current_value = df[df[timestamp_col] == current_time][value_col].iloc[0]
    
    ttf_roc_results = {}
    
    for period in periods:
        try:
            # Calculate ROC for period
            slope_per_day, _ = calculate_rate_change(
                df, 
                window_days=period, 
                column=value_col,
                timestamp_col=timestamp_col,
                convert_to='per_day'
            )
            
            # Calculate time to reach 5 sigma threshold
            # For upper threshold
            if slope_per_day > 0:
                days_to_upper = (five_sigma_upper - current_value) / slope_per_day
                ttf_upper = current_time + pd.Timedelta(days=days_to_upper)
                
                # Only include future breaches
                if days_to_upper > 0:
                    ttf_roc_results[f"TTF ROC {period}d (Upper 5σ)"] = ttf_upper
                else:
                    ttf_roc_results[f"TTF ROC {period}d (Upper 5σ)"] = None
            else:
                ttf_roc_results[f"TTF ROC {period}d (Upper 5σ)"] = None
            
            # For lower threshold  
            if slope_per_day < 0:
                days_to_lower = (five_sigma_lower - current_value) / slope_per_day
                ttf_lower = current_time + pd.Timedelta(days=days_to_lower)
                
                # Only include future breaches
                if days_to_lower > 0:
                    ttf_roc_results[f"TTF ROC {period}d (Lower 5σ)"] = ttf_lower
                else:
                    ttf_roc_results[f"TTF ROC {period}d (Lower 5σ)"] = None
            else:
                ttf_roc_results[f"TTF ROC {period}d (Lower 5σ)"] = None
                
        except Exception as e:
            print(f" - Could not calculate TTF ROC for {period}d period: {e}")
            ttf_roc_results[f"TTF ROC {period}d (Upper 5σ)"] = None
            ttf_roc_results[f"TTF ROC {period}d (Lower 5σ)"] = None
    
    print("\nTTF ROC (Rate of Change) Predictions:")
    for label, breach_time in ttf_roc_results.items():
        if breach_time:
            time_to_breach = breach_time - pd.Timestamp.now()
            print(f" - {label}: {breach_time.strftime('%Y-%m-%d %H:%M')} ({time_to_breach})")
        else:
            print(f" - {label}: None (trend not leading to breach)")
    
    return ttf_roc_results

def estimate_ttf_roc_combined(df: pd.DataFrame, timestamp_col: str = 'Timestamp', 
                             value_col: str = 'Value', periods: list = [14, 30, 60]) -> dict:
    # Estimate TTF ROC with simplified output that show earliest breach time for each period

    print("\nCalculating TTF ROC (Rate of Change) - Combined predictions...")
    
    # Sigma thresholds
    df_marked, thresholds = calculate_sigma_median(df, column=value_col)
    five_sigma_upper = thresholds['5σ+']
    five_sigma_lower = thresholds['5σ-']
    
    # Get most recent value 
    current_time = df[timestamp_col].max()
    current_value = df[df[timestamp_col] == current_time][value_col].iloc[0]
    
    ttf_roc_combined = {}
    
    for period in periods:
        try:
            # Calculate ROC for period
            slope_per_day, _ = calculate_rate_change(
                df, 
                window_days=period, 
                column=value_col,
                timestamp_col=timestamp_col,
                convert_to='per_day'
            )
            
            breach_times = []
            
            # Time to reach 5 sigma threshold
            if slope_per_day > 0:
                days_to_upper = (five_sigma_upper - current_value) / slope_per_day
                if days_to_upper > 0:
                    breach_times.append(current_time + pd.Timedelta(days=days_to_upper))
            
            if slope_per_day < 0:
                days_to_lower = (five_sigma_lower - current_value) / slope_per_day
                if days_to_lower > 0:
                    breach_times.append(current_time + pd.Timedelta(days=days_to_lower))
            
            # Take earliest breach time
            if breach_times:
                ttf_roc_combined[f"TTF ROC {period}d"] = min(breach_times)
            else:
                ttf_roc_combined[f"TTF ROC {period}d"] = None
                
        except Exception as e:
            print(f"   - Could not calculate TTF ROC for {period}d period: {e}")
            ttf_roc_combined[f"TTF ROC {period}d"] = None
    
    print("\nTTF ROC (Rate of Change) - Combined Predictions:")
    for label, breach_time in ttf_roc_combined.items():
        if breach_time:
            time_to_breach = breach_time - pd.Timestamp.now()
            print(f"   - {label}: {breach_time.strftime('%Y-%m-%d %H:%M')} ({time_to_breach})")
        else:
            print(f"   - {label}: None (trend not leading to breach)")
    
    return ttf_roc_combined 