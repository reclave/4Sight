import pandas as pd
import numpy as np
from typing import Optional, Callable, Any, Tuple
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def performance_score(mape_val, rmse_val, mae_val, data_std):
    # Composite performance score for forecasting model (traffic light system)
    """
    Returns:
        tuple: (composite_score, category, color) # Color for streamlit
    """
    # Normalise metrics to 0-100 scale
    # MAPE: 0% error = 100 score, 50%+ error = 0 score
    mape_score = max(0, min(100, (1 - min(mape_val, 0.5) / 0.5) * 100))
    
    # RMSE normalize by std
    # 0 error = 100 score, 2+ standard deviations = 0 score
    rmse_normalized = rmse_val / max(data_std, 1e-8)  # Avoid division by zero
    rmse_score = max(0, min(100, (1 - min(rmse_normalized, 2.0) / 2.0) * 100))
    
    # MAE normalize by std
    # 0 error = 100 score, 1+ standard deviation = 0 score
    mae_normalized = mae_val / max(data_std, 1e-8)  # Avoid division by zero
    mae_score = max(0, min(100, (1 - min(mae_normalized, 1.0)) * 100))
    
    # Weighted composite score
    composite_score = (mape_score * 0.4 + rmse_score * 0.3 + mae_score * 0.3)

    # Traffic light system (detemined together w/ team)
    if composite_score >= 70:
        category = "Good Forecasting Model Performance"
        color = "success"
    elif composite_score >= 30:
        category = "Moderate Forecasting Model Performance"
        color = "warning"
    else:
        category = "Poor Forecasting Model Performance"
        color = "error"

    return composite_score, category, color

def evaluate_model(train_data: pd.DataFrame, test_data: pd.DataFrame,
                   seasonality_mode: str = 'additive',
                   changepoint_prior_scale: float = 0.1,
                   n_changepoints: int = 6,
                   changepoint_range: float = 0.9,
                   daily_seasonality: bool = True,
                   weekly_seasonality: bool = True,
                   model_name: str = 'Prophet',
                   direct_evaluator: Optional[Callable[[pd.DataFrame, Optional[Any]], Tuple[Any, ...]]] = None,
                   evaluator_config: Optional[Any] = None) -> tuple:
    # Evaluate model on train/test split - returns mae, rmse, mape, y_pred.
    # If direct_evaluator is provided, run centralized evaluation flow for non-Prophet models.
    try:
        if direct_evaluator is not None:
            combined = pd.concat([train_data, test_data], ignore_index=True)
            result = direct_evaluator(combined, evaluator_config)
            if result is None or len(result) < 4:
                raise ValueError("Direct evaluator returned invalid evaluation output")

            mae, rmse, mape, y_pred = result[:4]
            print(f"\n{model_name} model evaluation completed.")
            return mae, rmse, mape, y_pred

        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=False,
            uncertainty_samples=100, #More samples give more reliable CI but increase computation time
            mcmc_samples=0  # Use MAP estimation for faster training
        )

        model.fit(train_data)

        # Create future dataframe for test period
        future = test_data[['ds']].copy()

        # Make predictions
        forecast = model.predict(future)
        y_pred = forecast['yhat'].values
        y_true = test_data['y'].values

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        print(f"\n{model_name} model evaluation completed.")
        return mae, rmse, mape, y_pred

    except Exception as e:
        print(f"\n{model_name} model evaluation failed: {e}")
        return None, None, None, None

def get_model_diagnostics(model, df_prepared: pd.DataFrame) -> dict:
    """
    Extract Prophet model diagnostics for analysis.
    """
    try:
        # Get trend changepoints
        changepoints = model.changepoints

        # Calculate trend strength
        trend_component = model.predict(df_prepared)['trend']
        trend_strength = (trend_component.max() - trend_component.min()) / df_prepared['y'].std()

        # Seasonality components (if any)
        forecast_full = model.predict(df_prepared)
        seasonality_components = [col for col in forecast_full.columns if col.endswith('_seasonal')]

        diagnostics = {
            'num_changepoints': len(changepoints),
            'trend_strength': trend_strength,
            'seasonality_components': seasonality_components,
            'last_changepoint': changepoints.max() if len(changepoints) > 0 else None,
            'model_params': {
                'changepoint_prior_scale': model.changepoint_prior_scale,
                'seasonality_mode': model.seasonality_mode,
                'interval_width': model.interval_width
            }
        }

        return diagnostics

    except Exception as e:
        print(f"   - Could not extract model diagnostics: {e}")
        return {}

def cross_validate_model(df_prepared: pd.DataFrame,
                          initial_days: int = 30,
                          period_days: int = 7,
                          horizon_days: int = 7) -> dict:
    """
    Perform time series cross-validation to assess model performance.
    """
    try:
        model = Prophet(
            seasonality_mode='additive',
            changepoint_prior_scale=0.1,
            n_changepoints=6, # Add potential changepoints
            changepoint_range=0.9,
            daily_seasonality='auto',
            weekly_seasonality='auto',
            yearly_seasonality=False,
            uncertainty_samples=100
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

        model.fit(df_prepared)

        df_cv = cross_validation(
            model,
            initial=f'{initial_days} days',
            period=f'{period_days} days',
            horizon=f'{horizon_days} days'
        )

        df_perf = performance_metrics(df_cv)

        cv_results = {
            'mae_mean': df_perf['mae'].mean(),
            'mae_std': df_perf['mae'].std(),
            'rmse_mean': df_perf['rmse'].mean(),
            'rmse_std': df_perf['rmse'].std(),
            'mape_mean': df_perf['mape'].mean(),
            'mape_std': df_perf['mape'].std(),
            'coverage': df_perf['coverage'].mean() if 'coverage' in df_perf.columns else None
        }

        print(f"Cross-validation results (±std):")
        print(f"   MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
        print(f"   RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
        print(f"   MAPE: {cv_results['mape_mean']:.4f} ± {cv_results['mape_std']:.4f}")

        return cv_results

    except Exception as e:
        print(f"   - Cross-validation failed: {e}")
        return {}
