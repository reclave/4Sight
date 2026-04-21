import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from models.forecast_utils import (
    build_feature_frame,
    build_time_features,
    estimate_ttf as shared_estimate_ttf,
    make_model_evaluator,
    prepare_data as shared_prepare_data,
    train_tree_forecaster,
    evaluate_tree_forecaster,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Check for Optuna
try:
    from models.optuna_optimizer import optimize_xgboost, is_optuna_available
    OPTUNA_AVAILABLE = is_optuna_available()
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class XGBoostConfig:
    """Configuration for the Direct Multi-Horizon XGBoost forecaster."""

    freq: str = "1H"
    learning_rate: float = 0.25 #0.05
    max_depth: int = 3 #5
    n_estimators: int = 500 #200
    subsample: float = 0.7 #0.8
    colsample_bytree: float = 0.9 #0.8
    reg_lambda: float = 0.27 #1.0
    reg_alpha: float = 0.12 #0.1
    min_child_weight: int = 7 #3
    random_state: int = 42
    test_size: float = 0.2
    
    # Blend XGBoost with linear trend for stability
    trend_blend_weight: float = 0.3  # 0 = pure XGBoost, 1 = pure trend
    
    # Optuna settings
    use_optuna: bool = False
    optuna_n_trials: int = 50
    optuna_timeout: Optional[int] = 300
    optuna_metric: str = "rmse"

    def model_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "min_child_weight": self.min_child_weight,
            "random_state": self.random_state,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }
    
    def update_from_optuna(self, best_params: Dict[str, Any]) -> "XGBoostConfig":
        """Create a new config with optimized parameters from Optuna."""
        return XGBoostConfig(
            freq=self.freq,
            learning_rate=best_params.get("learning_rate", self.learning_rate),
            max_depth=best_params.get("max_depth", self.max_depth),
            n_estimators=best_params.get("n_estimators", self.n_estimators),
            subsample=best_params.get("subsample", self.subsample),
            colsample_bytree=best_params.get("colsample_bytree", self.colsample_bytree),
            reg_lambda=best_params.get("reg_lambda", self.reg_lambda),
            reg_alpha=best_params.get("reg_alpha", self.reg_alpha),
            min_child_weight=best_params.get("min_child_weight", self.min_child_weight),
            random_state=self.random_state,
            test_size=self.test_size,
            trend_blend_weight=self.trend_blend_weight,
            use_optuna=False,
            optuna_n_trials=self.optuna_n_trials,
            optuna_timeout=self.optuna_timeout,
            optuna_metric=self.optuna_metric,
        )


def train_xgb_direct(
    df_prepared: pd.DataFrame,
    forecast_steps: int,
    freq: str,
    config: Optional[XGBoostConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, XGBRegressor]:
    config = config or XGBoostConfig()
    return train_tree_forecaster(
        df_prepared=df_prepared,
        forecast_steps=forecast_steps,
        freq=freq,
        config=config,
        estimator_factory=lambda current_config: XGBRegressor(**current_config.model_params()),
        feature_builder=build_time_features,
        include_calendar_patterns=True,
    )


def evaluate_xgb_direct(
    df_prepared: pd.DataFrame,
    config: Optional[XGBoostConfig] = None,
) -> Tuple[float, float, float, np.ndarray, float]:
    config = config or XGBoostConfig()
    return evaluate_tree_forecaster(
        df_prepared=df_prepared,
        config=config,
        estimator_factory=lambda current_config: XGBRegressor(**current_config.model_params()),
        feature_builder=build_time_features,
    )


prepare_data = shared_prepare_data
evaluate_model = make_model_evaluator(
    "XGBoost (Direct)",
    evaluate_xgb_direct,
    XGBoostConfig,
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
) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[XGBRegressor]]:
    """
    Uses Direct Multi-Horizon approach for robust long-term forecasting.
    """
    config = XGBoostConfig(
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
            
            opt_result = optimize_xgboost(
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
            print(f"   • Using optimized parameters")
        elif use_optuna and not OPTUNA_AVAILABLE:
            print("\nOptuna not available. Using default XGBoost parameters.")
        
        # Train and forecast using direct approach
        forecast_series, lower_series, upper_series, model = train_xgb_direct(
            df_prepared, forecast_steps, freq, config
        )
        
        if forecast_series is not None:
            print(f"\nXGBoost (Direct Multi-Horizon) training completed.")
            print(f"   - Forecast range: {forecast_series.min():.2f} to {forecast_series.max():.2f}")
            print(f"   - Uncertainty range: ±{(upper_series - lower_series).mean()/2:.2f}")
        
        return forecast_series, lower_series, upper_series, model
        
    except Exception as e:
        print(f"\nXGBoost model training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


estimate_ttf = shared_estimate_ttf
