import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
try:
    from models.optuna_optimizer import optimize_random_forest, is_optuna_available
    OPTUNA_AVAILABLE = is_optuna_available()
except ImportError:
    OPTUNA_AVAILABLE = False

@dataclass
class RandomForestConfig:
    freq: str = "1H"
    n_estimators: int = 300
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 3
    max_features: float = 0.8
    bootstrap: bool = True
    random_state: int = 42
    test_size: float = 0.2

    # Blend RF with linear trend (for stability)
    trend_blend_weight: float = 0.3  # 0 = pure RF, 1 = pure trend

    # Optuna settings
    use_optuna: bool = False
    optuna_n_trials: int = 50
    optuna_timeout: Optional[int] = 300
    optuna_metric: str = "rmse"

    def model_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_jobs": -1,
        }

    def update_from_optuna(self, best_params: Dict[str, Any]) -> "RandomForestConfig":
        # To create new config with optimized parameters from Optuna
        return RandomForestConfig(
            freq=self.freq,
            n_estimators=best_params.get("n_estimators", self.n_estimators),
            max_depth=best_params.get("max_depth", self.max_depth),
            min_samples_split=best_params.get("min_samples_split", self.min_samples_split),
            min_samples_leaf=best_params.get("min_samples_leaf", self.min_samples_leaf),
            max_features=best_params.get("max_features", self.max_features),
            bootstrap=best_params.get("bootstrap", self.bootstrap),
            random_state=self.random_state,
            test_size=self.test_size,
            trend_blend_weight=self.trend_blend_weight,
            use_optuna=False,
            optuna_n_trials=self.optuna_n_trials,
            optuna_timeout=self.optuna_timeout,
            optuna_metric=self.optuna_metric,
        )

def train_rf_direct(
    df_prepared: pd.DataFrame,
    forecast_steps: int,
    freq: str,
    config: Optional[RandomForestConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, RandomForestRegressor]:
    config = config or RandomForestConfig()
    return train_tree_forecaster(
        df_prepared=df_prepared,
        forecast_steps=forecast_steps,
        freq=freq,
        config=config,
        estimator_factory=lambda current_config: RandomForestRegressor(**current_config.model_params()),
        feature_builder=build_time_features,
        include_calendar_patterns=True,
    )

def evaluate_rf_direct(
    df_prepared: pd.DataFrame,
    config: Optional[RandomForestConfig] = None,
) -> Tuple[float, float, float, np.ndarray, float]:
    config = config or RandomForestConfig()
    return evaluate_tree_forecaster(
        df_prepared=df_prepared,
        config=config,
        estimator_factory=lambda current_config: RandomForestRegressor(**current_config.model_params()),
        feature_builder=build_time_features,
    )

prepare_data = shared_prepare_data
evaluate_model = make_model_evaluator(
    "Random Forest",
    evaluate_rf_direct,
    RandomForestConfig,
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
) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[RandomForestRegressor]]:
    config = RandomForestConfig(
        freq=freq,
        use_optuna=use_optuna,
        optuna_n_trials=optuna_n_trials,
        optuna_timeout=optuna_timeout,
    )

    try:
        # Run Optuna if enabled
        if use_optuna and OPTUNA_AVAILABLE:
            print("\nRunning hyperparameter optimization...")
            feature_df = build_feature_frame(df_prepared, build_time_features)

            opt_result = optimize_random_forest(
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
            print("\nOptuna not available. Using default Random Forest parameters.")

        # Train and forecast 
        forecast_series, lower_series, upper_series, model = train_rf_direct(
            df_prepared, forecast_steps, freq, config
        )

        if forecast_series is not None:
            print(f"\nRandom Forest (Direct Multi-Horizon) training completed.")
            print(f" - Forecast range: {forecast_series.min():.2f} to {forecast_series.max():.2f}")
            print(f" - Uncertainty range: ±{(upper_series - lower_series).mean()/2:.2f}")

        return forecast_series, lower_series, upper_series, model

    except Exception as e:
        print(f"\nRandom Forest model training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

estimate_ttf = shared_estimate_ttf