import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils.data_clean import data_clean
from models.forecast_utils import build_feature_frame
from models.xgb_model import prepare_data as prepare_xgb_data, prepare_svm_data, prepare_prophet_data
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None

warnings.filterwarnings("ignore", category=FutureWarning)

if OPTUNA_AVAILABLE:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_score: float
    metric_name: str
    n_trials_completed: int
    optimization_time_seconds: float
    study: Optional["optuna.Study"] = None
    
    def summary(self) -> str:
        # Return a summary string of optimization results
        return (
            f"Optimization Complete:\n"
            f"- Best {self.metric_name.upper()}: {self.best_score:.4f}\n"
            f"- Trials: {self.n_trials_completed}\n"
            f"- Time: {self.optimization_time_seconds:.1f}s\n"
            f"- Best Parameters:\n" +
            "\n".join(f" • {k}: {v}" for k, v in self.best_params.items())
        )

def check_optuna_available():
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna library not installed."
        )

# XGBoost
def create_xgboost_objective(
    feature_df: pd.DataFrame,
    test_size: float = 0.2,
    metric: str = "rmse",
    random_state: int = 42,
) -> Callable:

    feature_cols = [col for col in feature_df.columns if col not in {"ds", "y"}]
    
    # Split data for validation
    split_index = int(len(feature_df) * (1 - test_size))
    split_index = max(50, min(len(feature_df) - 10, split_index))
    
    train_df = feature_df.iloc[:split_index]
    val_df = feature_df.iloc[split_index:]
    
    X_train = train_df[feature_cols]
    y_train = train_df["y"]
    X_val = val_df[feature_cols]
    y_val = val_df["y"]
    
    def objective(trial: "optuna.Trial") -> float:
        # Function to minimize
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": random_state,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "verbosity": 0,
        }
        
        model = XGBRegressor(**params)
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_pred = model.predict(X_val)
        
        if metric == "mae":
            return mean_absolute_error(y_val, y_pred)
        elif metric == "mape":
            return mean_absolute_percentage_error(y_val, y_pred)
        else:  # Default to RMSE
            return np.sqrt(mean_squared_error(y_val, y_pred))
    
    return objective

def optimize_xgboost(
    feature_df: pd.DataFrame,
    n_trials: int = 50,
    timeout: Optional[int] = 300,
    metric: str = "rmse",
    test_size: float = 0.2,
    random_state: int = 42,
    show_progress: bool = True,
    verbose: bool = True,
) -> OptimizationResult:
    # Run Optuna
    import time
    check_optuna_available()
    
    if verbose:
        print(f"\n Starting XGBoost Optuna optimization...")
        print(f" - Trials: {n_trials}")
        print(f" - Timeout: {timeout}s" if timeout else "   - Timeout: None")
        print(f" - Metric: {metric.upper()}")
        print(f" - Data points: {len(feature_df)}")
    
    start_time = time.time()
    
    # Create study with TPE sampler
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="xgboost_optimization",
    )
    
    # Create, run objective
    objective = create_xgboost_objective(
        feature_df, 
        test_size=test_size,
        metric=metric,
        random_state=random_state
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
        n_jobs=1,
    )
    
    elapsed = time.time() - start_time
    
    result = OptimizationResult(
        best_params=study.best_params,
        best_score=study.best_value,
        metric_name=metric,
        n_trials_completed=len(study.trials),
        optimization_time_seconds=elapsed,
        study=study,
    )
    
    if verbose:
        print(f"\n XGBoost optimization complete")
        print(f" - Best {metric.upper()}: {result.best_score:.4f}")
        print(f" - Trials completed: {result.n_trials_completed}")
        print(f" - Time: {elapsed:.1f}s")
    
    return result

# Random Forest
def create_random_forest_objective(
    feature_df: pd.DataFrame,
    test_size: float = 0.2,
    metric: str = "rmse",
    random_state: int = 42,
) -> Callable:
    
    feature_cols = [col for col in feature_df.columns if col not in {"ds", "y"}]

    split_index = int(len(feature_df) * (1 - test_size))
    split_index = max(50, min(len(feature_df) - 10, split_index))

    train_df = feature_df.iloc[:split_index]
    val_df = feature_df.iloc[split_index:]

    X_train = train_df[feature_cols]
    y_train = train_df["y"]
    X_val = val_df[feature_cols]
    y_val = val_df["y"]

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": random_state,
            "n_jobs": -1,
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if metric == "mae":
            return mean_absolute_error(y_val, y_pred)
        elif metric == "mape":
            return mean_absolute_percentage_error(y_val, y_pred)
        else:
            return np.sqrt(mean_squared_error(y_val, y_pred))

    return objective

def optimize_random_forest(
    feature_df: pd.DataFrame,
    n_trials: int = 50,
    timeout: Optional[int] = 300,
    metric: str = "rmse",
    test_size: float = 0.2,
    random_state: int = 42,
    show_progress: bool = True,
    verbose: bool = True,
) -> OptimizationResult:
    import time
    check_optuna_available()

    if verbose:
        print(f"\n Starting Random Forest Optuna optimization...")
        print(f" - Trials: {n_trials}")
        print(f" - Timeout: {timeout}s" if timeout else "   - Timeout: None")
        print(f" - Metric: {metric.upper()}")
        print(f" - Data points: {len(feature_df)}")

    start_time = time.time()

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="random_forest_optimization",
    )

    objective = create_random_forest_objective(
        feature_df,
        test_size=test_size,
        metric=metric,
        random_state=random_state,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
        n_jobs=1,
    )

    elapsed = time.time() - start_time

    result = OptimizationResult(
        best_params=study.best_params,
        best_score=study.best_value,
        metric_name=metric,
        n_trials_completed=len(study.trials),
        optimization_time_seconds=elapsed,
        study=study,
    )

    if verbose:
        print(f"\n Random Forest optimization complete")
        print(f" - Best {metric.upper()}: {result.best_score:.4f}")
        print(f" - Trials completed: {result.n_trials_completed}")
        print(f" - Time: {elapsed:.1f}s")

    return result

# SVM
def create_svm_objective(
    feature_df: pd.DataFrame,
    test_size: float = 0.2,
    metric: str = "rmse",
    random_state: int = 42,
) -> Callable:
    
    feature_cols = [col for col in feature_df.columns if col not in {"ds", "y"}]

    # Split data for validation
    split_index = int(len(feature_df) * (1 - test_size))
    split_index = max(50, min(len(feature_df) - 10, split_index))

    train_df = feature_df.iloc[:split_index]
    val_df = feature_df.iloc[split_index:]

    X_train_full = train_df[feature_cols].to_numpy(dtype=float)
    y_train_full = train_df["y"].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["y"].to_numpy(dtype=float)

    def objective(trial: "optuna.Trial") -> float:
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("c", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)

        max_train_samples = trial.suggest_int("max_train_samples", 2000, 10000, step=1000)
        if len(X_train_full) > max_train_samples:
            idx = np.linspace(0, len(X_train_full) - 1, max_train_samples, dtype=int)
            X_train = X_train_full[idx]
            y_train = y_train_full[idx]
        else:
            X_train = X_train_full
            y_train = y_train_full

        model = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                SVR(**params),
            ),
            transformer=StandardScaler(),
        )

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        except Exception:
            return float("inf")

        if metric == "mae":
            return mean_absolute_error(y_val, y_pred)
        elif metric == "mape":
            return mean_absolute_percentage_error(y_val, y_pred)
        else:  # Default to RMSE
            return np.sqrt(mean_squared_error(y_val, y_pred))

    return objective

def optimize_svm(
    feature_df: pd.DataFrame,
    n_trials: int = 50,
    timeout: Optional[int] = 300,
    metric: str = "rmse",
    test_size: float = 0.2,
    random_state: int = 42,
    show_progress: bool = True,
    verbose: bool = True,
) -> OptimizationResult:
    import time
    check_optuna_available()

    if verbose:
        print(f"\n Starting SVM Optuna optimization...")
        print(f" - Trials: {n_trials}")
        print(f" - Timeout: {timeout}s" if timeout else "   - Timeout: None")
        print(f" - Metric: {metric.upper()}")
        print(f" - Data points: {len(feature_df)}")

    start_time = time.time()

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="svm_optimization",
    )

    objective = create_svm_objective(
        feature_df,
        test_size=test_size,
        metric=metric,
        random_state=random_state,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
        n_jobs=1,
    )

    elapsed = time.time() - start_time

    result = OptimizationResult(
        best_params=study.best_params,
        best_score=study.best_value,
        metric_name=metric,
        n_trials_completed=len(study.trials),
        optimization_time_seconds=elapsed,
        study=study,
    )

    if verbose:
        print(f"\n SVM optimization complete")
        print(f" - Best {metric.upper()}: {result.best_score:.4f}")
        print(f" - Trials completed: {result.n_trials_completed}")
        print(f" - Time: {elapsed:.1f}s")

    return result

# Prophet
def create_prophet_objective(
    df_prepared: pd.DataFrame,
    test_size: float = 0.2,
    metric: str = "rmse",
) -> Callable:
    from prophet import Prophet
    
    # Split data
    split_index = int(len(df_prepared) * (1 - test_size))
    split_index = max(50, min(len(df_prepared) - 10, split_index))
    
    train_df = df_prepared.iloc[:split_index]
    val_df = df_prepared.iloc[split_index:]
    
    def objective(trial: "optuna.Trial") -> float:
        changepoint_prior_scale = trial.suggest_float(
            "changepoint_prior_scale", 0.001, 0.5, log=True
        )
        seasonality_prior_scale = trial.suggest_float(
            "seasonality_prior_scale", 0.01, 10.0, log=True
        )
        seasonality_mode = trial.suggest_categorical(
            "seasonality_mode", ["additive", "multiplicative"]
        )
        n_changepoints = trial.suggest_int("n_changepoints", 3, 25)
        changepoint_range = trial.suggest_float("changepoint_range", 0.7, 0.95)
        
        # Seasonality toggles
        daily_seasonality = trial.suggest_categorical(
            "daily_seasonality", [True, False, "auto"]
        )
        weekly_seasonality = trial.suggest_categorical(
            "weekly_seasonality", [True, False, "auto"]
        )
        
        try:
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode=seasonality_mode,
                n_changepoints=n_changepoints,
                changepoint_range=changepoint_range,
                daily_seasonality=daily_seasonality,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=False,
                uncertainty_samples=0,
                mcmc_samples=0,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_df)
            
            # Predict on validation set
            future = val_df[['ds']].copy()
            forecast = model.predict(future)
            
            y_true = val_df['y'].values
            y_pred = forecast['yhat'].values
            
            if metric == "mae":
                return mean_absolute_error(y_true, y_pred)
            elif metric == "mape":
                return mean_absolute_percentage_error(y_true, y_pred)
            else:  # Default to RMSE
                return np.sqrt(mean_squared_error(y_true, y_pred))
                
        except Exception as e:
            # Return high score for failed trials
            return float('inf')
    
    return objective

def optimize_prophet(
    df_prepared: pd.DataFrame,
    n_trials: int = 30,
    timeout: Optional[int] = 300,
    metric: str = "rmse",
    test_size: float = 0.2,
    random_state: int = 42,
    show_progress: bool = True,
    verbose: bool = True,
) -> OptimizationResult:

    import time
    check_optuna_available()
    
    if verbose:
        print(f"\n Starting Prophet Optuna optimization...")
        print(f" - Trials: {n_trials}")
        print(f" - Timeout: {timeout}s" if timeout else "   - Timeout: None")
        print(f" - Metric: {metric.upper()}")
        print(f" - Data points: {len(df_prepared)}")
    
    start_time = time.time()
    
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="prophet_optimization",
    )
    
    objective = create_prophet_objective(
        df_prepared,
        test_size=test_size,
        metric=metric,
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
        n_jobs=1,
    )
    
    elapsed = time.time() - start_time
    
    result = OptimizationResult(
        best_params=study.best_params,
        best_score=study.best_value,
        metric_name=metric,
        n_trials_completed=len(study.trials),
        optimization_time_seconds=elapsed,
        study=study,
    )
    
    if verbose:
        print(f"\n Prophet optimization complete")
        print(f" - Best {metric.upper()}: {result.best_score:.4f}")
        print(f" - Trials completed: {result.n_trials_completed}")
        print(f" - Time: {elapsed:.1f}s")
    
    return result

#Utilities
def get_param_importance(study: "optuna.Study") -> Dict[str, float]:
    # Get hyperparameter importance scores.
    check_optuna_available()
    
    try:
        importance = optuna.importance.get_param_importances(study)
        return dict(importance)
    except Exception:
        return {}

def is_optuna_available() -> bool:
    return OPTUNA_AVAILABLE

# CLI
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    parser = argparse.ArgumentParser(
        description="Run Optuna optimization for a CDF time-series tag.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "prophet", "svm"],
        default="xgboost",
        help="Model to optimize.",
    )
    parser.add_argument(
        "--tag",
        default="KASAWARI_PI_PC.SKA.KSCPP.DCS.SW.22PIA-0028.PV",
        help="CDF external ID of tag.",
    )
    parser.add_argument(
        "--time-range",
        default="365d-ago",
        dest="time_range",
        help="Start of retrieval window (relative time string, e.g., 365d-ago).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Max seconds to spend on optimization (0 = no limit).",
    )
    parser.add_argument(
        "--metric",
        choices=["rmse", "mae", "mape"],
        default="rmse",
        help="Evaluation metric.",
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="CDF environment passed to client_gen.",
    )

    args = parser.parse_args()

    print(f"\nConnecting to CDF ({args.env})...")
    try:
        from auth.client_gen import client_gen
        client = client_gen(args.env)
    except Exception as exc:
        print(f"Failed to create CDF client: {exc}")
        sys.exit(1)

    # Retrieve time-series data
    print(f"Fetching tag: {args.tag}  [{args.time_range} → now]")
    try:
        from utils.data_ingestion import RetrieveData
        retriever = RetrieveData(client)
        df_raw = retriever.retrieve_data(args.tag, args.time_range, "now", "interpolation", "1h")
        print(f" • {len(df_raw)} raw data points retrieved.")
    except Exception as exc:
        print(f"Data retrieval failed: {exc}")
        sys.exit(1)

    # Clean data
    try:
        df_clean, removed, total = data_clean(df_raw, "Value")
        pct = (removed / total * 100) if total else 0
        print(f" • Removed {removed} outliers out of {total} rows ({pct:.1f}%).")
    except Exception as exc:
        print(f"Data cleaning failed: {exc}")
        sys.exit(1)

    # Prepare features
    timeout_val = args.timeout if args.timeout > 0 else None

    if args.model == "xgboost":
        try:
            df_prepared, _ = prepare_xgb_data(df_clean, freq="1H")
            feature_df = build_feature_frame(df_prepared)
            
            print(f"   • XGBoost feature matrix: {feature_df.shape[0]} rows × {feature_df.shape[1]} cols.")
        except Exception as exc:
            print(f" XGBoost feature prep failed: {exc}")
            sys.exit(1)

        result = optimize_xgboost(
            feature_df=feature_df,
            n_trials=args.trials,
            timeout=timeout_val,
            metric=args.metric,
            verbose=True,
            show_progress=True,
        )

    elif args.model == "svm":
        try:
            df_prepared, _ = prepare_svm_data(df_clean, freq="1H")
            feature_df = build_feature_frame(df_prepared)
            print(f" • SVM feature matrix: {feature_df.shape[0]} rows × {feature_df.shape[1]} cols.")
        except Exception as exc:
            print(f"SVM feature prep failed: {exc}")
            sys.exit(1)

        result = optimize_svm(
            feature_df=feature_df,
            n_trials=args.trials,
            timeout=timeout_val,
            metric=args.metric,
            verbose=True,
            show_progress=True,
        )

    else:  # prophet
        try:
            from models.prophet_model import prepare_data as prophet_prepare_data
            df_prepared, _ = prophet_prepare_data(df_clean)
            print(f" • Prophet prepared data: {len(df_prepared)} rows.")
        except Exception as exc:
            print(f"Prophet data prep failed: {exc}")
            sys.exit(1)

        result = optimize_prophet(
            df_prepared=df_prepared,
            n_trials=args.trials,
            timeout=timeout_val,
            metric=args.metric,
            verbose=True,
            show_progress=True,
        )

    print(f"\n{result.summary()}")