import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
from time import perf_counter
import pandas as pd
import numpy as np
from auth.client_gen import client_gen
from utils.data_ingestion import RetrieveData
from utils.data_clean import data_clean
from calc.sigma import calculate_sigma_median, count_threshold_crossings
from calc.rate_change import compare_multiple_periods
from utils.plot import plot_forecast, plot_model_accuracy, plot_rate_change
from utils.table_join import tbl_sigma_crossing_event, tbl_ttf_summary, tbl_forecast_pressure, tbl_rate_change, tbl_ttf_breach_times
from models.prophet_model import prepare_data as prophet_prepare_data, train_model as prophet_train_model, estimate_ttf as prophet_estimate_ttf
from utils.eval import evaluate_model as prophet_evaluate_model, performance_score
from models.xgb_model import prepare_data as xgboost_prepare_data, train_model as xgboost_train_model, estimate_ttf as xgboost_estimate_ttf
from models.xgb_model import evaluate_model as xgboost_evaluate_model
from models.rf_model import prepare_data as rf_prepare_data, train_model as rf_train_model, estimate_ttf as rf_estimate_ttf
from models.rf_model import evaluate_model as rf_evaluate_model
from models.svm_model import prepare_data as svm_prepare_data, train_model as svm_train_model, estimate_ttf as svm_estimate_ttf
from models.svm_model import evaluate_model as svm_evaluate_model
from models.lstm_model import prepare_data as lstm_prepare_data, train_model as lstm_train_model, estimate_ttf as lstm_estimate_ttf
from models.lstm_model import evaluate_model as lstm_evaluate_model

AVAILABLE_MODELS = {
    "XGBoost": {
        "prepare_data": xgboost_prepare_data,
        "train_model": xgboost_train_model,
        "estimate_ttf": xgboost_estimate_ttf,
        "evaluate_model": xgboost_evaluate_model,
    },
    "Random Forest": {
        "prepare_data": rf_prepare_data,
        "train_model": rf_train_model,
        "estimate_ttf": rf_estimate_ttf,
        "evaluate_model": rf_evaluate_model,
    },
    "SVM": {
        "prepare_data": svm_prepare_data,
        "train_model": svm_train_model,
        "estimate_ttf": svm_estimate_ttf,
        "evaluate_model": svm_evaluate_model,
    },
    "LSTM": {
        "prepare_data": lstm_prepare_data,
        "train_model": lstm_train_model,
        "estimate_ttf": lstm_estimate_ttf,
        "evaluate_model": lstm_evaluate_model,
    }
}

FORECAST_MODE_PRESETS = {
    "Fast": {
        "selected_models": ["Random Forest"],
        "ensemble_method": None,
        "use_optuna": True,
        "optuna_n_trials": 30,
        "optuna_timeout": 300,
    },
    "Balanced": {
        "selected_models": ["Random Forest", "XGBoost"],
        "ensemble_method": "Choose Best",
        "use_optuna": True,
        "optuna_n_trials": 50,
        "optuna_timeout": 300,
    },
    "Performance": {
        "selected_models": ["LSTM"],
        "ensemble_method": None,
        "use_optuna": False,
        "optuna_n_trials": 30,
        "optuna_timeout": 180,
    },
}

SELECTABLE_FORECAST_MODES = ["Fast", "Balanced", "Performance"]
CUSTOM_FORECAST_MODE = "Custom"
DEFAULT_TRAINING_PERIOD = "1 year ago"
DEFAULT_FORECAST_TIMESPAN = "2 months ahead"
DEFAULT_INTERVAL = "1h"
FORECAST_MODE_KEY = "predict_forecast_mode"
FORECAST_MODE_DISPLAY_KEY = "predict_forecast_mode_display"
FORECAST_MODE_PENDING_KEY = "predict_forecast_mode_pending"
FORECAST_MODE_APPLIED_KEY = "predict_forecast_mode_applied"
TRAINING_PERIOD_KEY = "predict_training_period"
FORECAST_TIMESPAN_KEY = "predict_forecast_timespan"
INTERVAL_KEY = "predict_interval"
SELECTED_MODELS_KEY = "predict_selected_models"
ENSEMBLE_METHOD_KEY = "predict_ensemble_method"
USE_OPTUNA_KEY = "predict_use_optuna"
OPTUNA_TRIALS_KEY = "predict_optuna_n_trials"
OPTUNA_TIMEOUT_KEY = "predict_optuna_timeout"


def _apply_forecast_mode_preset(mode_name):
    preset = FORECAST_MODE_PRESETS[mode_name]
    st.session_state[TRAINING_PERIOD_KEY] = DEFAULT_TRAINING_PERIOD
    st.session_state[FORECAST_TIMESPAN_KEY] = DEFAULT_FORECAST_TIMESPAN
    st.session_state[INTERVAL_KEY] = DEFAULT_INTERVAL
    st.session_state[SELECTED_MODELS_KEY] = list(preset["selected_models"])
    st.session_state[ENSEMBLE_METHOD_KEY] = preset["ensemble_method"]
    st.session_state[USE_OPTUNA_KEY] = preset["use_optuna"]
    st.session_state[OPTUNA_TRIALS_KEY] = preset["optuna_n_trials"]
    st.session_state[OPTUNA_TIMEOUT_KEY] = preset["optuna_timeout"]
    st.session_state[FORECAST_MODE_APPLIED_KEY] = mode_name


def _on_forecast_mode_change():
    new_mode = st.session_state.get(FORECAST_MODE_DISPLAY_KEY)
    if new_mode in FORECAST_MODE_PRESETS:
        _apply_forecast_mode_preset(new_mode)
    st.session_state[FORECAST_MODE_KEY] = new_mode


def _infer_forecast_mode(
    training_period,
    forecast_timespan,
    interval,
    selected_models,
    ensemble_method,
    use_optuna,
    optuna_n_trials,
    optuna_timeout,
):
    selected_models = list(selected_models or [])
    selected_set = set(selected_models)
    advanced_defaults_match = (
        training_period == DEFAULT_TRAINING_PERIOD
        and forecast_timespan == DEFAULT_FORECAST_TIMESPAN
        and interval == DEFAULT_INTERVAL
    )

    if (
        advanced_defaults_match
        and len(selected_models) == 1
        and selected_set == {"Random Forest"}
        and use_optuna
        and optuna_n_trials == 30
        and optuna_timeout == 300
    ):
        return "Fast"

    if (
        advanced_defaults_match
        and len(selected_models) == 2
        and selected_set == {"Random Forest", "XGBoost"}
        and ensemble_method == "Choose Best"
        and use_optuna
        and optuna_n_trials == 50
        and optuna_timeout == 300
    ):
        return "Balanced"

    if (
        advanced_defaults_match
        and len(selected_models) == 1
        and selected_set == {"LSTM"}
        and not use_optuna
    ):
        return "Performance"

    return CUSTOM_FORECAST_MODE


def _sync_forecast_mode_state(available_model_names, update_display_key=True):
    _KEY_DEFAULTS = {
        TRAINING_PERIOD_KEY: DEFAULT_TRAINING_PERIOD,
        FORECAST_TIMESPAN_KEY: DEFAULT_FORECAST_TIMESPAN,
        INTERVAL_KEY: DEFAULT_INTERVAL,
        SELECTED_MODELS_KEY: list(FORECAST_MODE_PRESETS["Balanced"]["selected_models"]),
        ENSEMBLE_METHOD_KEY: None,
        USE_OPTUNA_KEY: False,
        OPTUNA_TRIALS_KEY: 30,
        OPTUNA_TIMEOUT_KEY: 180,
    }
    missing_keys = [k for k in _KEY_DEFAULTS if k not in st.session_state]
    if missing_keys:
        if len(missing_keys) == len(_KEY_DEFAULTS):
            initial_mode = st.session_state.get(FORECAST_MODE_KEY, "Balanced")
            if initial_mode in FORECAST_MODE_PRESETS:
                _apply_forecast_mode_preset(initial_mode)
            else:
                _apply_forecast_mode_preset("Balanced")
        else:
            for k in missing_keys:
                st.session_state[k] = _KEY_DEFAULTS[k]

    valid_selected_models = [
        model
        for model in st.session_state.get(SELECTED_MODELS_KEY, [])
        if model in available_model_names
    ]
    if valid_selected_models != st.session_state.get(SELECTED_MODELS_KEY):
        fallback_mode = st.session_state.get(FORECAST_MODE_APPLIED_KEY, "Balanced")
        fallback_models = [
            model
            for model in FORECAST_MODE_PRESETS.get(fallback_mode, FORECAST_MODE_PRESETS["Balanced"])["selected_models"]
            if model in available_model_names
        ]
        st.session_state[SELECTED_MODELS_KEY] = valid_selected_models or fallback_models
        if not st.session_state[SELECTED_MODELS_KEY]:
            st.session_state[SELECTED_MODELS_KEY] = available_model_names[:1]

    inferred_mode = _infer_forecast_mode(
        training_period=st.session_state.get(TRAINING_PERIOD_KEY, DEFAULT_TRAINING_PERIOD),
        forecast_timespan=st.session_state.get(FORECAST_TIMESPAN_KEY, DEFAULT_FORECAST_TIMESPAN),
        interval=st.session_state.get(INTERVAL_KEY, DEFAULT_INTERVAL),
        selected_models=st.session_state.get(SELECTED_MODELS_KEY, []),
        ensemble_method=st.session_state.get(ENSEMBLE_METHOD_KEY),
        use_optuna=bool(st.session_state.get(USE_OPTUNA_KEY, False)),
        optuna_n_trials=int(st.session_state.get(OPTUNA_TRIALS_KEY, 30)),
        optuna_timeout=int(st.session_state.get(OPTUNA_TIMEOUT_KEY, 180)),
    )
    st.session_state[FORECAST_MODE_KEY] = inferred_mode
    if update_display_key:
        st.session_state[FORECAST_MODE_DISPLAY_KEY] = inferred_mode
    return inferred_mode


project_root = Path(__file__).parent
data_path = project_root / "data"
utils_path = project_root / "utils"

for path in [str(data_path), str(utils_path)]:
    if path not in sys.path:
        sys.path.append(path)

st.set_page_config(
    page_title="4Sight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _format_display_name(ts):
    if not ts.name or ts.name == "N/A":
        return "N/A"

    parts = ts.name.split('.')
    if len(parts) > 3:
        tag_id = parts[-2]
        system_info = '/'.join(parts[2:-2])
        display_name = f"{tag_id} ({system_info})"
    else:
        display_name = ts.name

    if ts.description and ts.description != "N/A":
        display_name += f": {ts.description}"

    if ts.unit and ts.unit != "N/A":
        display_name += f" ({ts.unit})"

    # if ts.external_id and ts.external_id != "N/A":
    #     display_name += f" [External ID {ts.external_id}]"

    return display_name

def search_ts(query="", limit=100, return_total_count=False, exact_match=False):
    try:
        client = client_gen('dev')
        if client is None:
            return [] if not return_total_count else ([], 0)

        # Check if external id is available
        def has_valid_external_id(ts):
            return (ts.external_id and 
                    ts.external_id.strip() != "" and 
                    ts.external_id != "N/A" and 
                    not ts.external_id.startswith("4Sight_"))

        # If no query, return all time series up to limit
        if not query or query.strip() == "":
            all_results = list(client.time_series.list(limit=limit))
            # Filter out 4Sight_ prefixed external IDs and unavailable external IDs
            filtered_results = [ts for ts in all_results if has_valid_external_id(ts)]
            total_count = len(filtered_results)
            return (filtered_results, total_count) if return_total_count else filtered_results

        clean_query = query.strip()
        
        # Check for quoted strings
        auto_exact_match = False
        if clean_query.startswith('"') and clean_query.endswith('"') and len(clean_query) > 2:
            clean_query = clean_query[1:-1]  # Remove quotes
            auto_exact_match = True
        
        # Use exact matching optional
        use_exact_match = exact_match or auto_exact_match
        
        # Search by name and desc
        name_results = list(client.time_series.search(name=clean_query, limit=limit))
        desc_results = list(client.time_series.search(description=clean_query, limit=limit))
        
        seen_ids = set()
        combined_results = []
        
        # Check for exact matching
        def is_exact_match(text, search_term):
            if not text:
                return False
            return search_term.lower() in text.lower()
        
        # Process name results
        for ts in name_results:
            if ts.id not in seen_ids:
                # Filter out 4Sight_ external IDs and NA external IDs
                if not has_valid_external_id(ts):
                    continue
                if not use_exact_match or is_exact_match(ts.name, clean_query):
                    seen_ids.add(ts.id)
                    combined_results.append(ts)
        
        # Process description results
        for ts in desc_results:
            if ts.id not in seen_ids:
                # Filter out 4Sight_ external IDs and NA external IDs
                if not has_valid_external_id(ts):
                    continue
                if not use_exact_match or is_exact_match(ts.description, clean_query):
                    seen_ids.add(ts.id)
                    combined_results.append(ts)

        paginated_results = combined_results[:limit]
        total_count = len(paginated_results)
            
        return (paginated_results, total_count) if return_total_count else paginated_results

    except Exception as e:
        print(f"Error during search: {e}") 
        return [] if not return_total_count else ([], 0)

def search_ts_for_selection(query="", limit=None, field_filter=None):
    try:
        results, total_count = search_ts(query, limit=limit, return_total_count=True)

        # Apply field filter if specified
        if field_filter:
            field_prefix_map = {
                "Field A": "KASAWARI",
                "Field B": "ANGSI"
            }
            prefix = field_prefix_map.get(field_filter, "")
            if prefix:
                results = [ts for ts in results if ts.external_id and ts.external_id.startswith(prefix)]
                total_count = len(results)

        formatted_results = []
        for ts in results:
            display_name = _format_display_name(ts)

            formatted_results.append({
                'id': ts.id,
                'external_id': ts.external_id,
                'display_name': display_name,
                'name': ts.name or "N/A",
                'unit': ts.unit or "N/A",
                'description': ts.description or "N/A"
            })

        return formatted_results, total_count

    except Exception as e:
        st.error(f"Error searching time series: {e}")
        return [], 0

def show_ts_selection():
    if 'predict_selected_ts' not in st.session_state:
        st.session_state.predict_selected_ts = None
    if 'predict_search_results' not in st.session_state:
        st.session_state.predict_search_results = []
    if 'predict_search_query' not in st.session_state:
        st.session_state.predict_search_query = ""
    if 'predict_selected_field' not in st.session_state:
        st.session_state.predict_selected_field = "Field A"
    
    selected_field = st.selectbox(
        "Select Field",
        options=["Field A", "Field B"],
        index=0,  # Default to Field A
        key="predict_field_selector"
    )
    
    # Update session state and clear search if field changed
    if selected_field != st.session_state.predict_selected_field:
        st.session_state.predict_selected_field = selected_field
        st.session_state.predict_search_results = []
        st.session_state.predict_search_query = ""
        st.session_state.predict_selected_ts = None

    # Search interface
    st.text_input(
        "Search Tag Type",
        key="predict_ts_search",
        placeholder="Type keyword or use \"exact phrase\" for exact matching...",
        on_change=lambda: setattr(st.session_state, 'predict_search_triggered', True),
        help="Use quotes around your search term for exact matching (e.g., \"exact phrase\")"
    )
    
    search_query = st.session_state.predict_ts_search
    
    # Trigger search if query changed
    if search_query and search_query.strip() and (
        st.session_state.get('predict_search_triggered') or 
        (search_query != st.session_state.get('predict_search_query'))
    ):
        st.session_state.predict_search_triggered = False
        st.session_state.predict_search_query = search_query
        
        with st.spinner("Searching..."):
            results, total_count = search_ts_for_selection(
                search_query, 
                field_filter=st.session_state.predict_selected_field
            )
            st.session_state.predict_search_results = results
            if results:
                # Show if exact matching is active from quotes
                is_quoted = search_query.strip().startswith('"') and search_query.strip().endswith('"')
                if is_quoted:
                    st.caption(f"Found **{total_count}** exact matches.")
                else:
                    st.caption(f"Found **{total_count}** results.")
            else:
                is_quoted = search_query.strip().startswith('"') and search_query.strip().endswith('"')
                match_type = "exact matches" if is_quoted else "results"
                st.caption(f"No {match_type} found.")
    
    # Clear results if search box is empty
    elif not search_query.strip() and st.session_state.predict_search_results:
        st.session_state.predict_search_results = []
        st.session_state.predict_search_query = ""
        st.session_state.predict_selected_ts = None
        st.session_state.predict_selectbox_ts = None
    
    if st.session_state.predict_search_results:
        with st.container(border=False):
            available_ts = st.session_state.predict_search_results
            options = {ts.get('external_id'): ts.get('display_name', 'N/A') for ts in available_ts if ts.get('external_id')}
            
            selected_external_id = st.selectbox(
                "Select Tag",
                options=list(options.keys()),
                format_func=lambda x: options[x],
                index=None,
                placeholder="Choose a tag...",
                key="predict_selectbox_ts"
            )
            
            if selected_external_id:
                st.session_state.predict_selected_ts = selected_external_id
                # Show selected time series info
                selected_ts_info = next((ts for ts in available_ts if ts['external_id'] == selected_external_id), None)
                return selected_external_id
    
    return st.session_state.predict_selected_ts

def main(time_series_external_id=None, start_time=None, end_time="now", forecast_steps=None, freq=None):    
    with st.sidebar:
        st.title("CDF 4Sight")
        st.divider()
        
        if not time_series_external_id:
            time_series_external_id = show_ts_selection()
        
        # Check if a tag is selected to enable/disable other fields
        tag_selected = bool(time_series_external_id)
        
        # Fallback to default if no selection made
        if not time_series_external_id:
            time_series_external_id = "KASAWARI_PI_PC.SKA.KSCPP.DCS.SW.22PIA-0028.PV"
        
        st.divider()
        
        model_names = list(AVAILABLE_MODELS.keys())
        if FORECAST_MODE_KEY not in st.session_state:
            st.session_state[FORECAST_MODE_KEY] = "Balanced"
        current_forecast_mode = _sync_forecast_mode_state(model_names, update_display_key=False)
        forecast_mode_options = list(SELECTABLE_FORECAST_MODES)
        if current_forecast_mode == CUSTOM_FORECAST_MODE:
            forecast_mode_options.append(CUSTOM_FORECAST_MODE)

        if FORECAST_MODE_PENDING_KEY in st.session_state:
            st.session_state[FORECAST_MODE_DISPLAY_KEY] = st.session_state.pop(FORECAST_MODE_PENDING_KEY)
        if FORECAST_MODE_DISPLAY_KEY not in st.session_state:
            st.session_state[FORECAST_MODE_DISPLAY_KEY] = current_forecast_mode
        if st.session_state.get(FORECAST_MODE_DISPLAY_KEY) not in forecast_mode_options:
            st.session_state[FORECAST_MODE_DISPLAY_KEY] = current_forecast_mode

        selected_forecast_mode = st.selectbox(
            "Analysis Mode",
            options=forecast_mode_options,
            key=FORECAST_MODE_DISPLAY_KEY,
            on_change=_on_forecast_mode_change,
            help=(
                "**Fast** is suitable for exploratory analysis, using Random Forest with 30 optimization trials.\n "
                "**Balanced** balances speed and performance, using Random Forest and XGBoost with 50 trials each, then chooses the best.\n "
                "**Performance** uses LSTM for best accuracy, while it takes time and is resource-intensive."
            ),
        )
        forecast_mode = st.session_state.get(FORECAST_MODE_KEY, current_forecast_mode)

        selected_models = list(st.session_state.get(SELECTED_MODELS_KEY, []))
        ensemble_method = st.session_state.get(ENSEMBLE_METHOD_KEY)
        use_optuna = bool(st.session_state.get(USE_OPTUNA_KEY, False))
        optuna_n_trials = int(st.session_state.get(OPTUNA_TRIALS_KEY, 30))
        optuna_timeout = int(st.session_state.get(OPTUNA_TIMEOUT_KEY, 180))
        
        # Advanced Options
        with st.expander("Advanced Options", expanded=False):
            time_period_options = {
                "90 days ago": 90,
                "180 days ago": 180, 
                "1 year ago": 365,
                "2 years ago": 730
            }
            
            if not start_time:
                selected_period = st.selectbox(
                    "Training Period", 
                    options=list(time_period_options.keys()),
                    index=2,  # Default: 1 year ago
                    key=TRAINING_PERIOD_KEY,
                    disabled=not tag_selected
                )
                start_time = time_period_options[selected_period]
            end_time = "now"
            
            timespan_options = {
                "3 days ahead": 3,
                "7 days ahead": 7,
                "14 days ahead": 14,
                "1 month ahead": 30,
                "2 months ahead": 60,
                "3 months ahead": 90,
                "6 months ahead": 180
            }
            
            selected_timespan = st.selectbox(
                "Forecast Timespan", 
                options=list(timespan_options.keys()),
                index=4,
                key=FORECAST_TIMESPAN_KEY,
                disabled=not tag_selected
            )
            
            if not freq:
                freq = st.selectbox(
                    "Interval", 
                    ["1h", "30min", "15min", "1d"], 
                    index=0,
                    key=INTERVAL_KEY,
                    disabled=not tag_selected
                )

            if not forecast_steps:
                # Calculate estimated forecast steps based on frequency and timespan
                timespan_days = timespan_options[selected_timespan]
                
                # Convert frequency to steps per day
                freq_to_steps_per_day = {
                    "1h": 24,      # 24 hours
                    "30min": 48,   # 48 half-hours
                    "15min": 96,   # 96 quarter-hours
                    "1d": 1        # 1 day
                }
            
            steps_per_day = freq_to_steps_per_day[freq]
            forecast_steps = timespan_days * steps_per_day
        
            st.caption(f"To generate **{forecast_steps:,}** forecast steps.")

            # Model Selection
            selected_models = st.multiselect(
                "Select Model(s)",
                options=model_names,
                key=SELECTED_MODELS_KEY,
                help="Select one or more models for forecasting.\nProphet: Time series forecasting with seasonality.\nXGBoost: Gradient boosting for time series.\nRandom Forest: Ensemble of decision trees for time series.\nSVM: Support Vector Machine regression for nonlinear patterns.\nLSTM: Deep learning recurrent neural network for time series."
            )
            
            # Ensemble options when multiple models selected
            if len(selected_models) > 1:
                if st.session_state.get(ENSEMBLE_METHOD_KEY) not in ["Average Results", "Choose Best"]:
                    st.session_state[ENSEMBLE_METHOD_KEY] = "Average Results"
                ensemble_method = st.radio(
                    "Ensemble Method",
                    options=["Average Results", "Choose Best"],
                    key=ENSEMBLE_METHOD_KEY,
                    help="Average Results: Combine predictions from all models\nChoose Best: Use model with lowest error"
                )
            else:
                ensemble_method = None
            
            # Optuna Hyperparameter Optimization
            use_optuna = st.checkbox(
                "Extra Optimization",
                key=USE_OPTUNA_KEY,
                help="Use Bayesian optimization to find best hyperparameters. Increases training time but may improve accuracy."
            )
            
            if use_optuna:
                with st.expander("Optimization", expanded=True):
                    optuna_n_trials = st.slider(
                        "Optimization Trials",
                        min_value=10,
                        max_value=100,
                        step=10,
                        key=OPTUNA_TRIALS_KEY,
                        help="More trials = better optimization but longer time"
                    )
                    optuna_timeout = st.slider(
                        "Timeout (seconds)",
                        min_value=60,
                        max_value=600,
                        step=30,
                        key=OPTUNA_TIMEOUT_KEY,
                        help="Maximum time for optimization per model"
                    )

        post_advanced_forecast_mode = _sync_forecast_mode_state(
            model_names,
            update_display_key=False,
        )
        if post_advanced_forecast_mode != st.session_state.get(FORECAST_MODE_DISPLAY_KEY):
            st.session_state[FORECAST_MODE_PENDING_KEY] = post_advanced_forecast_mode
            st.rerun()
        
        run_analysis = st.button(
            "Start", 
            type="primary",
            disabled=not tag_selected
        )

    analysis_state_key = f"predict_analysis_payload::{time_series_external_id}"
    cached_analysis = st.session_state.get(analysis_state_key)

    if run_analysis or cached_analysis is not None:
        use_cached_analysis = (not run_analysis) and (cached_analysis is not None)
        asset_metadata_table = None

        if use_cached_analysis:
            description = cached_analysis.get("description", f"{time_series_external_id}")
            unit = cached_analysis.get("unit", "")
            asset_metadata_table = cached_analysis.get("asset_metadata_table")
        else:
            client = client_gen('dev')
            retriever = RetrieveData(client)
            ts_metadata = retriever.retrieve_time_series_metadata(time_series_external_id)
            description = ts_metadata.description if ts_metadata and ts_metadata.description else f"{time_series_external_id}"
            unit = ts_metadata.unit if ts_metadata and ts_metadata.unit else ""

            metadata_records = [
                {"Field": "external_id", "Value": time_series_external_id},
                {"Field": "name", "Value": ts_metadata.name if ts_metadata and ts_metadata.name else "N/A"},
                {"Field": "description", "Value": description},
                {"Field": "unit", "Value": unit if unit else "N/A"},
                {"Field": "asset_id", "Value": ts_metadata.asset_id if ts_metadata and ts_metadata.asset_id else "N/A"},
                {"Field": "data_set_id", "Value": ts_metadata.data_set_id if ts_metadata and ts_metadata.data_set_id else "N/A"},
                {"Field": "is_string", "Value": ts_metadata.is_string if ts_metadata else "N/A"},
                {"Field": "is_step", "Value": ts_metadata.is_step if ts_metadata else "N/A"},
                {"Field": "selected_start_time", "Value": start_time},
                {"Field": "selected_end_time", "Value": end_time},
                {"Field": "selected_granularity", "Value": freq},
            ]
            ts_custom_metadata = ts_metadata.metadata if ts_metadata and ts_metadata.metadata else {}
            for key in sorted(ts_custom_metadata):
                metadata_records.append({"Field": f"metadata.{key}", "Value": ts_custom_metadata[key]})
            asset_metadata_table = pd.DataFrame(metadata_records)

        st.subheader(f"Forecast: {description}")
        st.caption(f"{time_series_external_id}")

        progress_divider = st.empty()
        progress_divider.divider()

        progress_bar = st.progress(0)
        status_text = st.empty()
        pipeline_duration_seconds = None
        
        try:
            if use_cached_analysis:
                status_text.caption("Loaded cached analysis results.")
                progress_bar.progress(100)

                selected_models = cached_analysis.get("selected_models", selected_models)
                ensemble_method = cached_analysis.get("ensemble_method", "Average Results")
                model_results = cached_analysis["model_results"]
                best_model = cached_analysis.get("best_model", selected_models[0] if selected_models else "Prophet")
                df_prepared = cached_analysis["df_prepared"]
                mae = cached_analysis["mae"]
                rmse = cached_analysis["rmse"]
                mape = cached_analysis["mape"]
                forecast_horizon = cached_analysis["forecast_horizon"]
                lower_ci = cached_analysis["lower_ci"]
                upper_ci = cached_analysis["upper_ci"]
                model_fit = cached_analysis.get("model_fit")
                ttf_all_dict = cached_analysis["ttf_all_dict"]
                df_raw = cached_analysis.get("df_raw")
                df_raw_cdf = cached_analysis.get("df_raw_cdf")
                df_clean = cached_analysis["df_clean"]
                df_marked = cached_analysis["df_marked"]
                sigma_stats = cached_analysis["sigma_stats"]
                crossing_counts = cached_analysis["crossing_counts"]
                rate_comparison = cached_analysis["rate_comparison"]
                removed_count = cached_analysis["removed_count"]
                total_count = cached_analysis["total_count"]
                plt_forecast = cached_analysis["plt_forecast"]
                pipeline_duration_seconds = cached_analysis.get("pipeline_duration_seconds")
                cached_start_time = cached_analysis.get("start_time", start_time)
                cached_end_time = cached_analysis.get("end_time", end_time)
                if df_raw is None:
                    df_raw = df_clean.copy()
                if df_raw_cdf is None:
                    try:
                        client = client_gen('dev')
                        retriever = RetrieveData(client)
                        df_raw_cdf = retriever.retrieve_raw_data(
                            time_series_external_id,
                            cached_start_time,
                            cached_end_time
                        )
                    except Exception:
                        st.warning("Raw datapoints could not be retrieved from CDF API. Raw export will be empty.")
                        df_raw_cdf = pd.DataFrame(columns=["Timestamp", "Value"])
                if asset_metadata_table is None or asset_metadata_table.empty:
                    asset_metadata_table = pd.DataFrame([
                        {"Field": "external_id", "Value": time_series_external_id},
                        {"Field": "description", "Value": description},
                        {"Field": "unit", "Value": unit if unit else "N/A"},
                    ])
            else:
                client = client_gen('dev')
                retriever = RetrieveData(client)
                pipeline_start_time = perf_counter()

                # Step 1: Data Retrieval
                status_text.caption("Retrieving data from Cognite Data Fusion...")
                progress_bar.progress(10)
                try:
                    df_raw_cdf = retriever.retrieve_raw_data(time_series_external_id, start_time, end_time)
                except Exception:
                    st.warning("Raw datapoints could not be retrieved from CDF API. Raw export will be empty.")
                    df_raw_cdf = pd.DataFrame(columns=["Timestamp", "Value"])
                df_raw = retriever.retrieve_data(time_series_external_id, start_time, end_time, "interpolation", freq)

                # Step 2: Data Cleaning
                status_text.caption("Cleaning data and removing outliers...")
                progress_bar.progress(20)
                df_clean, removed_count, total_count = data_clean(df_raw, "Value")

                # Step 3: Sigma Analysis
                status_text.caption("Calculating sigma statistics...")
                progress_bar.progress(30)
                df_marked, sigma_stats = calculate_sigma_median(df_clean)
                crossing_counts = count_threshold_crossings(df_marked)

                # Step 4: Rate of Change Analysis
                status_text.caption("Analyzing rate of change...")
                progress_bar.progress(40)
                rate_comparison = compare_multiple_periods(df_clean, convert_to='per_day')

                # Step 5: Model Training and Evaluation
                status_text.caption("Preparing data for modeling...")
                progress_bar.progress(50)
                model_results = {}

                if not selected_models:
                    selected_models = list(
                        FORECAST_MODE_PRESETS.get(forecast_mode, FORECAST_MODE_PRESETS["Balanced"])["selected_models"]
                    )

                for i, model_name in enumerate(selected_models):
                    model_config = AVAILABLE_MODELS[model_name]
                    status_text.caption(f"Training ({i+1}/{len(selected_models)})...")
                    model_start_time = perf_counter()

                    df_prepared, resampled = model_config["prepare_data"](df_clean)
                    test_steps = int(len(df_prepared) * 0.2)
                    train_data = df_prepared[:-test_steps]
                    test_data = df_prepared[-test_steps:]

                    mae, rmse, mape, y_pred = model_config["evaluate_model"](train_data, test_data)

                    evaluation_df = pd.DataFrame()
                    if y_pred is not None and not test_data.empty:
                        y_pred_array = np.asarray(y_pred).reshape(-1)
                        eval_len = min(len(test_data), len(y_pred_array))
                        evaluation_df = pd.DataFrame({
                            "ds": test_data["ds"].iloc[:eval_len].values,
                            "actual": test_data["y"].iloc[:eval_len].values,
                            "predicted": y_pred_array[:eval_len],
                        })
                        evaluation_df["residual"] = evaluation_df["actual"] - evaluation_df["predicted"]
                        evaluation_df["absolute_error"] = evaluation_df["residual"].abs()

                    progress_bar.progress(60 + (i * 10))
                    if use_optuna:
                        status_text.caption("Optimizing and training model...")
                    else:
                        status_text.caption("Generating forecasts...")

                    forecast_horizon, lower_ci, upper_ci, model_fit = model_config["train_model"](
                        df_prepared,
                        forecast_steps=forecast_steps,
                        freq=freq,
                        use_optuna=use_optuna,
                        optuna_n_trials=optuna_n_trials,
                        optuna_timeout=optuna_timeout,
                    )

                    if forecast_horizon is not None:
                        ttf_dict = model_config["estimate_ttf"](
                            forecast=forecast_horizon,
                            df_clean=df_clean,
                            freq=freq
                        )
                    else:
                        st.warning(f"**{model_name}** training failed — skipping TTF estimation.")
                        ttf_dict = {}

                    model_runtime_seconds = perf_counter() - model_start_time

                    model_results[model_name] = {
                        "df_prepared": df_prepared,
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape,
                        "y_pred": y_pred,
                        "evaluation_df": evaluation_df,
                        "forecast": forecast_horizon,
                        "lower_ci": lower_ci,
                        "upper_ci": upper_ci,
                        "model_fit": model_fit,
                        "ttf_dict": ttf_dict,
                        "runtime_seconds": model_runtime_seconds,
                        "composite_score": None
                    }

                progress_bar.progress(80)
                status_text.caption("Processing model results...")

                if len(selected_models) == 1:
                    best_model = selected_models[0]
                    final_result = model_results[best_model]
                    df_prepared = final_result["df_prepared"]
                    mae = final_result["mae"]
                    rmse = final_result["rmse"]
                    mape = final_result["mape"]
                    forecast_horizon = final_result["forecast"]
                    lower_ci = final_result["lower_ci"]
                    upper_ci = final_result["upper_ci"]
                    model_fit = final_result["model_fit"]
                    ttf_all_dict = final_result["ttf_dict"]
                else:
                    data_std = df_clean['Value'].std()
                    for model_name, result in model_results.items():
                        if result["mape"] is not None:
                            score, _, _ = performance_score(
                                result["mape"], result["rmse"], result["mae"], data_std
                            )
                            result["composite_score"] = score
                        else:
                            result["composite_score"] = 0

                    if ensemble_method == "Average Results":
                        forecasts = [r["forecast"] for r in model_results.values() if r["forecast"] is not None]
                        if forecasts:
                            forecast_df = pd.DataFrame({name: r["forecast"] for name, r in model_results.items() if r["forecast"] is not None})
                            forecast_horizon = forecast_df.mean(axis=1)
                            forecast_horizon.name = 'forecast'

                            lower_df = pd.DataFrame({name: r["lower_ci"] for name, r in model_results.items() if r["lower_ci"] is not None})
                            lower_ci = lower_df.mean(axis=1)
                            lower_ci.name = 'lower'

                            upper_df = pd.DataFrame({name: r["upper_ci"] for name, r in model_results.items() if r["upper_ci"] is not None})
                            upper_ci = upper_df.mean(axis=1)
                            upper_ci.name = 'upper'

                            valid_results = [r for r in model_results.values() if r["mae"] is not None]
                            mae = np.mean([r["mae"] for r in valid_results])
                            rmse = np.mean([r["rmse"] for r in valid_results])
                            mape = np.mean([r["mape"] for r in valid_results])

                            df_prepared = list(model_results.values())[0]["df_prepared"]
                            model_fit = None

                            from models.prophet_model import estimate_ttf as combined_estimate_ttf
                            ttf_all_dict = combined_estimate_ttf(
                                forecast=forecast_horizon,
                                df_clean=df_clean,
                                freq=freq
                            )

                            best_model = "Ensemble (Averaged)"
                        else:
                            st.error("No valid forecasts from selected models")
                            return
                    else:
                        best_model = max(model_results.keys(), key=lambda m: model_results[m]["composite_score"] or 0)
                        final_result = model_results[best_model]
                        df_prepared = final_result["df_prepared"]
                        mae = final_result["mae"]
                        rmse = final_result["rmse"]
                        mape = final_result["mape"]
                        forecast_horizon = final_result["forecast"]
                        lower_ci = final_result["lower_ci"]
                        upper_ci = final_result["upper_ci"]
                        model_fit = final_result["model_fit"]
                        ttf_all_dict = final_result["ttf_dict"]

                progress_bar.progress(90)
                status_text.caption("Finalizing analysis...")

                df_recent_series = pd.Series(
                    df_prepared['y'].values,
                    index=df_prepared['ds'].values
                )

                if len(selected_models) == 1:
                    model_title_suffix = f"({selected_models[0]})"
                elif ensemble_method == "Average Results":
                    model_title_suffix = f"(Ensemble: {', '.join(selected_models)})"
                else:
                    model_title_suffix = f"(Best: {best_model})"

                plt_forecast = plot_forecast(
                    df_recent=df_recent_series,
                    forecast=forecast_horizon,
                    lower=lower_ci,
                    upper=upper_ci,
                    sigma_stats=sigma_stats,
                    ttf_dict=ttf_all_dict,
                    title=f"{description} Actual + Forecast {model_title_suffix}",
                    show_ci=False,
                )
                pipeline_duration_seconds = perf_counter() - pipeline_start_time

                st.session_state[analysis_state_key] = {
                    "description": description,
                    "unit": unit,
                    "forecast_mode": forecast_mode,
                    "selected_models": selected_models,
                    "ensemble_method": ensemble_method if 'ensemble_method' in locals() else "Average Results",
                    "use_optuna": use_optuna,
                    "optuna_n_trials": optuna_n_trials,
                    "optuna_timeout": optuna_timeout,
                    "model_results": model_results,
                    "best_model": best_model if 'best_model' in locals() else None,
                    "df_prepared": df_prepared,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "forecast_horizon": forecast_horizon,
                    "lower_ci": lower_ci,
                    "upper_ci": upper_ci,
                    "model_fit": model_fit,
                    "ttf_all_dict": ttf_all_dict,
                    "df_raw": df_raw,
                    "df_raw_cdf": df_raw_cdf,
                    "df_clean": df_clean,
                    "df_marked": df_marked,
                    "sigma_stats": sigma_stats,
                    "crossing_counts": crossing_counts,
                    "rate_comparison": rate_comparison,
                    "removed_count": removed_count,
                    "total_count": total_count,
                    "asset_metadata_table": asset_metadata_table,
                    "start_time": start_time,
                    "end_time": end_time,
                    "freq": freq,
                    "plt_forecast": plt_forecast,
                    "pipeline_duration_seconds": pipeline_duration_seconds,
                }

            pipeline_duration_display = (
                f"{pipeline_duration_seconds:.2f}s"
                if pipeline_duration_seconds is not None
                else "n/a"
            )
            
            tab1, tab2, tab3 = st.tabs(["Forecast Results", "Model Details", "Pipeline Details"])
            
            with tab1:
                # Performance metrics
                data_std = df_clean['Value'].std()
                score, category, color = performance_score(mape, rmse, mae, data_std)
                
                st.plotly_chart(plt_forecast, use_container_width=True)

                # Traffic light performance indicator
                performance_container = st.container()
                with performance_container:
                    if color == "success":
                        st.success(
                            f"**{category}.** "
                            f"The forecast model shows reliable performance with low prediction errors."
                        )
                    elif color == "warning":
                        st.warning(
                            f"**{category}.** "
                            f"The forecast model shows moderate performance for this tag. Results should be interpreted with discretion."
                        )
                    else:
                        st.error(
                            f"**{category}.** "
                            f"The forecast model shows poor performance with high prediction errors for this tag. "
                            f"Consider using a longer historical period, checking data quality, or "
                            f"verifying if the time series has sufficient patterns for forecasting."
                        )
                
                # TTF Results
                if ttf_all_dict:
                    now = datetime.now()
                    ttf_items = []
                    for label, ts in ttf_all_dict.items():
                        if ts:
                            time_diff = ts - now
                            days = time_diff.days
                            hours, remainder = divmod(time_diff.seconds, 3600)
                            minutes, _ = divmod(remainder, 60)
                            time_to_ttf = f"{days}d {hours}h {minutes}m"
                            ts_formatted = ts.strftime('%d %B %Y - %H:%M')
                            # Add rows where TTF timestamp and Time to TTF are not null
                            ttf_items.append((label, ts_formatted, time_to_ttf))

                    # df only with non-null TTF
                    if ttf_items:
                        ttf_df = pd.DataFrame(ttf_items, columns=['Threshold', 'Threshold Timestamp', 'Time to Threshold'])
                        st.dataframe(ttf_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No threshold breaches predicted in the forecast period.")

            with tab2:
                st.write("**Model Performance**")
                
                if len(selected_models) > 1 and model_results:
                    comparison_data = []
                    for model_name, result in model_results.items():
                        if result["mae"] is not None:
                            data_std_comp = df_clean['Value'].std()
                            data_var_comp = data_std_comp ** 2
                            nmse_comp = (result['rmse'] ** 2) / data_var_comp if data_var_comp > 0 else float('inf')
                            runtime_seconds = result.get("runtime_seconds")
                            runtime_display = f"{runtime_seconds:.2f}s" if runtime_seconds is not None else "n/a"
                            score, category, color = performance_score(
                                result["mape"], result["rmse"], result["mae"], data_std_comp
                            )
                            comparison_data.append({
                                "Model": model_name,
                                "Runtime": runtime_display,
                                "MAE": f"{result['mae']:.3f}",
                                "RMSE": f"{result['rmse']:.3f}",
                                "MAPE": f"{result['mape']:.3%}",
                                "NMSE": f"{nmse_comp:.4f}",
                                "Score": f"{score:.1f}%",
                                "Performance": category.replace(" Forecasting Model Performance", "")
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Show which model/method was used
                        if ensemble_method == "Average Results":
                            st.info(f"**Ensemble Method:** Results averaged across all {len(selected_models)} models.")
                        else:
                            st.success(f"**Best Model Selected:** {best_model} (Highest composite score)")
                        
                        # Model-specific details expander
                        with st.expander("Individual Model Forecasts", expanded=False):
                            for model_name, result in model_results.items():
                                if result["forecast"] is not None:
                                    st.write(f"**{model_name}**")
                                    col1, col2, col3, col4 = st.columns(4)
                                    model_runtime_seconds = result.get("runtime_seconds")
                                    model_runtime_display = (
                                        f"{model_runtime_seconds:.2f}s"
                                        if model_runtime_seconds is not None
                                        else "n/a"
                                    )
                                    with col1:
                                        st.metric("Forecast Min", f"{result['forecast'].min():.2f}")
                                    with col2:
                                        st.metric("Forecast Max", f"{result['forecast'].max():.2f}")
                                    with col3:
                                        st.metric("Forecast Mean", f"{result['forecast'].mean():.2f}")
                                    with col4:
                                        st.metric(
                                            "Model Runtime",
                                            model_runtime_display,
                                            help="Per-model runtime for data prep, evaluation, training, and forecast generation.",
                                        )
                else:
                    data_std_single = df_clean['Value'].std()
                    data_var_single = data_std_single ** 2
                    nmse_single = (rmse ** 2) / data_var_single if data_var_single > 0 else float('inf')
                    score_single, category_single, _ = performance_score(mape, rmse, mae, data_std_single)
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                        st.metric("MAE", f"{mae:.3f}", help="Mean Absolute Error")
                    with col2:
                        st.metric("RMSE", f"{rmse:.3f}", help="Root Mean Square Error")
                    with col3:
                        st.metric("NMSE", f"{nmse_single:.3f}", help="Normalised Mean Squared Error")
                    with col4:
                        st.metric("MAPE", f"{mape:.3%}", help="Mean Absolute Percentage Error")
                    with col5:
                        st.metric("Composite Score", f"{score_single:.1f}%", help=category_single)
                    with col6:
                        st.metric(
                            "Total Processing Time",
                            pipeline_duration_display,
                            help="End-to-end runtime from data retrieval to completed pipeline output.",
                        )

                st.divider()

                for model_name, result in model_results.items():
                    evaluation_df = result.get("evaluation_df")
                    if evaluation_df is None or evaluation_df.empty:
                        st.warning(f"{model_name}: Holdout graph unavailable.")
                        continue

                    left_accuracy_fig, right_accuracy_fig = plot_model_accuracy(
                        evaluation_df=evaluation_df,
                        model_name=model_name,
                        unit=unit,
                        mae=result["mae"],
                        rmse=result["rmse"],
                        mape=result["mape"],
                    )
                    left_col, right_col = st.columns(2)
                    with left_col:
                        st.plotly_chart(left_accuracy_fig, use_container_width=True)
                    with right_col:
                        st.plotly_chart(right_accuracy_fig, use_container_width=True)

                st.caption(
                    "These figures use the final 20% of the prepared series as an out-of-sample holdout set. "
                    "Left: actual vs predicted over time. Right: predicted vs actual against the ideal 45-degree line."
                )

            with tab3:
                st.write("**Data Cleaning**")
                col1, col2, col3 = st.columns(3)
                percent_removed = (removed_count / total_count) * 100 if total_count else 0
                with col1:
                    st.metric("All Datapoints (Before)", f"{total_count:,}")
                with col2:
                    st.metric("Outliers Removed", f"{removed_count:,} ({percent_removed:.2f}%)")
                with col3:
                    st.metric("Clean Datapoints (After)", f"{len(df_clean):,}")

                # Rate of change
                st.divider()                
                plt_roc = plot_rate_change(df_clean, convert_to='per_day')
                st.plotly_chart(plt_roc, use_container_width=True)

                items = list(rate_comparison.items())
                num_items = len(items)
                
                for i in range(0, num_items, 3):
                    rate_cols = st.columns(3)
                    chunk = items[i:i+3]
                    for j, (period, roc) in enumerate(chunk):
                        with rate_cols[j]:
                            st.metric(f"{period} Rate", f"{roc:.4f} {unit}/day")
                
                # Threshold Analysis
                # st.plotly_chart(plt_threshold, use_container_width=True)
                
                # Statistical Analysis
                st.divider()           
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sigma Statistics**")
                    st.dataframe(sigma_stats, use_container_width=True)
                
                with col2:
                    st.write("**Historical Threshold Crossings**")
                    st.dataframe(crossing_counts, use_container_width=True)
            
            # Step 10: Prepare Export Data
            status_text.caption("Preparing export data...")
            progress_bar.progress(95)
            
            slopes_dict = compare_multiple_periods(df_clean, convert_to='per_day')
            
            # Export tables
            tbl_sigma, tbl_sigma_summary = tbl_sigma_crossing_event(df_marked, sigma_stats)
            
            # Data std (for performance score)
            data_std = df_clean['Value'].std()
            
            # Model name for summary
            if len(selected_models) == 1:
                model_name_for_summary = selected_models[0]
            elif ensemble_method == "Average Results":
                model_name_for_summary = f"Ensemble ({', '.join(selected_models)})"
            else:
                model_name_for_summary = f"{best_model} (Best)"
            
            tbl_ttf = tbl_ttf_summary(ttf_all_dict, model_name_for_summary, mae, rmse, mape, data_std)
            tbl_ttf.insert(0, "Tags", time_series_external_id)
            
            tbl_forecast = tbl_forecast_pressure(forecast_horizon, lower_ci, upper_ci, freq=freq)
            tbl_forecast["Tag"] = time_series_external_id
            
            tbl_rate = tbl_rate_change(time_series_external_id, slopes_dict)
            
            tbl_ttf_breach = tbl_ttf_breach_times(time_series_external_id, ttf_all_dict)
            tbl_historical = df_raw.copy()
            tbl_historical["Tag"] = time_series_external_id
            tbl_raw_data_cdf = df_raw_cdf.copy()
            tbl_raw_data_cdf["Tag"] = time_series_external_id
            
            progress_bar.progress(100)
            status_text.caption("Analysis completed")
            
            # Download section
            st.divider()
            st.write("**Download Results**")
            download_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="Sigma Crossing Events",
                    data=tbl_sigma.to_csv(index=False),
                    file_name=f"sigma_crossing_events_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )
                
                st.download_button(
                    label="Forecast Data",
                    data=tbl_forecast.to_csv(index=False),
                    file_name=f"forecast_data_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="Sigma Summary",
                    data=tbl_sigma_summary.to_csv(index=False),
                    file_name=f"sigma_summary_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )
                
                st.download_button(
                    label="Rate of Change",
                    data=tbl_rate.to_csv(index=False),
                    file_name=f"rate_of_change_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )
            
            with col3:
                st.download_button(
                    label="TTF Summary",
                    data=tbl_ttf.to_csv(index=False),
                    file_name=f"ttf_summary_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )
                
                st.download_button(
                    label="TTF Breach Times",
                    data=tbl_ttf_breach.to_csv(index=False),
                    file_name=f"ttf_breach_times_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )

            extra_col1, extra_col2, extra_col3 = st.columns(3)
            with extra_col1:
                st.download_button(
                    label="Historical Data (Interpolated)",
                    data=tbl_historical.to_csv(index=False),
                    file_name=f"historical_data_interpolated_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )

            with extra_col2:
                st.download_button(
                    label="Raw Data (CDF API)",
                    data=tbl_raw_data_cdf.to_csv(index=False),
                    file_name=f"raw_data_cdf_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )

            with extra_col3:
                st.download_button(
                    label="Asset Metadata",
                    data=asset_metadata_table.to_csv(index=False),
                    file_name=f"asset_metadata_{time_series_external_id}_{download_suffix}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
        
        finally:
            progress_bar.empty()
            status_text.empty()
            progress_divider.empty()
    
    else:
        st.info("Please configure parameters in the sidebar.")
        st.divider()
        st.markdown("""
        ##### About
        
        This tool provides comprehensive predictive analytics for industrial time series data:
        - Predictive analytics model for critical asset performance forecasting.
        - Time to Failure (TTF) predictions.
        - Rate of change (slope).
        - Standard deviation thresholds and statistics.
        - Outlier detection using IQR and removal.
        """)

def handle(time_series_external_id=None, start_time=None, end_time=None, forecast_steps=None, freq=None):
    main(time_series_external_id, start_time, end_time, forecast_steps, freq)

if __name__ == '__main__':
    main()
