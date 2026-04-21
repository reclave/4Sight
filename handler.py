# Engine to run 4Sight for automation functions (not used in Streamlit app)

from utils.data_ingestion import RetrieveData
from utils.data_clean import data_clean
from calc.sigma import calculate_sigma_median, count_threshold_crossings
from calc.rate_change import compare_multiple_periods
from models.prophet_model import prepare_data, train_model, estimate_ttf, estimate_ttf_roc_combined
from utils.eval import evaluate_model
from models.xgb_model import (
    prepare_data as prepare_xgb_data,
    evaluate_model as evaluate_xgb_model,
    train_model as train_xgb_model,
    estimate_ttf as estimate_xgb_ttf,
)
from utils.table_join import run_all_exports
from utils.data_modeling import write_model_output_to_dm
from utils.time_series import forecast_ts, get_ts_metadata
from utils.plot import plot_forecast # TODO: Remove before deploy to Functions
from cognite.client.exceptions import CogniteAPIError
import concurrent.futures
import threading

def single_process(tag, client, thread_lock, model_type="prophet"):
    try:
        with thread_lock:
            print(f"[{tag}] Loading...")
        
        retriever = RetrieveData(client)
        df_raw = retriever.retrieve_data(tag, "365d-ago", "now", "interpolation", "1h")
        
        tag = "PI.EXAMPLE.PV" # TODO: Remove before deploy to Functions

        with thread_lock:
            print(f"[{tag}] Loaded {len(df_raw)} data points.")
            print(f"[{tag}] Removing outliers using IQR method...")
        
        df_clean, removed_count, total_count = data_clean(df_raw, 'Value')

        percent_removed = (removed_count / total_count) * 100 if total_count else 0
        with thread_lock:
            print(f"[{tag}] Removed {removed_count} outliers out of {total_count} rows ({percent_removed:.2f}%).")

        # Calculate median, sigma, and threshold crossings
        df_marked, sigma_stats = calculate_sigma_median(df_clean)
        crossing_counts = count_threshold_crossings(df_marked)

        with thread_lock:
            print(f"[{tag}] Sigma Statistics")
            print(sigma_stats)
            print(f"[{tag}] Threshold Crossings")
            print(crossing_counts)
            print(f"[{tag}] Rate of Change (Slope)")
        
        rate_comparison = compare_multiple_periods(df_clean, convert_to='per_day')
        with thread_lock:
            for period, roc in rate_comparison.items():
                print(f"[{tag}] {period} Rate of Change (Slope): {roc:.4f} barg/day")

        model_type = (model_type or "prophet").lower()
        estimate_ttf_fn = estimate_ttf
        if model_type == "xgboost":
            df_prepared, _ = prepare_xgb_data(df_clean, freq="1H")
            test_steps = max(1, int(len(df_prepared) * 0.2))
            train_data = df_prepared[:-test_steps]
            test_data = df_prepared[-test_steps:]
            mae, rmse, mape, y_pred = evaluate_xgb_model(train_data, test_data)
            with thread_lock:
                print(f"[{tag}] (XGBoost) MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4%}")

            forecast_horizon, lower_ci, upper_ci, model_fit = train_xgb_model(
                df_prepared,
                forecast_steps=2016,
                freq="1h",
            )
            model_name = "XGBoost"
            estimate_ttf_fn = estimate_xgb_ttf
        else:
            # Step 1: Prepare data
            df_prepared, resampled = prepare_data(df_clean)

            # Step 2: Evaluate Model
            test_steps = max(1, int(len(df_prepared) * 0.2)) # 20% data for testing
            train_data = df_prepared[:-test_steps]
            test_data = df_prepared[-test_steps:]
            mae, rmse, mape, y_pred = evaluate_model(train_data, test_data)
            with thread_lock:
                print(f"[{tag}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4%}")

            # Step 3: Train model and forecast (2016 - 1-hour intervals for 12 weeks)
            forecast_horizon, lower_ci, upper_ci, model_fit = train_model(df_prepared, forecast_steps=2016, freq="1h")
            model_name = "Prophet"

        # Step 3: Estimate TTF
        ttf_all_dict = estimate_ttf_fn(
            forecast=forecast_horizon,
            df_clean=df_clean,
            freq="1h"
        )
        
        # Step 4: Estimate TTF ROC (Rate of Change)
        ttf_roc_dict = estimate_ttf_roc_combined(
            df=df_clean,
            periods=[14, 30, 60]
        )
        
        # Merge TTF dictionaries
        ttf_all_dict.update(ttf_roc_dict)

        # Compute sigma stats
        _, sigma_stats = calculate_sigma_median(df_clean)
        
        # Create slope dictionary
        slopes_dict = compare_multiple_periods(df_clean, convert_to='per_day')
        
        # Calculate data standard deviation for performance score
        data_std = df_clean['Value'].std()

        # Generate data model for output
        data_model = run_all_exports(
            ext_id=retriever.ext_id,          # External tag ID
            df_clean=df_clean,                # Cleaned data after outlier removal
            df_marked=df_marked,              # Marked data with sigma levels
            sigma_stats=sigma_stats,          # Sigma statistics dictionary
            forecast=forecast_horizon,        # Forecast series
            lower=lower_ci,                   # Lower bound confidence interval
            upper=upper_ci,                   # Upper bound confidence interval
            ttf_all_dict=ttf_all_dict,        # Dictionary of all TTF threshold breach times
            model_name=model_name,            # Model type used for forecasting
            mae=mae,                          # Model evaluation MAE
            rmse=rmse,                        # Model evaluation RMSE
            mape=mape,                        # Model evaluation MAPE
            slopes_dict=slopes_dict,          # Dictionary of slopes for different periods
            freq="1h",                        # Forecast frequency
            data_std=data_std                 # Standard deviation for performance score
        )
        
        with thread_lock:
            print(f"[{tag}] Data model generated successfully.")
            print(f"[{tag}] Creating forecast timeseries in CDF...")
        
        # Get base (original) timeseries metadata
        base_metadata = get_ts_metadata(client, tag)
        
        # Create forecast ts and insert datapoints
        forecast_success = forecast_ts(
            client = client,
            base_tag = tag,
            forecast_data = forecast_horizon,
            tag_unit = base_metadata.get("unit"),
            metadata = {
                "asset_id": base_metadata.get("asset_id"),
                "data_set_id": base_metadata.get("data_set_id")
            }
        )
        
        if forecast_success:
            with thread_lock:
                print(f"[{tag}] Forecast timeseries created and populated successfully.")
        else:
            with thread_lock:
                print(f"[{tag}] Warning: Failed to create forecast timeseries.")
        
        return {
            "tag": tag, 
            "data_model": data_model, 
            "status": "success",
            "forecast_timeseries_created": forecast_success
        }
        
    except Exception as e:
        with thread_lock:
            print(f"[{tag}] Error processing tag: {str(e)}")
        return {"tag": tag, "data_model": None, "status": "error", "error": str(e)}

def handle(data=[], client=None, secrets=None, function_call_info=None):
    # For local testing purposes
    if data is None and client is None and secrets is None and function_call_info is None:
        try:
            from auth.client_gen import client_gen # TODO: Remove before deploying to Cognite Functions
            client = client_gen('dev') # TODO: Remove before deploying to Cognite Functions
            tags = ["PI.EXAMPLE.PV"] 
            print(f"Running locally with tag: {tags[0]}")
        except Exception as e:
            print(f"Failed to initialize CDF client: {str(e)}")
            return {"error": f"Failed to initialize CDF client: {str(e)}"}
        model_type = "prophet"
    else:
        # Provide tags in 'data' argument in Functions, e.g., {"tags": ["tag1", "tag2"]}
        data = data or {}
        tags = data.get("tags")
        if not tags:
            print("No tags provided in the input data.")
            return {"error": "No tags provided."}
        model_type = data.get("model_type", "prophet")
      
    print(f"Processing {len(tags)} tags...")
    
    thread_lock = threading.Lock()
    
    result_data_models = []
    processing_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        future_to_tag = {
            executor.submit(single_process, tag, client, thread_lock, model_type): tag 
            for tag in tags
        }
        
        for future in concurrent.futures.as_completed(future_to_tag):
            tag = future_to_tag[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    result_data_models.append(result['data_model'])
                    processing_results.append({
                        "tag": result['tag'],
                        "status": "success",
                        "message": f"Successfully processed tag {result['tag']}"
                    })
                    # TODO write to data modeling
                    print(f"Writing {result['tag']} results to Cognite Data Model.")
                    try:
                        write_result = write_model_output_to_dm(client, result["data_model"])
                        if write_result:
                            processing_results[-1]["write_status"] = "success"
                            processing_results[-1]["nodes_written"] = len(write_result.nodes)
                        else:
                            processing_results[-1]["write_status"] = "no_data"
                    except CogniteAPIError as e: 
                        processing_results[-1]["write_status"] = "error"
                        processing_results[-1]["write_error"] = str(e)
                        print(f"Failed to write result of {result['tag']} to CDF. Check data model.")
                else:
                    processing_results.append({
                        "tag": result['tag'],
                        "status": "error",
                        "error": result.get('error', 'Unknown error')
                    })
            except Exception as exc:
                error_msg = f"Tag {tag} generated an exception: {exc}"
                print(error_msg)
                processing_results.append({
                    "tag": tag,
                    "status": "error",
                    "error": error_msg
                })
    
    success_count = sum(1 for r in processing_results if r['status'] == 'success')
    print(f"Successfully processed {success_count} out of {len(tags)} tags.")
    
    # Return JSON serializable object
    return {
        "processed_tags": len(tags),
        "successful_tags": success_count,
        "results": processing_results
    }

handle(None, None, None, None) # TODO: Remove before deploying to Cognite Functions