import pandas as pd
from datetime import datetime
from calc.sigma import extract_sigma_crossing_events, combine_sigma_events
from utils.eval import performance_score
from typing import Optional

# Table join for sigma crossing events
def tbl_sigma_crossing_events(df_marked: pd.DataFrame, sigma_stats: Optional[dict] = None) -> pd.DataFrame:
    event_dict = extract_sigma_crossing_events(df_marked)
    combined_events = combine_sigma_events(event_dict)

    # Optional: reset index and sort
    combined_events = combined_events.reset_index(drop=True).sort_values("Timestamp")

    return combined_events[["Timestamp", "Value", "Type"]]

def tbl_sigma_crossing_event(df_marked: pd.DataFrame, sigma_stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Sigma crossing event records
    event_dict = extract_sigma_crossing_events(df_marked)
    combined_events = combine_sigma_events(event_dict).reset_index(drop=True).sort_values("Timestamp")
    
    event_table = combined_events[["Timestamp", "Value", "Type"]]

    sigma_summary = pd.DataFrame([{
        "Median": sigma_stats["median"],
        "MAD": sigma_stats["mad"],
        "Sigma": sigma_stats["sigma"],
        "+1σ": sigma_stats["1σ+"],
        "+2σ": sigma_stats["2σ+"],
        "+3σ": sigma_stats["3σ+"],
        "+4σ": sigma_stats["4σ+"],
        "+5σ": sigma_stats["5σ+"],
        "-1σ": sigma_stats["1σ-"],
        "-2σ": sigma_stats["2σ-"],
        "-3σ": sigma_stats["3σ-"],
        "-4σ": sigma_stats["4σ-"],
        "-5σ": sigma_stats["5σ-"],
    }])
    
    return event_table, sigma_summary

def tbl_ttf_summary(ttf_all_dict: dict,
                    model_name,
                    mae: float,
                    rmse: float,
                    mape: float,
                    data_std: Optional[float]) -> pd.DataFrame:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Handle NA for data_std
    if data_std is None:
        data_std = 0.0

    # Calculate performance score
    composite_score, category, color = performance_score(mape, rmse, mae, data_std)

    summary = {
        "Date Modelling": now,
        "Model": str(model_name),
        "Model_MAE": round(mae, 4),
        "Model_RMSE": round(rmse, 4),
        "Model_MAPE": round(mape, 4),
        "Score": round(composite_score, 2)
    }

    # Add each TTF label
    for label, ts in ttf_all_dict.items():
        summary[label] = ts

    return pd.DataFrame([summary])

def tbl_ttf_sigma(ext_id: str, ttf_sigma_dict: dict) -> pd.DataFrame:
    row = {"Tag": ext_id}
    row.update(ttf_sigma_dict)
    return pd.DataFrame([row])

def tbl_forecast_pressure(forecast: pd.Series,
                           lower: pd.Series,
                           upper: pd.Series,
                           freq: str = "1H") -> pd.DataFrame:
    forecast_index = pd.date_range(start=pd.Timestamp.now(), periods=len(forecast), freq=freq)

    # Round timestamp to HH:00
    forecast_index = forecast_index.round('H')

    df_forecast = pd.DataFrame({
        "Timestamp": forecast_index,
        "Forecast": forecast.values,
        "Lower_CI": lower.values,
        "Upper_CI": upper.values
    })

    return df_forecast

def tbl_rate_change(ext_id: str, slopes_dict: dict) -> pd.DataFrame:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    records = []
    for period, slope in slopes_dict.items():
        records.append({
            "Tag": ext_id,
            "Period": period,
            "Slope_per_Day": slope if isinstance(slope, (int, float)) else None,
            "Date_Modelling": now
        })

    return pd.DataFrame(records)

def tbl_ttf_breach_times(ext_id: str, ttf_all_dict: dict) -> pd.DataFrame:
    records = []
    for breach_type, timestamp in ttf_all_dict.items():
        records.append({
            "Tag": ext_id,
            "Breach_Type": breach_type,
            "Predicted_Timestamp": timestamp
        })
    return pd.DataFrame(records)

# Exports
def run_all_exports(
    ext_id: str,
    df_clean: pd.DataFrame,
    df_marked: pd.DataFrame,
    sigma_stats: dict,
    forecast: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    ttf_all_dict: dict,
    model_name: str,
    mae: float,
    rmse: float,
    mape: float,
    slopes_dict: dict,
    freq: str = "1h",
    data_std: Optional[float] = None
):

    # Tbl 1, 2: Sigma Crossing Events + Sigma Summary
    tbl_sigma, tbl_sigma_summary = tbl_sigma_crossing_event(df_marked, sigma_stats)
    tbl_sigma.to_csv("output/tbl_sigma_crossing_event.csv", index=False) # TODO: Comment out exports
    tbl_sigma_summary.to_csv("output/tbl_sigma_summary.csv", index=False) # TODO: Comment out exports

    # Tbl 3: TTF Summary (now uses full ttf_all_dict)
    # Calculate data_std if not provided
    if data_std is None and 'Value' in df_clean.columns:
        data_std = df_clean['Value'].std()
    
    tbl_ttf = tbl_ttf_summary(ttf_all_dict, model_name, mae, rmse, mape, data_std)
    tbl_ttf.insert(0, "Tags", ext_id)
    tbl_ttf.to_csv("output/tbl_ttf_summary.csv", index=False) # TODO: Comment out exports

    # Tbl 4: Forecast Pressure Table
    tbl_forecast = tbl_forecast_pressure(forecast, lower, upper, freq=freq)
    tbl_forecast["Tag"] = ext_id
    tbl_forecast.to_csv("output/tbl_forecast_pressure.csv", index=False) # TODO: Comment out exports

    # Tbl 5: Rate of Change
    tbl_rate = tbl_rate_change(ext_id, slopes_dict)
    tbl_rate.to_csv("output/tbl_rate_change.csv", index=False) # TODO: Comment out exports

    return tbl_sigma, tbl_sigma_summary, tbl_ttf, tbl_forecast, tbl_rate