# Helper function to write time-series directly to CDF

from cognite.client.exceptions import CogniteAPIError
from cognite.client.data_classes import TimeSeriesWrite
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

def forecast_ts(client, base_tag: str, forecast_data: pd.Series, 
                               tag_unit: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
    # Create a new timeseries in CDF for forecast data and insert (only) forecast values
    # If time series already exists, it will just insert the new forecast data

    forecast_external_id = f"4Sight_{base_tag}"
    try:
        unit = tag_unit or ""
        
        # Check if forecast timeseries already exists
        try:
            existing_ts = client.time_series.retrieve(external_id=forecast_external_id)
            print(f"Forecast timeseries {forecast_external_id} already exists, updating data only.")
            ts_exists = True
        except:
            ts_exists = False
        
        if not ts_exists:
            ts_metadata = {
                "source": "PRIME4SIGHT",
                "model_type": "Prophet",
                "forecast_horizon": "12_weeks",
                "base_tag": base_tag,
                "created_by": "automated_pipeline"
            }
            if metadata:
                ts_metadata.update(metadata)
            
            # Create ts object
            ts = TimeSeriesWrite(
                external_id=forecast_external_id,
                name=f"4Sight_{base_tag}",
                description=f"Forecast data for {base_tag} generated using 4Sight",
                unit=unit,
                metadata=ts_metadata
            )
            
            # Create ts in CDF
            created_ts = client.time_series.create(ts)
            print(f"Successfully created forecast timeseries: {forecast_external_id}")
        
        # Insert forecast nodes
        success = insert_nodes(client, forecast_external_id, forecast_data)
        
        return success
        
    except CogniteAPIError as e:
        print(f"Failed to create forecast timeseries {forecast_external_id}: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error creating forecast timeseries {forecast_external_id}: {str(e)}")
        return False

def insert_nodes(client, external_id: str, forecast_data: pd.Series) -> bool:
    # Insert forecast nodes into existing timeseries

    try:
        data_tuples = []
        for timestamp, value in forecast_data.items():
            if isinstance(timestamp, (pd.Timestamp, datetime)):
                timestamp_ms = int(timestamp.timestamp() * 1000)
            else:
                # Parse string to ms
                try:
                    parsed_timestamp = pd.to_datetime(str(timestamp))
                    timestamp_ms = int(parsed_timestamp.timestamp() * 1000)
                except Exception:
                    print(f"Skipping invalid timestamp: {timestamp}")
                    continue
            
            if pd.isna(value):
                continue
                
            data_tuples.append((timestamp_ms, float(value)))
        
        if not data_tuples:
            print(f"No valid nodes to insert for {external_id}")
            return False
        
        # Insert nodes in batches (to avoid API limit)
        batch_size = 10000
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i + batch_size]
            client.time_series.data.insert(batch, external_id=external_id)
        
        print(f"Successfully inserted {len(data_tuples)} forecast nodes to: {external_id}")
        return True
         
    except CogniteAPIError as e:
        print(f"Failed to insert forecast nodes to {external_id}: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error inserting forecast nodes to {external_id}: {str(e)}")
        return False

def get_ts_metadata(client, external_id: str) -> Dict:
    # Retrieve metadata from original time series to use for forecast time series

    try:
        ts = client.time_series.retrieve(external_id=external_id)
        if ts:
            return {
                "unit": ts.unit,
                "asset_id": ts.asset_id,
                "data_set_id": ts.data_set_id
            }
    except Exception as e:
        print(f"Could not retrieve metadata for {external_id}: {str(e)}")
    
    return {"unit": "N/A"}