from cognite.client import CogniteClient
import pandas as pd
from auth.client_gen import client_gen # TODO: For Streamlit

class RetrieveData:
    # Retrieve CDF PI tag data with aggregation method
    def __init__(self, client: CogniteClient):
        self.client = client
        self.pitag = None
        self.ext_id = None
        self.agg = None
        self.description = None
 
    def retrieve_time_series_metadata(self, ex_id):
        try:
            ts = self.client.time_series.retrieve(external_id=ex_id)
            if ts is not None:
                self.description = ts.description
            else:
                self.description = "N/A"
            return ts
        except Exception as e:
            print(f"Error retrieving time series metadata for {ex_id}: {e}")
            self.description = "N/A"
            return None

    def retrieve_data(self, ex_id, start_time_str, end_time_str, agg, interval):
        self.ext_id = ex_id
        self.agg = agg
        # Extract tag name for file saving
        raw_tag = ex_id.split(".")[-1]
        self.pitag = raw_tag.replace(".", "_")
        # Retrieve datapoints
        datapoints = self.client.time_series.data.retrieve(
            external_id = ex_id,
            start = start_time_str,
            end = end_time_str,
            aggregates = agg,
            granularity = interval
        )
        df = datapoints.to_pandas()
 
        df = df.rename(columns={df.columns[0]:"Value"})
        df = df.reset_index()
        df.columns = ["Timestamp", "Value"]
        return df

    def retrieve_raw_data(self, ex_id, start_time_str, end_time_str):
        self.ext_id = ex_id
        raw_tag = ex_id.split(".")[-1]
        self.pitag = raw_tag.replace(".", "_")

        datapoints = self.client.time_series.data.retrieve(
            external_id=ex_id,
            start=start_time_str,
            end=end_time_str
        )
        df = datapoints.to_pandas()

        if df.empty:
            return pd.DataFrame(columns=["Timestamp", "Value"])

        df = df.rename(columns={df.columns[0]: "Value"})
        df = df.reset_index()
        df.columns = ["Timestamp", "Value"]
        return df

    def save_to_csv(self, retrieved_data, file_name = None):
        output_path = file_name or f"{self.pitag}.csv"
        retrieved_data.to_csv(output_path)
        return output_path