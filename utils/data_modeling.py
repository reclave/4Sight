from cognite.client.data_classes.data_modeling import (SpaceApply, NodeApply, ViewId, NodeOrEdgeData)
from cognite.client.exceptions import CogniteAPIError
import pandas as pd
import time

MODEL_SPACE = "Kasawari" # Name of Cognite environment
INSTANCE_SPACE = "Kasawari"

def check_spaces(client):
    if not client.data_modeling.spaces.retrieve(MODEL_SPACE):
        space =  SpaceApply(
                space=MODEL_SPACE,
                name=MODEL_SPACE,
                description=f"Kasawari",
            )
        client.data_modeling.spaces.apply(spaces=space)
    if not client.data_modeling.spaces.retrieve(INSTANCE_SPACE):
        space =  SpaceApply(
                space=INSTANCE_SPACE,
                name=INSTANCE_SPACE,
                description=f"Kasawari",
            )
        client.data_modeling.spaces.apply(spaces=space)

def delete_nodes(client, tag):
    # Delete existing nodes for a forecast tag prior to writing new nodes
    # Overwrite all nodes since CDF cannot attach datapoints to existing time-series from SDK
    # Helper function not for app
    forecast_id = f"4Sight_{tag}"
    try:
        # Check if time series exists
        try:
            existing_ts = client.time_series.retrieve(external_id=forecast_id)
            if existing_ts:
                print(f"Overwriting existing nodes for {forecast_id}...")
                
                # Delete all nodes
                client.time_series.data.delete_range(
                    external_id=forecast_id,
                    start=0,
                    end="now"
                )
                
                print(f"Successfully overwritten existing nodes for {forecast_id}")
            else:
                print(f"Time series {forecast_id} does not exist, will be created with new data")
        except CogniteAPIError as retrieve_error:
            if "not found" in str(retrieve_error).lower():
                print(f"Time series {forecast_id} does not exist, will be created with new data")
            else:
                print(f"Warning: Could not check if time series {forecast_id} exists: {str(retrieve_error)}")
        
        return True
        
    except CogniteAPIError as e:
        print(f"Error overwriting existing nodes for {forecast_id}: {str(e)}")
        return False

def delete_instances(client, tag):
    # Delete existing data model instances for a forecast tag prior to writing new instances
    # Goes side by side with delete_nodes but to update data warehouse attributes
    # Helper function not for app
    try:
        print(f"Deleting existing data model instances for forecast tag {tag}...")
        
        view_ids = [
            ViewId("Kasawari", "Forecast_Pressure", "7548c1357c9c45"),
            # ViewId("Kasawari", "Rate_of_Change", "831adde070985b"),
            # ViewId("Kasawari", "Sigma_Summary", "cb8761a073327c"),
            # ViewId("Kasawari", "TTF_Summary_N", "f3358c8b6a0f29"),
            # ViewId("Kasawari", "Sigma_Crossing_Event", "a76ade6ceeef85"),
        ]
        
        instances_deleted = 0
        
        for view_id in view_ids:
            try:
                # Query instances
                query_result = client.data_modeling.instances.list(
                    instance_type="node",
                    sources=view_id,
                    limit=-1
                )
                
                # Filter tags/instances
                instance_to_delete = []
                for instance in query_result:
                    if (instance.external_id.startswith(f"{tag}_forecast_") # or 
                        # instance.external_id.startswith(f"{tag}_rate_") or
                        # instance.external_id == f"{tag}_sigma_summary" or
                        # instance.external_id == f"{tag}_ttf_summary" or
                        # instance.external_id.startswith(f"{tag}_sigma_")
                        ):
                        instance_to_delete.append(instance.external_id)

                # Delete instances
                if instance_to_delete:
                    client.data_modeling.instances.delete(
                        nodes=[(INSTANCE_SPACE, ext_id) for ext_id in instance_to_delete]
                    )
                    instances_deleted += len(instance_to_delete)
                    print(f"Deleted {len(instance_to_delete)} instances from {view_id.external_id}")
                    
            except CogniteAPIError as e:
                print(f"Warning: Could not delete instances from view {view_id.external_id}: {str(e)}")
                continue
        
        print(f"Successfully deleted {instances_deleted} data model instances for forecast tag {tag}")
        return True
        
    except Exception as e:
        print(f"Error deleting data model instances for forecast tag {tag}: {str(e)}")
        return False

def write_model_output_to_dm(client, data_export_tuple):
    check_spaces(client)
    
    tbl_sigma, tbl_sigma_summary, tbl_ttf, tbl_forecast, tbl_rate = data_export_tuple
    
    tag = tbl_forecast["Tag"].iloc[0] if not tbl_forecast.empty else "unknown_tag"
    
    # Delete old nodes from time series and instances from data model for every write
    delete_nodes(client, tag)
    delete_instances(client, tag)
    
    nodes_to_create = []
    
    # 1. Forecast_Pressure
    for _, row in tbl_forecast.iterrows():
        # external_id with timestamp
        timestamp_str = row['Timestamp'].strftime('%Y%m%d_%H%M%S') if row['Timestamp'] is not None and pd.notna(row['Timestamp']) else 'unknown'
        external_id = f'{tag}_forecast_{timestamp_str}'
        
        property_map = {
            "Timestamp": row['Timestamp'],
            "Forecast": float(row['Forecast']) if row['Forecast'] is not None and pd.notna(row['Forecast']) else None,
            "Lower_CI": float(row['Lower_CI']) if row['Lower_CI'] is not None and pd.notna(row['Lower_CI']) else None,
            "Upper_CI": float(row['Upper_CI']) if row['Upper_CI'] is not None and pd.notna(row['Upper_CI']) else None,
            "Tag": tag,
        }
        
        sources = [
            NodeOrEdgeData(
                ViewId("Kasawari", "Forecast_Pressure", "7548c1357c9c45"),
                property_map,
            )
        ]
        
        node = NodeApply(
            space=INSTANCE_SPACE,
            sources=sources,
            external_id=external_id,
        )
        nodes_to_create.append(node)
    
    # 2. Rate of Change
    for _, row in tbl_rate.iterrows():
        external_id = f'{tag}_rate_{row["Period"]}'
        
        property_map = {
            "Tag": row['Tag'],
            "Period": row['Period'],
            "Slope_per_day": float(row['Slope_per_Day']) if row['Slope_per_Day'] is not None and pd.notna(row['Slope_per_Day']) else None,
            "Date_Modelling": pd.Timestamp.now(),
        }
        
        sources = [
            NodeOrEdgeData(
                ViewId("Kasawari", "Rate_of_Change", "831adde070985b"),
                property_map,
            )
        ]
        
        node = NodeApply(
            space=INSTANCE_SPACE,
            sources=sources,
            external_id=external_id,
        )
        nodes_to_create.append(node)
    
    # 3. TTF Breach
    # for _, row in tbl_ttf_breach.iterrows():
    #     external_id = f'{tag}_ttf_{row["Breach_Type"]}'.replace('+', 'pos').replace('-', 'neg')
    #     
    #     property_map = {
    #         "Tag": row['Tag'],
    #         "Breach_Type": row['Breach_Type'],
    #         "Predicted_Timestamp": row['Predicted_Timestamp'],
    #     }
    #     
    #     sources = [
    #         NodeOrEdgeData(
    #             ViewId("Kasawari", "TTF_Breach_Times", "latest"),
    #             property_map,
    #         )
    #     ]
    #     
    #     node = NodeApply(
    #         space=INSTANCE_SPACE,
    #         sources=sources,
    #         external_id=external_id,
    #     )
    #     nodes_to_create.append(node)
    
    # 4. Sigma Summary
    if not tbl_sigma_summary.empty:
        external_id = f'{tag}_sigma_summary'
        row = tbl_sigma_summary.iloc[0]
        
        property_map = {
            "Tag": tag,
            "Median": float(row['Median']) if row['Median'] is not None and pd.notna(row['Median']) else None,
            "MAD": float(row['MAD']) if row['MAD'] is not None and pd.notna(row['MAD']) else None,
            "Sigma": float(row['Sigma']) if row['Sigma'] is not None and pd.notna(row['Sigma']) else None,
            "Pov_1Sigma": float(row['+1σ']) if '+1σ' in tbl_sigma_summary.columns and row['+1σ'] is not None and pd.notna(row['+1σ']) else None,
            "Pov_2Sigma": float(row['+2σ']) if '+2σ' in tbl_sigma_summary.columns and row['+2σ'] is not None and pd.notna(row['+2σ']) else None,
            "Pov_3Sigma": float(row['+3σ']) if '+3σ' in tbl_sigma_summary.columns and row['+3σ'] is not None and pd.notna(row['+3σ']) else None,
            "Pov_4Sigma": float(row['+4σ']) if '+4σ' in tbl_sigma_summary.columns and row['+4σ'] is not None and pd.notna(row['+4σ']) else None,
            "Pov_5Sigma": float(row['+5σ']) if '+5σ' in tbl_sigma_summary.columns and row['+5σ'] is not None and pd.notna(row['+5σ']) else None,
            "Neg_1Sigma": float(row['-1σ']) if '-1σ' in tbl_sigma_summary.columns and row['-1σ'] is not None and pd.notna(row['-1σ']) else None,
            "Neg_2Sigma": float(row['-2σ']) if '-2σ' in tbl_sigma_summary.columns and row['-2σ'] is not None and pd.notna(row['-2σ']) else None,
            "Neg_3Sigma": float(row['-3σ']) if '-3σ' in tbl_sigma_summary.columns and row['-3σ'] is not None and pd.notna(row['-3σ']) else None,
            "Neg_4Sigma": float(row['-4σ']) if '-4σ' in tbl_sigma_summary.columns and row['-4σ'] is not None and pd.notna(row['-4σ']) else None,
            "Neg_5Sigma": float(row['-5σ']) if '-5σ' in tbl_sigma_summary.columns and row['-5σ'] is not None and pd.notna(row['-5σ']) else None,
            "Date_Modelling": pd.Timestamp.now(),
        }
        
        sources = [
            NodeOrEdgeData(
                ViewId("Kasawari", "Sigma_Summary", "cb8761a073327c"),
                property_map,
            )
        ]
        
        node = NodeApply(
            space=INSTANCE_SPACE,
            sources=sources,
            external_id=external_id,
        )
        nodes_to_create.append(node)
    
    # 5. TTF Summary
    if not tbl_ttf.empty:
        external_id = f'{tag}_ttf_summary'
        row = tbl_ttf.iloc[0] 
        
        property_map = {
            "Tags": tag,
            "Date_Modelling": pd.Timestamp.now(),
            "Model_MAE": float(row['Model_MAE']) if 'Model_MAE' in row and row['Model_MAE'] is not None and pd.notna(row['Model_MAE']) else None,
            "Model_RMSE": float(row['Model_RMSE']) if 'Model_RMSE' in row and row['Model_RMSE'] is not None and pd.notna(row['Model_RMSE']) else None,
            "Model_MAPE": float(row['Model_MAPE']) if 'Model_MAPE' in row and row['Model_MAPE'] is not None and pd.notna(row['Model_MAPE']) else None,
            "Score": float(row['Score']) if 'Score' in row and row['Score'] is not None and pd.notna(row['Score']) else None,
            # "TTF_High_Alarm": row['Positive 1σ'] if 'Positive 1σ' in row and pd.notna(row['Positive 1σ']) else None,
            # "TTF_Low_Alarm": row['Negative 1σ'] if 'Negative 1σ' in row and pd.notna(row['Negative 1σ']) else None,
            "TTF_Pos_1Sigma": row['Positive 1σ'] if 'Positive 1σ' in row and pd.notna(row['Positive 1σ']) else None,
            "TTF_Pos_2Sigma": row['Positive 2σ'] if 'Positive 2σ' in row and pd.notna(row['Positive 2σ']) else None,
            "TTF_Pos_3Sigma": row['Positive 3σ'] if 'Positive 3σ' in row and pd.notna(row['Positive 3σ']) else None,
            "TTF_Pos_4Sigma": row['Positive 4σ'] if 'Positive 4σ' in row and pd.notna(row['Positive 4σ']) else None,
            "TTF_Pos_5Sigma": row['Positive 5σ'] if 'Positive 5σ' in row and pd.notna(row['Positive 5σ']) else None,
            "TTF_Neg_1Sigma": row['Negative 1σ'] if 'Negative 1σ' in row and pd.notna(row['Negative 1σ']) else None,
            "TTF_Neg_2Sigma": row['Negative 2σ'] if 'Negative 2σ' in row and pd.notna(row['Negative 2σ']) else None,
            "TTF_Neg_3Sigma": row['Negative 3σ'] if 'Negative 3σ' in row and pd.notna(row['Negative 3σ']) else None,
            "TTF_Neg_4Sigma": row['Negative 4σ'] if 'Negative 4σ' in row and pd.notna(row['Negative 4σ']) else None,
            "TTF_Neg_5Sigma": row['Negative 5σ'] if 'Negative 5σ' in row and pd.notna(row['Negative 5σ']) else None,
            "TTF_ROC_14d": row['TTF ROC 14d'] if 'TTF ROC 14d' in row and pd.notna(row['TTF ROC 14d']) else None,
            "TTF_ROC_30d": row['TTF ROC 30d'] if 'TTF ROC 30d' in row and pd.notna(row['TTF ROC 30d']) else None,
            "TTF_ROC_60d": row['TTF ROC 60d'] if 'TTF ROC 60d' in row and pd.notna(row['TTF ROC 60d']) else None,
        }
        
        sources = [
            NodeOrEdgeData(
                ViewId("Kasawari", "TTF_Summary_N", "f3358c8b6a0f29"),
                property_map,
            )
        ]
        
        node = NodeApply(
            space=INSTANCE_SPACE,
            sources=sources,
            external_id=external_id,
        )
        nodes_to_create.append(node)
    
    # 6. Sigma Crossing Event
    for _, row in tbl_sigma.iterrows():
        timestamp_str = row['Timestamp'].strftime('%Y%m%d_%H%M%S') if row['Timestamp'] is not None and pd.notna(row['Timestamp']) else 'unknown'
        type_str = str(row['Type']).replace('+', 'pos').replace('-', 'neg').replace(' ', '_')
        external_id = f'{tag}_sigma_{type_str}_{timestamp_str}'
        
        property_map = {
            "Timestamp": row['Timestamp'],
            "Value": float(row['Value']) if row['Value'] is not None and pd.notna(row['Value']) else None,
            "Type": row['Type'],
            "Tag": tag,
        }
        
        sources = [
            NodeOrEdgeData(
                ViewId("Kasawari", "Sigma_Crossing_Event", "a76ade6ceeef85"),
                property_map,
            )
        ]
        
        node = NodeApply(
            space=INSTANCE_SPACE,
            sources=sources,
            external_id=external_id,
        )
        nodes_to_create.append(node)
    
    if nodes_to_create:
        try:
            result = client.data_modeling.instances.apply(nodes=nodes_to_create)
            print(f"Successfully wrote forecast nodes to Cognite Data Model for tag {tag}")
            return result
        except Exception as e:
            print(f"Failed to write nodes to CDF for tag {tag}: {str(e)}")
            raise
    else:
        print(f"No nodes to create for tag {tag}")
        return None