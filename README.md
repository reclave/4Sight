# 4Sight

### About
4Sight produces forecasts, plots, and CLI which consists of:
- Predictive model for critical asset performance forecasting.
- Chooses best model with parameter turning for each unique tag.
- Time to Failure (TTF) prediction.
- Rate of Change (slope).
- Standard deviation thresholds and statistics.
- Outlier detection using IQR and removal.

### How to Run Engine (CDF Touchpoint)
- Local: Run 'handler.py'.
- Cognite Functions: Refer to "Deploying to Cognite Functions" section below to automate 4Sight.

### How to Run Application (User Interface)
- Open terminal and run 'streamlit run streamlit.py'.

### Authentication
- Default CDF auth parameters is defined in directory specified in 'client_gen.py'.
- Sample file is provided below (refer to Cognite documentations for more info).
- Run 'pip install pyyaml' to install yaml.
#### Sample auth .yaml file
```
token_url: https://example.com/oauth2/v2.0/token
client_id: abcd1234
client_secret: abcd1234
token_scopes:
  - https://[cluster].cognitedata.com/.default
cognite_project: testenv
cdf_cluster: [cluster]
client_name: your_name
base_url: f"https://{CDF_CLUSTER}.cognitedata.com"
```

### Requirements
- Please refer to the 'requirements.txt' file. 
- Install requirements by opening terminal and running 'pip install -r requirements.txt'.

### Cognite Functions Deployment
- Auth directory is only needed for local testing. Remove before uploading to Cognite Functions (discard client_gen calls from 'handler.py' and 'data_ingestion.py').
- Remove 'streamlit.py' before deploying to Cognite Functions.
- Remove lines marked by "TODO" before deploying to Cognite Functions.

### Cognite Functions Requirements
- Cognite Functions only accepts Python 3.10 and 3.11 interpreters.
- Needs 'handler.py' as entry point with function 'handle' that accepts one or more arguments: data, client, secrets, and function_call_info.
- Max of 100 concurrent calls and 10-minute timeout, with 1.5GB bandwidth for Azure.
- Extra Azure parameters defined in requirements.txt.