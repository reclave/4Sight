import pandas as pd

def calculate_sigma_median(df: pd.DataFrame, column: str = 'Value'):
    median = df[column].median()
    mad = (df[column] - median).abs().median()
    sigma = 1.4826 * mad

    thresholds = {
        "median": median,
        "1σ+": median + sigma,
        "2σ+": median + 2 * sigma,
        "3σ+": median + 3 * sigma,
        "4σ+": median + 4 * sigma,
        "5σ+": median + 5 * sigma,
        "1σ-": median - sigma,
        "2σ-": median - 2 * sigma,
        "3σ-": median - 3 * sigma,
        "4σ-": median - 4 * sigma,
        "5σ-": median - 5 * sigma,
        "mad": mad,
        "sigma": sigma
    }

    df_marked = df.copy()
    df_marked['Above 1σ'] = df_marked[column] >= thresholds['1σ+']
    df_marked['Above 2σ'] = df_marked[column] >= thresholds['2σ+']
    df_marked['Above 3σ'] = df_marked[column] >= thresholds['3σ+']
    df_marked['Above 4σ'] = df_marked[column] >= thresholds['4σ+']
    df_marked['Above 5σ'] = df_marked[column] >= thresholds['5σ+']
    df_marked['Below 1σ'] = df_marked[column] <= thresholds['1σ-']
    df_marked['Below 2σ'] = df_marked[column] <= thresholds['2σ-']
    df_marked['Below 3σ'] = df_marked[column] <= thresholds['3σ-']
    df_marked['Below 4σ'] = df_marked[column] <= thresholds['4σ-']
    df_marked['Below 5σ'] = df_marked[column] <= thresholds['5σ-']

    return df_marked, thresholds

def count_continuous_crossings(df: pd.DataFrame, col_flag: str):
    # Datapoints staying above/below a threshold
    return df[col_flag].sum()

def count_threshold_crossings(df_marked: pd.DataFrame):
    # Upward/downward sigma crossings
    count_dict = {
        "Crossed Above 1σ": count_continuous_crossings(df_marked, "Above 1σ"),
        "Crossed Above 2σ": count_continuous_crossings(df_marked, "Above 2σ"),
        "Crossed Above 3σ": count_continuous_crossings(df_marked, "Above 3σ"),
        "Crossed Above 4σ": count_continuous_crossings(df_marked, "Above 4σ"),
        "Crossed Above 5σ": count_continuous_crossings(df_marked, "Above 5σ"),
        "Crossed Below 1σ": count_continuous_crossings(df_marked, "Below 1σ"),
        "Crossed Below 2σ": count_continuous_crossings(df_marked, "Below 2σ"),
        "Crossed Below 3σ": count_continuous_crossings(df_marked, "Below 3σ"),
        "Crossed Below 4σ": count_continuous_crossings(df_marked, "Below 4σ"),
        "Crossed Below 5σ": count_continuous_crossings(df_marked, "Below 5σ"),
    }
    return count_dict

def extract_sigma_crossing_events(df_marked: pd.DataFrame):
    # Rows where values cross each sigma threshold (TTF)
    event_tables = {
        "Crossed Above 1σ": df_marked[df_marked['Above 1σ']][['Timestamp', 'Value']],
        "Crossed Above 2σ": df_marked[df_marked['Above 2σ']][['Timestamp', 'Value']],
        "Crossed Above 3σ": df_marked[df_marked['Above 3σ']][['Timestamp', 'Value']],
        "Crossed Above 4σ": df_marked[df_marked['Above 4σ']][['Timestamp', 'Value']],
        "Crossed Above 5σ": df_marked[df_marked['Above 5σ']][['Timestamp', 'Value']],
        "Crossed Below 1σ": df_marked[df_marked['Below 1σ']][['Timestamp', 'Value']],
        "Crossed Below 2σ": df_marked[df_marked['Below 2σ']][['Timestamp', 'Value']],
        "Crossed Below 3σ": df_marked[df_marked['Below 3σ']][['Timestamp', 'Value']],
        "Crossed Below 4σ": df_marked[df_marked['Below 4σ']][['Timestamp', 'Value']],
        "Crossed Below 5σ": df_marked[df_marked['Below 5σ']][['Timestamp', 'Value']],
    }
    return event_tables

def combine_sigma_events(event_tables: dict) -> pd.DataFrame:
    combined = pd.concat(
        [df.assign(Type=label) for label, df in event_tables.items()],
        ignore_index=True
    ).sort_values("Timestamp")
    return combined