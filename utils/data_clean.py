import pandas as pd
# from utils.plot import boxplot_outliers

def detect_outliers_iqr(df: pd.DataFrame, column: str = 'Value'):
    # Using 10th and 90th percentiles
    Q1 = df[column].quantile(0.10)
    Q3 = df[column].quantile(0.90)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound

def remove_outliers(df: pd.DataFrame, column: str = 'Value'):
    lower_bound, upper_bound = detect_outliers_iqr(df, column)

    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    removed_count = len(df) - len(df_clean)
    return df_clean, removed_count, len(df)

def data_clean(df: pd.DataFrame, column: str = 'Value'):
    # plt_boxplot = boxplot_outliers(df, column)
    df_clean, removed_count, total_count = remove_outliers(df, column)
    return df_clean, removed_count, total_count