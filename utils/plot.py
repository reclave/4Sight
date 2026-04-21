# TODO: Remove file before deploying to Cognite Functions
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from calc.rate_change import calculate_rate_change

# def plot_with_thresholds(df: pd.DataFrame, column: str = 'Value', timestamp_col: str = 'Timestamp'):
#     """
#     Plots a time series with alarm thresholds. Highlights values above high limit and below low limit.
#     Parameters:
#     - df: Cleaned DataFrame with Timestamp and Value
#     - column: Value column name
#     - timestamp_col: Timestamp column name
#     """
#     fig = go.Figure()
    
#     # Main time series line
#     fig.add_trace(go.Scatter(
#         x=df[timestamp_col], 
#         y=df[column],
#         mode='lines',
#         name='Value',
#         line=dict(color='blue', width=1)
#     ))
    
#     # Add alarm threshold lines
#     fig.add_hline(
#         y=high_limit, 
#         line_dash="dash", 
#         line_color="red",
#         annotation_text=f"High Limit ({high_limit})",
#         annotation_position="bottom right"
#     )
#     fig.add_hline(
#         y=low_limit, 
#         line_dash="dash", 
#         line_color="orange",
#         annotation_text=f"Low Limit ({low_limit})",
#         annotation_position="top right"
#     )
    
#     # Highlight points above/below thresholds
#     above_high = df[df[column] >= high_limit]
#     below_low = df[df[column] <= low_limit]
    
#     if not above_high.empty:
#         fig.add_trace(go.Scatter(
#             x=above_high[timestamp_col],
#             y=above_high[column],
#             mode='markers',
#             name='Above High Limit',
#             marker=dict(color='red', size=4)
#         ))
    
#     if not below_low.empty:
#         fig.add_trace(go.Scatter(
#             x=below_low[timestamp_col],
#             y=below_low[column],
#             mode='markers',
#             name='Below Low Limit',
#             marker=dict(color='orange', size=4)
#         ))
    
#     fig.update_layout(
#         title='Time Series with Alarm Thresholds',
#         showlegend=True,
#         width=900,
#         height=450,  # Increased height to accommodate legend
#         template="plotly_white",
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=-0.2,
#             xanchor="center",
#             x=0.5
#         )
#     )
    
#     return fig
    
def plot_sigma(df: pd.DataFrame, sigma_stats: dict, column: str = 'Value', timestamp_col: str = 'Timestamp'):
    # Plot time-series with alarm thresholds and sigma levels
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[timestamp_col], 
        y=df[column],
        mode='lines',
        name='Value',
        line=dict(color='blue', width=1)
    ))
    
    # Median line
    fig.add_hline(
        y=sigma_stats['median'], 
        line_color="lightgrey",
        line_width=1,
        annotation_text="Median",
        annotation_position="bottom left"
    )
    
    # Sigma lines
    colors_pos = ['orange', 'darkorange', 'sandybrown']
    colors_neg = ['purple', 'mediumpurple', 'plum']
    
    for i, (color_pos, color_neg) in enumerate(zip(colors_pos, colors_neg), 1):
        fig.add_hline(
            y=sigma_stats[f'{i}σ+'], 
            line_dash="dash", 
            line_color=color_pos,
            line_width=1,
            annotation_text=f"+{i}σ",
            annotation_position="top left"
        )
        fig.add_hline(
            y=sigma_stats[f'{i}σ-'], 
            line_dash="dash", 
            line_color=color_neg,
            line_width=1,
            annotation_text=f"-{i}σ",
            annotation_position="bottom left"
        )
    
    fig.update_layout(
        title='Time Series with Alarm and Sigma Thresholds',
        showlegend=True,
        width=900,
        height=400,
        template="plotly_white"
    )
    
    return fig

def plot_forecast(df_recent: pd.Series,
                                     forecast: pd.Series,
                                     lower: pd.Series,
                                     upper: pd.Series,
                                     sigma_stats: dict,
                                     ttf_dict: dict,
                                     title="Forecast + Sigma Breaches",
                                     show_ci: bool = True):
    # Plot for actual + forecasted values with sigma bands and alarm/sigma breaches
    fig = go.Figure()

    if show_ci:
        # Confidence interval (optional)
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=lower,
            mode='lines',
            line=dict(width=0),
            name='90% Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(211, 211, 211, 0.3)',  # Light gray
            hoverinfo='skip'
        ))
    
    # Sigma thresholds (the shaded areas)
    x_range = list(df_recent.index) + list(forecast.index)
    
    # 3-sigma band (set as red)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[sigma_stats['3σ+']] * len(x_range),
        mode='lines', line=dict(width=0),
        hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[sigma_stats['3σ-']] * len(x_range),
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)',
        name='±3σ', hoverinfo='skip', showlegend=True
    ))

    # 2-sigma band (yellow)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[sigma_stats['2σ+']] * len(x_range),
        mode='lines', line=dict(width=0),
        hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[sigma_stats['2σ-']] * len(x_range),
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)',
        name='±2σ', hoverinfo='skip', showlegend=True
    ))

    # 1-sigma band (green)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[sigma_stats['1σ+']] * len(x_range),
        mode='lines', line=dict(width=0),
        hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[sigma_stats['1σ-']] * len(x_range),
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)',
        name='±1σ', hoverinfo='skip', showlegend=True
    ))

    # Historical values
    fig.add_trace(go.Scatter(
        x=df_recent.index,
        y=df_recent.values,
        mode='lines',
        name='Recent Values',
        line=dict(color='blue', width=1)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode='lines',
        name='Forecast',
        line=dict(color='magenta', width=1)
    ))
    
    # Median line
    median = sigma_stats["median"]
    fig.add_hline(
        y=median,
        line_color="lightgrey",
        line_width=1,
        annotation_text="Median",
        annotation_position="bottom left"
    )
    
    # Breach timestamp vertical lines
    for label, breach_time in ttf_dict.items():
        if breach_time is not None:
            # Convert timestamp to datetime
            if hasattr(breach_time, 'to_pydatetime'):
                breach_time = breach_time.to_pydatetime()
            
            # Using add_shape instead of add_vline for better datetime support
            fig.add_shape(
                type="line",
                x0=breach_time,
                x1=breach_time,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color="magenta",
                    width=2,
                    dash="dot"
                ),
                opacity=0.6
            )
            
            # Text annotation
            fig.add_annotation(
                x=breach_time,
                y=1,
                yref="paper",
                text=label,
                textangle=90,
                yanchor="bottom",
                showarrow=False,
                font=dict(color="magenta", size=10)
            )
    
    # Y-axis range
    all_values = pd.concat([df_recent, forecast])
    y_min = all_values.min()
    y_max = all_values.max()
    buffer = (y_max - y_min) * 0.1  # 10% buffer

    fig.update_layout(
        title=title,
        yaxis_range=[y_min - buffer, y_max + buffer],
        showlegend=True,
        width=1000,
        height=450,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_model_accuracy(
    evaluation_df: pd.DataFrame,
    model_name: str,
    unit: str = "",
    mae: float = None,
    rmse: float = None,
    mape: float = None,
):
    plot_df = evaluation_df.copy()
    plot_df = plot_df.dropna(subset=["ds", "actual", "predicted"])

    if plot_df.empty:
        empty_left = go.Figure()
        empty_left.update_layout(
            title=f"{model_name} Holdout Actual vs Predicted",
            template="plotly_white",
            height=420,
        )
        empty_right = go.Figure()
        empty_right.update_layout(
            title=f"{model_name} Predicted vs Actual",
            template="plotly_white",
            height=420,
        )
        return empty_left, empty_right

    value_label = f"Value ({unit})" if unit else "Value"

    left_fig = go.Figure()
    left_fig.add_trace(
        go.Scatter(
            x=plot_df["ds"],
            y=plot_df["actual"],
            mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
        )
    )
    left_fig.add_trace(
        go.Scatter(
            x=plot_df["ds"],
            y=plot_df["predicted"],
            mode="lines",
            name="Predicted",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
    )

    actual_min = min(plot_df["actual"].min(), plot_df["predicted"].min())
    actual_max = max(plot_df["actual"].max(), plot_df["predicted"].max())
    axis_buffer = (actual_max - actual_min) * 0.05
    if axis_buffer == 0:
        axis_buffer = max(abs(actual_max) * 0.05, 1.0)

    min_bound = actual_min - axis_buffer
    max_bound = actual_max + axis_buffer

    right_fig = go.Figure()
    right_fig.add_trace(
        go.Scatter(
            x=plot_df["actual"],
            y=plot_df["predicted"],
            mode="markers",
            name="Holdout Points",
            marker=dict(color="#2ca02c", size=7, opacity=0.7),
            showlegend=False,
        )
    )
    right_fig.add_trace(
        go.Scatter(
            x=[min_bound, max_bound],
            y=[min_bound, max_bound],
            mode="lines",
            name="Ideal Fit",
            line=dict(color="#7f7f7f", width=2, dash="dot"),
        )
    )

    left_fig.update_xaxes(title_text="Date")
    left_fig.update_yaxes(title_text=value_label)
    left_fig.update_layout(
        title=f"{model_name}: Actual vs Predicted",
        template="plotly_white",
        height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=80, l=40, r=20, b=40),
    )

    right_fig.update_xaxes(title_text=f"Actual {value_label}", range=[min_bound, max_bound])
    right_fig.update_yaxes(title_text=f"Predicted {value_label}", range=[min_bound, max_bound])
    right_fig.update_layout(
        title=f"{model_name}: Parity Plot Actual vs Predicted",
        template="plotly_white",
        height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=80, l=40, r=20, b=40),
    )

    return left_fig, right_fig

def plot_rate_change(df: pd.DataFrame, periods: list = [365, 180, 90, 60, 30],
                               column: str = 'Value', timestamp_col: str = 'Timestamp',
                               convert_to: str = 'per_day'):
    """
    Plots the linear regression lines over different periods.
    Slopes are converted to per_day/per_hour/per_second as needed.
    """
    fig = go.Figure()
    
    # Original data
    fig.add_trace(go.Scatter(
        x=df[timestamp_col],
        y=df[column],
        mode='lines',
        name='Original Data',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, days in enumerate(periods):
        try:
            slope, df_window = calculate_rate_change(df, days, column, timestamp_col, convert_to)

            X = df_window['time_seconds'].values.reshape(-1, 1)
            model = LinearRegression().fit(X, df_window[column].values.reshape(-1, 1))
            y_pred = model.predict(X)

            fig.add_trace(go.Scatter(
                x=df_window[timestamp_col],
                y=y_pred.flatten(),
                mode='lines',
                name=f"{days}d slope: {slope:.4f} ({convert_to})",
                line=dict(color=colors[i], width=2)
            ))

        except Exception as e:
            continue
    
    fig.update_layout(
        title="Rate of Change Comparison Over Time Windows",
        showlegend=True,
        width=900,
        height=450,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def boxplot_outliers(df: pd.DataFrame, column: str = 'Value'):
    # Boxplot to visually inspect outliers (legacy feature)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(df[column], vert=False)
    ax.set_title(f"Boxplot for {column} (with potential outliers)")
    ax.grid(True)
    return fig