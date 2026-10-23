"""
src/forecasting.py
Future Demand Forecasting Module.
Executes recursive multi-step autoregressive demand forecasting for 7, 14, or 30 days.
Updates lag (Lag_1, Lag_7) and rolling features (Rolling_Mean_7) iteratively.
Generates interactive Plotly line charts combining Historical Demand + Forecast Demand.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
FORECAST_CSV_PATH = BASE_DIR / "outputs" / "forecasts" / "forecast.csv"


def generate_future_forecast(
    model: Any,
    feature_cols: List[str],
    historical_df: pd.DataFrame,
    horizon_days: int = 30,
    date_col: str = "Date",
    demand_col: str = "Demand",
    product_col: str = "Product",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce recursive forward demand predictions for 7, 14, or 30 days into the future.
    """
    hist = historical_df.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    last_date = hist[date_col].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    has_product = product_col in hist.columns
    products = hist[product_col].unique().tolist() if has_product else ["Aggregate"]

    # Maintain recent buffer per product for lag/rolling feeding
    buffer_list = []
    for p in products:
        p_slice = hist[hist[product_col] == p].sort_values(by=date_col) if has_product else hist.sort_values(by=date_col)
        buffer_list.append(p_slice.tail(30))
    active_buffer = pd.concat(buffer_list, ignore_index=True)

    predictions = []

    # Step day-by-day
    for current_day in future_dates:
        for p in products:
            p_hist = active_buffer[active_buffer[product_col] == p].sort_values(by=date_col) if has_product else active_buffer.sort_values(by=date_col)
            past_demands = p_hist[demand_col].values
            n_past = len(past_demands)

            lag_1 = past_demands[-1] if n_past >= 1 else 30.0
            lag_7 = past_demands[-7] if n_past >= 7 else lag_1
            rm_7 = float(np.mean(past_demands[-7:])) if n_past >= 7 else lag_1

            month = current_day.month
            day_of_week = current_day.dayofweek
            quarter = current_day.quarter
            year = current_day.year
            is_weekend = 1 if day_of_week in [5, 6] else 0

            month_sin = np.sin(2 * np.pi * month / 12.0)
            month_cos = np.cos(2 * np.pi * month / 12.0)

            row_features: Dict[str, Any] = {
                "Year": year,
                "Month": month,
                "DayOfWeek": day_of_week,
                "Quarter": quarter,
                "Month_Sin": month_sin,
                "Month_Cos": month_cos,
                "Lag_1": lag_1,
                "Lag_7": lag_7,
                "Rolling_Mean_7": rm_7,
                "IsWeekend": is_weekend,
                "Promotion": 0.0,
                "Holiday": 0.0,
            }

            if "Price" in feature_cols:
                row_features["Price"] = float(p_hist["Price"].mean()) if "Price" in p_hist.columns else 45.0

            # One-hot encoded product features
            for feat in feature_cols:
                if feat.startswith(f"{product_col}_"):
                    target_prod = feat.split(f"{product_col}_", 1)[1]
                    row_features[feat] = 1 if target_prod == p else 0

            # Vector matching training schema
            feat_vector = pd.DataFrame([{col: row_features.get(col, 0.0) for col in feature_cols}])
            pred_demand = float(np.clip(model.predict(feat_vector)[0], 0, None))
            pred_demand = round(pred_demand, 1)

            predictions.append({
                date_col: current_day.strftime("%Y-%m-%d"),
                product_col: p,
                demand_col: pred_demand,
            } if has_product else {
                date_col: current_day.strftime("%Y-%m-%d"),
                demand_col: pred_demand,
            })

            # Append to buffer for next day's recursive lags
            new_entry = {date_col: current_day, demand_col: pred_demand}
            if has_product:
                new_entry[product_col] = p
                if "Price" in p_hist.columns:
                    new_entry["Price"] = row_features.get("Price", 45.0)

            active_buffer = pd.concat([active_buffer, pd.DataFrame([new_entry])], ignore_index=True)

    detailed_df = pd.DataFrame(predictions)

    # Aggregate daily demand
    aggregate_df = detailed_df.groupby(date_col)[demand_col].sum().reset_index()
    aggregate_df.columns = ["Date", "Predicted Demand"]
    aggregate_df["Predicted Demand"] = aggregate_df["Predicted Demand"].round(1)

    # Save to outputs/forecasts/forecast.csv
    out_file = Path(FORECAST_CSV_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    aggregate_df.to_csv(out_file, index=False)

    return detailed_df, aggregate_df


def create_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_col: str = "Date",
    demand_col: str = "Demand",
    lookback_days: int = 90,
) -> go.Figure:
    """
    Renders simple interactive line chart: Historical Demand + Forecast Demand.
    """
    hist = historical_df.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    hist_daily = hist.groupby(date_col)[demand_col].sum().reset_index()
    hist_daily = hist_daily.sort_values(by=date_col).tail(lookback_days)

    fc = forecast_df.copy()
    fc[date_col] = pd.to_datetime(fc[date_col])
    fc_val_col = demand_col if demand_col in fc.columns else "Predicted Demand"
    fc_daily = fc.groupby(date_col)[fc_val_col].sum().reset_index()
    fc_daily.columns = [date_col, demand_col]

    # Join the last historical point with the first forecast point for seamless continuity
    last_point = hist_daily.iloc[[-1]] if len(hist_daily) > 0 else None
    bridge_fc = pd.concat([last_point, fc_daily], ignore_index=True) if last_point is not None else fc_daily

    fig = go.Figure()

    # Historical trace
    fig.add_trace(go.Scatter(
        x=hist_daily[date_col],
        y=hist_daily[demand_col],
        mode="lines",
        name="Historical Demand",
        line=dict(color="#1E293B", width=2.2),
        hovertemplate="<b>History:</b> %{x|%Y-%m-%d}<br>Demand: %{y:,.0f} units<extra></extra>",
    ))

    # Forecast trace
    fig.add_trace(go.Scatter(
        x=bridge_fc[date_col],
        y=bridge_fc[demand_col],
        mode="lines+markers",
        name="Predicted Demand (Forecast)",
        line=dict(color="#2563EB", width=2.4, dash="dash"),
        marker=dict(size=5, color="#2563EB"),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.07)",
        hovertemplate="<b>Forecast:</b> %{x|%Y-%m-%d}<br>Demand: %{y:,.0f} units<extra></extra>",
    ))

    # Add vertical separator line
    if last_point is not None:
        divider = last_point[date_col].iloc[0]
        fig.add_vline(
            x=divider,
            line_width=1.5,
            line_dash="dot",
            line_color="#64748B",
            annotation_text="Horizon Start",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#64748B"),
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#334155"),
        title=dict(
            text="<b>Historical Demand + Future Forecast Trajectory</b>",
            font=dict(size=14, color="#1E293B"),
        ),
        hovermode="x unified",
        margin=dict(l=35, r=30, t=50, b=40),
        xaxis=dict(title="Timeline", showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        yaxis=dict(title="Total Demand Units", showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    return fig
