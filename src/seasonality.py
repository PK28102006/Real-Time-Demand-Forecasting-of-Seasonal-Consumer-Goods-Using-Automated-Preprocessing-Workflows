"""
src/seasonality.py
Simple Seasonality Analysis Module.
Provides the 3 required charts:
1. Demand over time
2. Average demand by month
3. Average demand by product
And dynamically calculates:
- Peak demand month
- Lowest demand month
- Highest-demand product
- Lowest-demand product
"""

from typing import Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

CHART_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#334155"),
    margin=dict(l=35, r=30, t=50, b=40),
)


def calculate_seasonality_insights(
    df: pd.DataFrame,
    date_col: str = "Date",
    demand_col: str = "Demand",
    product_col: str = "Product",
) -> Dict[str, Any]:
    """
    Dynamically identify peak/lowest month and highest/lowest demand product from actual data.
    """
    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col])
    if "Month" not in work_df.columns:
        work_df["Month"] = work_df[date_col].dt.month

    # 1. Monthly Averages
    monthly_avg = work_df.groupby("Month")[demand_col].mean().reset_index()
    monthly_avg["MonthName"] = monthly_avg["Month"].map(MONTH_NAMES)

    peak_month_row = monthly_avg.loc[monthly_avg[demand_col].idxmax()]
    lowest_month_row = monthly_avg.loc[monthly_avg[demand_col].idxmin()]

    peak_month = peak_month_row["MonthName"]
    peak_month_val = round(float(peak_month_row[demand_col]), 1)
    lowest_month = lowest_month_row["MonthName"]
    lowest_month_val = round(float(lowest_month_row[demand_col]), 1)

    # 2. Product Averages
    if product_col in work_df.columns:
        prod_agg = work_df.groupby(product_col)[demand_col].sum().reset_index()
        highest_prod_row = prod_agg.loc[prod_agg[demand_col].idxmax()]
        lowest_prod_row = prod_agg.loc[prod_agg[demand_col].idxmin()]
        highest_product = str(highest_prod_row[product_col])
        highest_product_demand = int(round(float(highest_prod_row[demand_col])))
        lowest_product = str(lowest_prod_row[product_col])
        lowest_product_demand = int(round(float(lowest_prod_row[demand_col])))
    else:
        highest_product = "N/A"
        highest_product_demand = 0
        lowest_product = "N/A"
        lowest_product_demand = 0

    return {
        "peak_month": peak_month,
        "peak_month_val": peak_month_val,
        "lowest_month": lowest_month,
        "lowest_month_val": lowest_month_val,
        "highest_product": highest_product,
        "highest_product_demand": highest_product_demand,
        "lowest_product": lowest_product,
        "lowest_product_demand": lowest_product_demand,
    }


def create_demand_over_time_chart(
    df: pd.DataFrame,
    date_col: str = "Date",
    demand_col: str = "Demand",
) -> go.Figure:
    """
    Chart 1: Daily Demand Over Time with clean styling and smooth curve.
    """
    plot_df = df.copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    daily = plot_df.groupby(date_col)[demand_col].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily[demand_col],
        mode="lines",
        name="Daily Demand",
        line=dict(color="#2563EB", width=2),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.06)",
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Demand:</b> %{y:,.0f} units<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>1. Daily Demand Over Time</b> (Aggregate Volume)", font=dict(size=14, color="#1E293B")),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Timeline", tickfont=dict(color="#64748B")),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Total Demand Units", tickfont=dict(color="#64748B")),
    )
    return fig


def create_monthly_demand_chart(
    df: pd.DataFrame,
    date_col: str = "Date",
    demand_col: str = "Demand",
) -> go.Figure:
    """
    Chart 2: Average Demand by Month with polished gradient styling.
    """
    plot_df = df.copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df["Month"] = plot_df[date_col].dt.month

    monthly = plot_df.groupby("Month")[demand_col].mean().reset_index()
    monthly["MonthName"] = monthly["Month"].map(lambda m: MONTH_NAMES[m][:3])
    monthly = monthly.sort_values(by="Month")

    fig = px.bar(
        monthly,
        x="MonthName",
        y=demand_col,
        title="<b>2. Average Demand by Month</b> (Seasonal Pattern)",
        labels={"MonthName": "Calendar Month", demand_col: "Mean Demand"},
        color=demand_col,
        color_continuous_scale=["#BFDBFE", "#3B82F6", "#1D4ED8"],
        text_auto=".1f",
    )
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        textfont=dict(size=11, color="#1E3A8A"),
        hovertemplate="<b>%{x}:</b> %{y:.1f} avg units/day<extra></extra>",
    )
    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(font=dict(size=14, color="#1E293B")),
        coloraxis_showscale=False,
        xaxis=dict(showgrid=False, tickfont=dict(color="#64748B")),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
    )
    return fig


def create_product_demand_chart(
    df: pd.DataFrame,
    product_col: str = "Product",
    demand_col: str = "Demand",
) -> go.Figure:
    """
    Chart 3: Average Daily Demand by Product with horizontal bars.
    """
    if product_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No Product column present", showarrow=False)
        return fig

    prod_summary = df.groupby(product_col)[demand_col].mean().reset_index()
    prod_summary = prod_summary.sort_values(by=demand_col, ascending=True)

    fig = px.bar(
        prod_summary,
        x=demand_col,
        y=product_col,
        orientation="h",
        title="<b>3. Average Daily Demand by Product</b>",
        labels={demand_col: "Average Daily Demand", product_col: "Product"},
        color=demand_col,
        color_continuous_scale=["#A7F3D0", "#10B981", "#047857"],
        text_auto=".1f",
    )
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        textfont=dict(size=11, color="#064E3B"),
        hovertemplate="<b>%{y}:</b> %{x:.1f} units/day<extra></extra>",
    )
    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(font=dict(size=14, color="#1E293B")),
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        yaxis=dict(showgrid=False, tickfont=dict(color="#64748B")),
    )
    return fig
