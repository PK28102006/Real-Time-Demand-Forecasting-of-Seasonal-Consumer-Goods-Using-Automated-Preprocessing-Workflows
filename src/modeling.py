"""
src/modeling.py
Machine Learning Training, Evaluation, and Best Model Selection Module.
Implements:
- Chronological 80/20 train/test split (no random splitting)
- Three supervised regression models:
  1. Linear Regression
  2. Random Forest Regressor
  3. Gradient Boosting Regressor
- Metrics: MAE, RMSE, R2
- Model comparison table
- Automatic selection of best model based on lowest RMSE
- Persistence of best model to models/best_model.pkl
- Actual vs Predicted Plotly graph
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_SAVE_PATH = BASE_DIR / "models" / "best_model.pkl"


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Chronological 80/20 split to strictly prevent temporal data leakage.
    """
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    split_point = int(len(df_sorted) * train_ratio)

    train_df = df_sorted.iloc[:split_point].copy()
    test_df = df_sorted.iloc[split_point:].copy()

    split_info = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": str(train_df[date_col].min())[:10],
        "train_end": str(train_df[date_col].max())[:10],
        "test_start": str(test_df[date_col].min())[:10],
        "test_end": str(test_df[date_col].max())[:10],
    }
    return train_df, test_df, split_info


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    demand_col: str = "Demand",
    random_state: int = 42,
    model_save_path: Path | str = MODEL_SAVE_PATH,
) -> Tuple[pd.DataFrame, str, Dict[str, float], Dict[str, np.ndarray], Any]:
    """
    Train Linear Regression, Random Forest, and Gradient Boosting.
    Evaluate on test set, construct comparison table, and select best model by lowest RMSE.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[demand_col]
    X_test = test_df[feature_cols]
    y_test = np.asarray(test_df[demand_col])

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = np.clip(lr.predict(X_test), 0, None)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = np.clip(rf.predict(X_test), 0, None)

    # 3. Gradient Boosting Regressor
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=random_state)
    gb.fit(X_train, y_train)
    gb_pred = np.clip(gb.predict(X_test), 0, None)

    models_dict = {
        "Linear Regression": (lr, lr_pred),
        "Random Forest": (rf, rf_pred),
        "Gradient Boosting": (gb, gb_pred),
    }

    records = []
    predictions_dict = {}

    for name, (model_obj, y_pred) in models_dict.items():
        predictions_dict[name] = y_pred
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        records.append({
            "Model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R²": round(r2, 4),
        })

    comparison_df = pd.DataFrame(records)

    # Automatically select model with lowest RMSE
    best_row = comparison_df.sort_values(by="RMSE", ascending=True).iloc[0]
    best_model_name = str(best_row["Model"])
    best_metrics = {
        "best_model": best_model_name,
        "mae": float(best_row["MAE"]),
        "rmse": float(best_row["RMSE"]),
        "r2": float(best_row["R²"]),
    }

    best_model_obj = models_dict[best_model_name][0]

    # Save best model
    save_file = Path(model_save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_name": best_model_name,
        "model_instance": best_model_obj,
        "feature_cols": feature_cols,
    }, save_file)

    return comparison_df, best_model_name, best_metrics, predictions_dict, best_model_obj


def create_actual_vs_predicted_chart(
    test_df: pd.DataFrame,
    date_col: str,
    demand_col: str,
    predictions: Dict[str, np.ndarray],
    model_name: str,
) -> go.Figure:
    """
    Generate interactive Plotly line chart comparing Actual vs Predicted Demand on chronological test set.
    """
    plot_df = test_df[[date_col, demand_col]].copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df["Predicted"] = predictions[model_name]

    daily = plot_df.groupby(date_col)[[demand_col, "Predicted"]].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily[demand_col],
        mode="lines",
        name="Actual Demand (Ground Truth)",
        line=dict(color="#1E293B", width=2.2),
        hovertemplate="<b>Actual:</b> %{y:,.0f} units<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily["Predicted"],
        mode="lines",
        name=f"Predicted ({model_name})",
        line=dict(color="#2563EB", width=2.2, dash="dash"),
        hovertemplate="<b>Predicted:</b> %{y:,.0f} units<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#334155"),
        title=dict(
            text=f"<b>Actual vs Predicted Demand</b> ({model_name}) — Chronological Test Period",
            font=dict(size=14, color="#1E293B"),
        ),
        hovermode="x unified",
        margin=dict(l=35, r=30, t=50, b=40),
        xaxis=dict(title="Chronological Test Period", showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        yaxis=dict(title="Total Demand Units", showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    return fig
