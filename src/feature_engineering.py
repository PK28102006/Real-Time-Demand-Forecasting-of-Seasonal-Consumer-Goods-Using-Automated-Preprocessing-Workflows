"""
src/feature_engineering.py
Feature Engineering Module for Seasonal Demand Forecasting.
Creates:
- Date features: Year, Month, DayOfWeek, Quarter
- Lag features: Lag_1, Lag_7
- Rolling feature: Rolling_Mean_7 (shifted by 1 day to guarantee NO DATA LEAKAGE)
- Seasonal features: Month_Sin, Month_Cos
- Categorical encoding for Product
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def engineer_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    demand_col: str = "Demand",
    product_col: str = "Product",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate date, lag, rolling, and seasonal features with strictly zero lookahead bias.
    """
    featured = df.copy()
    featured[date_col] = pd.to_datetime(featured[date_col])

    # Sort chronologically by product and date
    sort_cols = [product_col, date_col] if product_col in featured.columns else [date_col]
    featured = featured.sort_values(by=sort_cols).reset_index(drop=True)

    # 1. Date Features
    featured["Year"] = featured[date_col].dt.year
    featured["Month"] = featured[date_col].dt.month
    featured["DayOfWeek"] = featured[date_col].dt.dayofweek
    featured["Quarter"] = featured[date_col].dt.quarter

    # 2. Seasonal Cyclical Features
    featured["Month_Sin"] = np.sin(2 * np.pi * featured["Month"] / 12.0)
    featured["Month_Cos"] = np.cos(2 * np.pi * featured["Month"] / 12.0)

    # 3. Lag Features (Lag_1, Lag_7)
    # Strictly past values
    if product_col in featured.columns:
        featured["Lag_1"] = featured.groupby(product_col)[demand_col].shift(1)
        featured["Lag_7"] = featured.groupby(product_col)[demand_col].shift(7)
    else:
        featured["Lag_1"] = featured[demand_col].shift(1)
        featured["Lag_7"] = featured[demand_col].shift(7)

    # 4. Rolling Feature (Rolling_Mean_7)
    # CRITICAL LEAKAGE PREVENTION: Shift by 1 first so current day demand is never included!
    if product_col in featured.columns:
        featured["Rolling_Mean_7"] = (
            featured.groupby(product_col)[demand_col]
            .transform(lambda s: s.shift(1).rolling(window=7, min_periods=7).mean())
        )
    else:
        featured["Rolling_Mean_7"] = featured[demand_col].shift(1).rolling(window=7, min_periods=7).mean()

    # Drop initial rows where lag/rolling features are NaN (first 7 days)
    featured = featured.dropna(subset=["Lag_1", "Lag_7", "Rolling_Mean_7"]).reset_index(drop=True)

    # 5. Optional Exogenous Columns (Price, Promotion, Holiday)
    base_features = [
        "Year", "Month", "DayOfWeek", "Quarter",
        "Month_Sin", "Month_Cos",
        "Lag_1", "Lag_7",
        "Rolling_Mean_7",
    ]

    for exog in ["Price", "Promotion", "Holiday", "IsWeekend"]:
        if exog in featured.columns and exog not in base_features:
            base_features.append(exog)

    # One-Hot Encoding for Product
    encoded_cols = []
    if product_col in featured.columns:
        dummies = pd.get_dummies(featured[product_col], prefix=product_col, drop_first=True, dtype=int)
        encoded_cols = list(dummies.columns)
        featured = pd.concat([featured, dummies], axis=1)

    feature_columns = base_features + encoded_cols
    feature_columns = [c for c in feature_columns if c in featured.columns and np.issubdtype(featured[c].dtype, np.number)]

    return featured, feature_columns
