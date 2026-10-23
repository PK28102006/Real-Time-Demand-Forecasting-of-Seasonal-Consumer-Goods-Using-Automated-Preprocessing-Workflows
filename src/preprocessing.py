"""
src/preprocessing.py
Automated Preprocessing Workflow for Seasonal Consumer Demand Forecasting.
Performs:
1. Dataset structure detection
2. Missing value detection and statistical imputation (median for numeric, mode for categorical)
3. Date conversion to datetime & chronological sorting
4. Duplicate row detection and removal
5. Demand outlier detection using IQR & winsorization capping
6. Extraction of date features (Year, Month, Day, DayOfWeek, Quarter, IsWeekend)
7. Production of before/after comparison statistics
8. Persistence to data/processed/seasonal_demand_processed.csv
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "seasonal_demand_processed.csv"


class AutomatedPreprocessor:
    """
    Automated data cleansing and preprocessing pipeline.
    """

    def __init__(self, date_col: str = "Date", demand_col: str = "Demand"):
        self.date_col = date_col
        self.demand_col = demand_col
        self.audit_summary: Dict[str, Any] = {}

    def inspect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Inspect dataset structure and initial anomalies.
        """
        # Determine actual date and demand columns
        actual_date = self.date_col if self.date_col in df.columns else None
        actual_demand = self.demand_col if self.demand_col in df.columns else None

        if not actual_date:
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    actual_date = col
                    break

        if not actual_demand:
            for col in df.columns:
                if any(k in col.lower() for k in ["demand", "sales", "qty", "quantity"]):
                    actual_demand = col
                    break

        outliers_count = 0
        if actual_demand and actual_demand in df.columns:
            s = df[actual_demand].dropna()
            if len(s) > 0:
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                lower_b = max(0.0, q1 - 1.5 * iqr)
                upper_b = q3 + 1.5 * iqr
                outliers_count = int(((s < lower_b) | (s > upper_b)).sum())

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "missing_per_column": df.isna().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "outliers": outliers_count,
            "date_col": actual_date,
            "demand_col": actual_demand,
        }

    def process(
        self,
        df: pd.DataFrame,
        output_path: Path | str = PROCESSED_DATA_PATH,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete automated preprocessing workflow.
        """
        # Resolve column names
        initial_stats = self.inspect(df)
        self.date_col = initial_stats["date_col"] or "Date"
        self.demand_col = initial_stats["demand_col"] or "Demand"

        if self.date_col not in df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found.")
        if self.demand_col not in df.columns:
            raise ValueError(f"Demand column '{self.demand_col}' not found.")

        rows_before = len(df)
        missing_before = int(df.isna().sum().sum())
        duplicates_before = int(df.duplicated().sum())

        # Baseline outlier detection on raw data
        s_raw = df[self.demand_col].dropna()
        q1_raw = float(s_raw.quantile(0.25))
        q3_raw = float(s_raw.quantile(0.75))
        iqr_raw = q3_raw - q1_raw
        lb_raw = max(0.0, q1_raw - 1.5 * iqr_raw)
        ub_raw = q3_raw + 1.5 * iqr_raw
        outliers_before = int(((s_raw < lb_raw) | (s_raw > ub_raw)).sum())

        # 1. Duplicate removal
        clean_df = df.drop_duplicates().copy()

        # 2. Date conversion & chronological sorting
        clean_df[self.date_col] = pd.to_datetime(clean_df[self.date_col], errors="coerce")
        clean_df = clean_df.dropna(subset=[self.date_col]).copy()
        clean_df = clean_df.sort_values(by=self.date_col).reset_index(drop=True)

        # 3. Missing Value Imputation
        # For Demand: use time-based forward/backward fill then groupwise median
        if "Product" in clean_df.columns:
            clean_df[self.demand_col] = clean_df.groupby("Product")[self.demand_col].transform(
                lambda s: s.interpolate(method="linear").bfill().ffill()
            )
        else:
            clean_df[self.demand_col] = clean_df[self.demand_col].interpolate(method="linear").bfill().ffill()

        if clean_df[self.demand_col].isna().any():
            clean_df[self.demand_col] = clean_df[self.demand_col].fillna(clean_df[self.demand_col].median())

        # For Numerical columns: Median Imputation
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if clean_df[col].isna().any():
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())

        # For Categorical columns: Mode Imputation
        cat_cols = clean_df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if clean_df[col].isna().any():
                mode_val = clean_df[col].mode().iloc[0] if len(clean_df[col].mode()) > 0 else "Unknown"
                clean_df[col] = clean_df[col].fillna(mode_val)

        # 4. Outlier detection using IQR & winsorization capping
        q1 = float(clean_df[self.demand_col].quantile(0.25))
        q3 = float(clean_df[self.demand_col].quantile(0.75))
        iqr = q3 - q1
        lower_bound = max(0.0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr

        # Cap / Winsorize outliers
        clean_df[self.demand_col] = clean_df[self.demand_col].clip(lower=lower_bound, upper=upper_bound)

        # 5. Extract Date Features
        clean_df["Year"] = clean_df[self.date_col].dt.year
        clean_df["Month"] = clean_df[self.date_col].dt.month
        clean_df["Day"] = clean_df[self.date_col].dt.day
        clean_df["DayOfWeek"] = clean_df[self.date_col].dt.dayofweek
        clean_df["Quarter"] = clean_df[self.date_col].dt.quarter
        clean_df["IsWeekend"] = clean_df["DayOfWeek"].isin([5, 6]).astype(int)

        # Ensure no duplicates remain after imputation and capping
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)

        # Post-processing audit
        rows_after = len(clean_df)
        missing_after = int(clean_df.isna().sum().sum())
        duplicates_after = int(clean_df.duplicated().sum())
        outliers_after = int(((clean_df[self.demand_col] < lower_bound) | (clean_df[self.demand_col] > upper_bound)).sum())

        comparison_df = pd.DataFrame([
            {"Metric": "Rows", "Before": rows_before, "After": rows_after},
            {"Metric": "Missing values", "Before": missing_before, "After": missing_after},
            {"Metric": "Duplicate rows", "Before": duplicates_before, "After": duplicates_after},
            {"Metric": "Outliers", "Before": outliers_before, "After": outliers_after},
        ])

        self.audit_summary = {
            "comparison_table": comparison_df,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "duplicates_before": duplicates_before,
            "duplicates_after": duplicates_after,
            "outliers_before": outliers_before,
            "outliers_after": outliers_after,
            "iqr_bounds": {
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower Bound": round(lower_bound, 2),
                "Upper Bound": round(upper_bound, 2),
            },
        }

        # Save processed data
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(out_path, index=False)

        return clean_df, self.audit_summary
