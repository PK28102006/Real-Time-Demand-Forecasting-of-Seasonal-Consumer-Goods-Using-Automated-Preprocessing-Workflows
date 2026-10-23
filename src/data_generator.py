"""
src/data_generator.py
Realistic Synthetic Dataset Generator for Seasonal Consumer Demand Forecasting.
Generates 3 years of daily data across 6 seasonal products with:
- Seasonal patterns
- Product-specific demand
- Price effect
- Promotion effect
- Holiday effect
- General trend
- Random variation
- Realistic missing values, duplicate rows, and demand outliers
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "seasonal_demand_raw.csv"


def generate_synthetic_dataset(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    random_state: int = 42,
    output_path: Path | str = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Generate realistic 3-year seasonal consumer demand dataset.

    Columns: Date, Product, Price, Promotion, Holiday, Demand
    Products: Air Conditioner, Fan, Umbrella, Winter Jacket, Sunscreen, School Bag
    """
    np.random.seed(random_state)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    # 6 seasonal products with tailored behavioral drivers
    product_profiles = {
        "Air Conditioner": {
            "base_price": 550.0,
            "base_demand": 32.0,
            "peak_months": [5, 6, 7],      # Summer peak
            "trough_months": [11, 12, 1],  # Winter trough
            "promo_boost": 0.35,
            "holiday_boost": 0.20,
            "trend": 0.05,
        },
        "Fan": {
            "base_price": 45.0,
            "base_demand": 55.0,
            "peak_months": [4, 5, 6, 7],   # Warm / Summer
            "trough_months": [11, 12, 1],
            "promo_boost": 0.30,
            "holiday_boost": 0.15,
            "trend": 0.03,
        },
        "Umbrella": {
            "base_price": 18.0,
            "base_demand": 28.0,
            "peak_months": [6, 7, 8, 9],   # Monsoon / Rainy season
            "trough_months": [12, 1, 2],
            "promo_boost": 0.25,
            "holiday_boost": 0.10,
            "trend": 0.02,
        },
        "Winter Jacket": {
            "base_price": 120.0,
            "base_demand": 30.0,
            "peak_months": [11, 12, 1, 2], # Winter cold months
            "trough_months": [5, 6, 7],    # Summer trough
            "promo_boost": 0.45,
            "holiday_boost": 0.30,
            "trend": 0.04,
        },
        "Sunscreen": {
            "base_price": 22.0,
            "base_demand": 42.0,
            "peak_months": [3, 4, 5, 6],   # Sunny / Summer
            "trough_months": [11, 12, 1],
            "promo_boost": 0.30,
            "holiday_boost": 0.20,
            "trend": 0.04,
        },
        "School Bag": {
            "base_price": 38.0,
            "base_demand": 35.0,
            "peak_months": [5, 6, 7],      # Academic reopening
            "trough_months": [10, 11, 12],
            "promo_boost": 0.35,
            "holiday_boost": 0.15,
            "trend": 0.03,
        },
    }

    # Holiday calendar indicator approximation
    holidays = np.zeros(n_days, dtype=int)
    for i, dt in enumerate(dates):
        m, d = dt.month, dt.day
        if (m == 1 and d in [1, 26]) or (m == 8 and d in [15]) or (m == 10 and d in [2, 24, 31]) or (m == 11 and d in [12, 24]) or (m == 12 and d in [25, 31]):
            holidays[i] = 1

    records = []
    year_progression = (dates - dates[0]).days / 365.25

    for prod_name, profile in product_profiles.items():
        base_price = profile["base_price"]
        base_demand = profile["base_demand"]
        peak_months = profile["peak_months"]
        trough_months = profile["trough_months"]
        promo_boost = profile["promo_boost"]
        holiday_boost = profile["holiday_boost"]
        trend_rate = profile["trend"]

        # Random promotions (~15% of days)
        promotions = (np.random.rand(n_days) < 0.15).astype(int)

        for i, dt in enumerate(dates):
            month = dt.month
            promo = int(promotions[i])
            is_holiday = int(holidays[i])

            # Price fluctuations: small discounts during promotions
            discount = 0.10 if promo else 0.0
            price = round(float(np.random.normal(base_price * (1.0 - discount), base_price * 0.04)), 2)
            price = max(price, round(base_price * 0.7, 2))

            # Seasonal Multiplier
            if month in peak_months:
                season_mult = 1.45 + 0.10 * np.sin(i / 14.0)
            elif month in trough_months:
                season_mult = 0.55 + 0.05 * np.cos(i / 14.0)
            else:
                season_mult = 1.00

            # Price elasticity (higher price dampens demand)
            price_ratio = price / base_price
            price_mult = max(0.65, 1.0 - 0.7 * (price_ratio - 1.0))

            # Trend Multiplier
            trend_mult = 1.0 + trend_rate * year_progression[i]

            # Promotion & Holiday uplifts
            promo_mult = 1.0 + (promo_boost if promo else 0.0)
            holiday_mult = 1.0 + (holiday_boost if is_holiday else 0.0)

            # Random variation (Gaussian noise)
            noise = np.random.normal(0, base_demand * 0.10)

            calc_demand = (
                base_demand
                * season_mult
                * price_mult
                * trend_mult
                * promo_mult
                * holiday_mult
                + noise
            )
            final_demand = int(np.clip(np.round(calc_demand), 0, None))

            records.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Product": prod_name,
                "Price": price,
                "Promotion": promo,
                "Holiday": is_holiday,
                "Demand": final_demand,
            })

    df = pd.DataFrame(records)

    # ---------------- INTRODUCE DATA QUALITY ISSUES ----------------
    # 1. Duplicate rows (~25 duplicates)
    dup_indices = np.random.choice(len(df), size=25, replace=False)
    df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

    # 2. Missing values (~2% in Price and Demand)
    n_total = len(df)
    missing_price_idx = np.random.choice(n_total, size=int(n_total * 0.02), replace=False)
    missing_demand_idx = np.random.choice(n_total, size=int(n_total * 0.02), replace=False)
    df.loc[missing_price_idx, "Price"] = np.nan
    df.loc[missing_demand_idx, "Demand"] = np.nan

    # 3. Demand outliers (~20 spikes)
    valid_demand_idx = df[df["Demand"].notna()].index
    outlier_idx = np.random.choice(valid_demand_idx, size=20, replace=False)
    for idx in outlier_idx:
        df.loc[idx, "Demand"] = float(np.random.randint(250, 420))

    # Save to data/raw/seasonal_demand_raw.csv
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print(f"Generated {len(df)} rows to {RAW_DATA_PATH}")
    print(df.head())
