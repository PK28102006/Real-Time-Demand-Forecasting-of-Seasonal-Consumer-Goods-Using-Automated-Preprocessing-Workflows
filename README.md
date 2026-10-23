# Real-Time Demand Forecasting of Seasonal Consumer Goods Using Automated Preprocessing Workflows

---

## 1. Project Title
**Real-Time Demand Forecasting of Seasonal Consumer Goods Using Automated Preprocessing Workflows**

---

## 2. Introduction
Accurate demand forecasting for seasonal consumer products is critical for inventory control, minimizing stockouts, and preventing excess inventory holding costs. Real-world retail datasets, however, frequently contain missing records, duplicate logging entries, timestamps, and extreme demand anomalies. 

This project demonstrates a streamlined, college-level Data Science and Machine Learning workflow built using Python and Streamlit. The system autonomously cleans raw data, computes seasonal signatures, engineers non-leaking lag and rolling features, fits multiple regression models using chronological splitting, and outputs forward-looking multi-step forecasts with interactive Plotly visualizations.

---

## 3. Problem Statement
Consumer goods with strong seasonal demand (such as Air Conditioners, Fans, Umbrellas, Winter Jackets, Sunscreen, and School Bags) experience dramatic swings throughout the year. Manual data preprocessing is labor-intensive and prone to data leakage (e.g., shuffling temporal rows or using future observations). When new transaction data arrives, businesses need an automated preprocessing workflow that can clean the raw dataset, retrain predictive models, and produce immediate future demand forecasts.

---

## 4. Objectives
1. **Automate Data Quality Cleansing:** Autonomously inspect incoming data, handle missing values, and eliminate duplicate records.
2. **Mitigate Anomalies:** Detect and winsorize demand outliers using the statistical Interquartile Range (IQR) method.
3. **Leakage-Free Feature Engineering:** Extract calendar features, cyclical seasonal indicators ($\sin$ and $\cos$ transformations), and strictly shifted lag/rolling features.
4. **Chronological Machine Learning:** Train Linear Regression, Random Forest, and Gradient Boosting on an 80/20 chronological train/test split.
5. **Automated Best Model Selection:** Benchmark models across MAE, RMSE, and $R^2$, and deploy the model with the lowest test RMSE.
6. **Multi-Step Demand Forecasting:** Forecast 7, 14, or 30 days into the future and visualize historical and predicted trends.

---

## 5. Methodology
The project strictly implements this 5-stage sequential workflow:

```text
Raw Dataset (Synthetic or Uploaded CSV)
                  ↓
       Automated Preprocessing
  (Deduplication, Imputation, IQR Capping)
                  ↓
          Seasonality Analysis
  (Demand over time, Monthly & Product Curves)
                  ↓
          Feature Engineering
   (Date, Lags, Shifted Rolling Window)
                  ↓
     Chronological Model Training
(Linear Regression, Random Forest, Gradient Boosting)
                  ↓
       Model Evaluation & Selection
          (Lowest Test RMSE)
                  ↓
        Future Demand Forecast
   (7, 14, 30 Days Recursive Forecast)
```

---

## 6. Dataset
Because no historical dataset was initially available, the application generates a realistic synthetic dataset (`data/raw/seasonal_demand_raw.csv`) covering **3 years of daily observations** across 6 seasonal consumer goods:

| Product | Primary Seasonality Driver |
|---|---|
| **Air Conditioner** | Summer heat surges (May–July) |
| **Fan** | Warm spring/summer transition |
| **Umbrella** | Monsoon / Rainy season (June–September) |
| **Winter Jacket** | Winter cold drops (November–February) |
| **Sunscreen** | Sunny summer months (March–June) |
| **School Bag** | Academic reopening period (May–July) |

The synthetic demand explicitly accounts for:
- Product-specific baseline demand
- Calendar month seasonality curves
- Price elasticity (negative price-to-demand relationship)
- Promotional campaigns (+25% to +45% uplift)
- Holiday surges
- Long-term annual market growth trend
- Random Gaussian noise
- **Realistic Quality Flaws:** ~2% missing values, ~25 duplicate rows, and ~20 demand outliers to demonstrate automated cleansing.

*Users can also upload their own custom sales CSV files directly through the interface.*

---

## 7. Automated Preprocessing
Located in `src/preprocessing.py`, the automated preprocessing workflow performs:
1. **Structure Detection:** Identifies `Date` and `Demand` columns.
2. **Duplicate Removal:** Automatically detects and purges duplicated rows.
3. **Date Conversion:** Parses timestamps into datetime and chronologically sorts records.
4. **Missing Value Imputation:**
   - Numerical columns: Filled using **median**
   - Categorical columns: Filled using **mode**
   - Demand target: Handled via linear time-series interpolation
5. **IQR Outlier Capping:**
   $$Q_1 = \text{25th percentile}, \quad Q_3 = \text{75th percentile}, \quad \text{IQR} = Q_3 - Q_1$$
   $$\text{Lower Bound} = \max(0, Q_1 - 1.5 \times \text{IQR}), \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
   Outliers beyond the bounds are capped (winsorized) rather than dropped.
6. **Date Feature Extraction:** Derives `Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`, and `IsWeekend`.
7. **Export:** Saves clean data to `data/processed/seasonal_demand_processed.csv`.

---

## 8. Seasonality Analysis
Located in `src/seasonality.py`, this section provides 3 clear visualizations:
1. **Demand Over Time:** Daily aggregate sales trajectory over the 3-year period.
2. **Average Demand by Month:** Highlights seasonal peaks across calendar months.
3. **Average Demand by Product:** Compares demand volumes across the 6 goods.

The system dynamically computes and reports:
- **Peak Demand Month**
- **Lowest Demand Month**
- **Highest-Demand Product**
- **Lowest-Demand Product**

---

## 9. Feature Engineering
Located in `src/feature_engineering.py`, all features strictly prevent data leakage:
- **Date Features:** `Year`, `Month`, `DayOfWeek`, `Quarter`
- **Lag Features:** `Lag_1`, `Lag_7` (demand shifted strictly into past days)
- **Rolling Feature:** `Rolling_Mean_7` (shifted by 1 day prior to computing the 7-day moving average so today's demand is never included)
- **Seasonal Cyclical Features:**
  $$\text{Month\_Sin} = \sin\left(\frac{2\pi \cdot \text{Month}}{12}\right), \quad \text{Month\_Cos} = \cos\left(\frac{2\pi \cdot \text{Month}}{12}\right)$$
- **Categorical Encoding:** One-hot dummy encoding for `Product`.

---

## 10. Machine Learning Models
Located in `src/modeling.py`, three regression models are trained:
1. **Linear Regression:** Baseline parametric model.
2. **Random Forest Regressor:** Non-linear ensemble model capturing complex splits.
3. **Gradient Boosting Regressor:** Sequential boosting trees minimizing squared residuals.

### Chronological Train/Test Split
Data is **never randomly shuffled**. The first **80% of historical time** is allocated for training, and the remaining **20% of unseen future time** is used for evaluation.

---

## 11. Evaluation Metrics
Models are benchmarked on the unseen chronological test partition across:
- **MAE (Mean Absolute Error):** Average magnitude of absolute errors.
- **RMSE (Root Mean Squared Error):** Primary selection criterion penalizing larger errors.
- **$R^2$ (Coefficient of Determination):** Proportion of variance explained by features.

The model with the lowest test RMSE is automatically selected, saved with `joblib` to `models/best_model.pkl`, and used for forecasting.

---

## 12. Forecasting
Located in `src/forecasting.py`, the user selects:
- **7 days**
- **14 days**
- **30 days**

The system:
- Generates future daily calendar timestamps starting right after the last historical date.
- Employs recursive multi-step forecasting: predictions from day $t$ feed into the lag inputs of day $t+1$.
- Displays an interactive line chart connecting **Historical Demand + Forecast Demand**.
- Provides a download button to export the forecast as `forecast.csv`.

---

## 13. Technology Used
- **Language:** Python 3.10+
- **Application Framework:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Plotly
- **Model Serialization:** Joblib

---

## 14. Project Structure
```text
demand_forecasting/
│
├── app.py                     # Streamlit application (4 pages)
├── requirements.txt           # Core Python dependencies
├── README.md                  # Complete academic documentation
│
├── data/
│   ├── raw/
│   │   └── seasonal_demand_raw.csv         # Raw dataset with quality flaws
│   └── processed/
│       └── seasonal_demand_processed.csv   # Cleaned, preprocessed dataset
│
├── src/
│   ├── data_generator.py      # 3-year synthetic dataset generator
│   ├── preprocessing.py       # AutomatedPreprocessor class & IQR capping
│   ├── feature_engineering.py # Leakage-free lag, rolling, and cyclical features
│   ├── modeling.py            # Chronological split, training, evaluation, best model
│   ├── seasonality.py         # Seasonality charts and dynamic peak/trough metrics
│   └── forecasting.py         # Recursive multi-step demand forecaster
│
├── models/
│   └── best_model.pkl         # Serialized best model
│
└── outputs/
    └── forecasts/
        └── forecast.csv       # Exported future demand forecast
```

---

## 15. Installation
Open terminal in the project directory:

```bash
cd d:\Data_Science_Project
pip install -r requirements.txt
```

---

## 16. How to Run
Start the Streamlit web application:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in any web browser.

---

## 17. Expected Output
1. **Home:** Clean introduction and buttons to generate synthetic data or upload CSV.
2. **Data & Preprocessing:** Data preview, dataset shape, missing value count, and Before/After statistics table showing complete resolution of missing values, duplicates, and outliers.
3. **Analysis & Model:**
   - 3 seasonality charts (Demand Over Time, Monthly Average Demand, Product Demand)
   - Dynamic identification of peak month, lowest month, and top product
   - Leaderboard table comparing Linear Regression, Random Forest, and Gradient Boosting
   - Best model callout banner with lowest RMSE
   - Actual vs Predicted line graph
4. **Forecast:** 7/14/30 day horizon selector, recursive predictions table, interactive Historical + Forecast line chart, and downloadable `forecast.csv`.

---

## 18. Limitations
1. **Exogenous Assumptions:** Future promotional dates or price shifts are held constant during forward horizons.
2. **Aggregated Scope:** The application models daily product and aggregate demand; regional store-level granularity requires hierarchical aggregation.

---

## 19. Future Scope
1. **Deep Learning Integration:** Testing LSTM or Temporal Fusion Transformer (TFT) architectures for long-range cross-series forecasting.
2. **Real-Time API Webhooks:** Automatically fetching live transaction batches via cloud webhooks.
3. **Automated Inventory Reorder Triggers:** Recommending reorder points when forecasted demand exceeds safety stock.

---

## 20. Conclusion
This project provides a clean, self-contained Data Science and Machine Learning solution for seasonal consumer demand forecasting. By automating data cleaning, IQR outlier capping, non-leaking feature engineering, chronological model evaluation, and recursive multi-step forecasting, the system ensures reliable predictions suitable for academic demonstration, viva examinations, and portfolio presentations.
