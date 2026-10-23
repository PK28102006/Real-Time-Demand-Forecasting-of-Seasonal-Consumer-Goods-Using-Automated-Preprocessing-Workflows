"""
app.py
Real-Time Demand Forecasting of Seasonal Consumer Goods Using Automated Preprocessing Workflows
Clean, modern, and professional Streamlit application providing 4 focused pages:
1. Home
2. Data & Preprocessing
3. Analysis & Model
4. Forecast
"""

from io import BytesIO
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Internal pipeline modules
from src.data_generator import generate_synthetic_dataset, RAW_DATA_PATH
from src.preprocessing import AutomatedPreprocessor, PROCESSED_DATA_PATH
from src.feature_engineering import engineer_features
from src.seasonality import (
    calculate_seasonality_insights,
    create_demand_over_time_chart,
    create_monthly_demand_chart,
    create_product_demand_chart,
)
from src.modeling import (
    chronological_split,
    train_and_evaluate_models,
    create_actual_vs_predicted_chart,
    MODEL_SAVE_PATH,
)
from src.forecasting import generate_future_forecast, create_forecast_chart, FORECAST_CSV_PATH

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Seasonal Demand Forecasting | Automated Preprocessing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html_str: str):
    """
    Safely render HTML in Streamlit by stripping leading indentation from each line.
    Prevents Python Markdown from mistakenly converting indented HTML into <pre><code> blocks.
    """
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ----------------- CLEAN & MODERN CSS STYLING -----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Typography & Deep Contrast */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
        color: #0F172A !important;
        background-color: #F8FAFC !important;
    }

    /* Force ALL Headings to High-Contrast Dark */
    h1, h2, h3, h4, h5, h6, 
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3,
    [data-testid="stHeadingWithActionElements"] h4 {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* Force ALL text, paragraphs, spans, lists to Dark Slate */
    .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div,
    p, span, li, label, strong {
        color: #1E293B !important;
    }

    /* Sidebar full text contrast */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] li, 
    section[data-testid="stSidebar"] strong, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] [class*="css"] {
        color: #1E293B !important;
    }

    /* Metric Cards: Value and Label */
    div[data-testid="stMetric"], div[data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover, div[data-testid="metric-container"]:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stMetricValue"], 
    div[data-testid="stMetricValue"] *, 
    [data-testid="stMetricValue"] div {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.7rem !important;
    }
    div[data-testid="stMetricLabel"], 
    div[data-testid="stMetricLabel"] *, 
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] label,
    [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Badge Pill */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background-color: #EFF6FF;
        color: #2563EB;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        border: 1px solid #BFDBFE;
        margin-bottom: 0.6rem;
    }
    
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.25;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    
    .project-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.6rem;
        line-height: 1.6;
        color: #334155;
    }
    
    /* Clean Card Container */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        margin-bottom: 1.25rem;
    }
    
    /* Primary Action Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-1px);
    }
    
    /* Secondary & Download Buttons */
    button[kind="secondary"], .stDownloadButton>button {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-weight: 600 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover, .stDownloadButton>button:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }
    
    /* Best Model Champion Card */
    .champion-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1.5px solid #93C5FD;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .champion-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1E40AF;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .champion-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 0.75rem;
    }
    .champion-stat {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(147, 197, 253, 0.6);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    .stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #3B82F6;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1E3A8A;
    }
    
    /* Sidebar Brand */
    .sidebar-brand {
        padding: 0.5rem 0 0.8rem 0;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 0.8rem;
    }
    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 500;
    }
    
    /* Sidebar Radio Button High Contrast Styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.75rem !important;
        margin-bottom: 0.35rem !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #EFF6FF !important;
        border-color: #93C5FD !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #EFF6FF !important;
        border-color: #2563EB !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #1D4ED8 !important;
        font-weight: 800 !important;
    }
    
    /* Custom status chip */
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #059669;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------- SESSION STATE MANAGEMENT -----------------
def initialize_state():
    if "raw_df" not in st.session_state:
        if RAW_DATA_PATH.exists():
            st.session_state.raw_df = pd.read_csv(RAW_DATA_PATH)
        else:
            st.session_state.raw_df = generate_synthetic_dataset()

    if "clean_df" not in st.session_state:
        st.session_state.clean_df = None
    if "audit" not in st.session_state:
        st.session_state.audit = None
    if "featured_df" not in st.session_state:
        st.session_state.featured_df = None
    if "feature_cols" not in st.session_state:
        st.session_state.feature_cols = []
    if "train_df" not in st.session_state:
        st.session_state.train_df = None
    if "test_df" not in st.session_state:
        st.session_state.test_df = None
    if "comparison_df" not in st.session_state:
        st.session_state.comparison_df = None
    if "best_model_name" not in st.session_state:
        st.session_state.best_model_name = None
    if "best_metrics" not in st.session_state:
        st.session_state.best_metrics = None
    if "best_model_obj" not in st.session_state:
        st.session_state.best_model_obj = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}
    if "forecast_df" not in st.session_state:
        st.session_state.forecast_df = None


initialize_state()


def run_pipeline_if_needed():
    """Ensure data is preprocessed and initial models trained so all pages display real calculated values."""
    if st.session_state.clean_df is None and st.session_state.raw_df is not None:
        prep = AutomatedPreprocessor()
        clean, audit = prep.process(st.session_state.raw_df)
        st.session_state.clean_df = clean
        st.session_state.audit = audit

        # Feature engineering
        feat_df, cols = engineer_features(clean)
        st.session_state.featured_df = feat_df
        st.session_state.feature_cols = cols

        # Chronological split
        tr_df, te_df, _ = chronological_split(feat_df)
        st.session_state.train_df = tr_df
        st.session_state.test_df = te_df

        # Models training & evaluation
        comp_df, b_name, b_metrics, preds, b_obj = train_and_evaluate_models(tr_df, te_df, cols)
        st.session_state.comparison_df = comp_df
        st.session_state.best_model_name = b_name
        st.session_state.best_metrics = b_metrics
        st.session_state.predictions = preds
        st.session_state.best_model_obj = b_obj


run_pipeline_if_needed()


# ----------------- CLEAN SIDEBAR NAVIGATION -----------------
with st.sidebar:
    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">📈 DemandForecast AI</div>
            <div class="sidebar-brand-sub">Seasonal Consumer Goods Machine Learning</div>
        </div>
        """
    )

    render_html('<div class="status-chip">● System Active & Calibrated</div>')
    st.write("")

    page = st.radio(
        "Navigation",
        options=[
            "Home",
            "Data & Preprocessing",
            "Analysis & Model",
            "Forecast",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        **Pipeline Pillars:**
        - 🧹 **Auto Preprocessing:** Deduplication, median/mode imputation & IQR capping.
        - 📅 **Leakage-Free Features:** Cyclical sine/cosine, autoregressive lags & shifted rolling means.
        - ⏱ **Chronological Split:** 80% train / 20% test.
        - 🔮 **Multi-Step Forecast:** Recursive forward predictions.
        """
    )


# ----------------- WORKFLOW TIMELINE COMPONENT -----------------
def render_timeline(active_step_idx: int):
    steps = [
        ("Step 1", "Raw Data"),
        ("Step 2", "Auto Clean"),
        ("Step 3", "Seasonality"),
        ("Step 4", "ML Modeling"),
        ("Step 5", "Forecasting"),
    ]
    cols = st.columns(5)
    for i, (col, (num, title)) in enumerate(zip(cols, steps)):
        is_active = (i == active_step_idx)
        border_col = "#2563EB" if is_active else "#E2E8F0"
        bg_col = "#EFF6FF" if is_active else "#FFFFFF"
        text_col = "#1D4ED8" if is_active else "#1E293B"
        badge_col = "#2563EB" if is_active else "#64748B"
        col.markdown(
            f'<div style="background:{bg_col}; border:1.5px solid {border_col}; border-radius:10px; padding:0.65rem 0.5rem; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.03);">'
            f'<div style="font-size:0.7rem; font-weight:800; color:{badge_col}; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.2rem;">{num}</div>'
            f'<div style="font-size:0.85rem; font-weight:700; color:{text_col};">{title}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.write("")


# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if page == "Home":
    render_timeline(0)

    render_html('<div class="badge-pill">Automated Machine Learning Workflow</div>')
    render_html('<div class="hero-title">Real-Time Demand Forecasting of Seasonal Consumer Goods Using Automated Preprocessing Workflows</div>')
    render_html('<div class="hero-subtitle">An automated pipeline that cleans raw consumer sales data, detects seasonal trajectories, trains machine learning models, and outputs recursive future forecasts.</div>')

    render_html(
        """
        <div class="project-card">
            <strong>System Overview:</strong> This system automatically preprocesses seasonal consumer demand data,
            analyzes seasonal patterns, trains machine learning models, selects the best model and forecasts future demand.
        </div>
        """
    )

    # Dataset Ingestion Options
    st.markdown("### ⚡ Dataset Ingestion & Configuration")
    col1, col2 = st.columns(2)

    with col1:
        render_html(
            """
            <div class="clean-card">
                <h4 style="margin-top:0; color:#1E293B; font-weight:700;">🔄 Generate Demo Dataset</h4>
                <p style="color:#64748B; font-size:0.9rem; margin-bottom:1rem;">
                    Synthesizes 3 full calendar years (2022–2024) of daily demand across 6 seasonal goods with authentic data flaws.
                </p>
            </div>
            """
        )
        if st.button("Generate Demo Dataset ⚡", type="primary", use_container_width=True):
            with st.spinner("Generating fresh synthetic dataset..."):
                new_df = generate_synthetic_dataset()
                st.session_state.raw_df = new_df
                st.session_state.clean_df = None
                run_pipeline_if_needed()
                st.success("Synthetic dataset generated and pipeline refreshed!")
                st.rerun()

    with col2:
        render_html(
            """
            <div class="clean-card">
                <h4 style="margin-top:0; color:#1E293B; font-weight:700;">📤 Upload Your Own CSV</h4>
                <p style="color:#64748B; font-size:0.9rem; margin-bottom:0.6rem;">
                    Load a custom CSV file containing historical sales records with Date and Demand columns.
                </p>
            </div>
            """
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                up_df = pd.read_csv(uploaded_file)
                if len(up_df) < 30:
                    st.error("Uploaded CSV has too few rows (minimum 30 rows required for time-series forecasting).")
                else:
                    if st.button("Apply Uploaded CSV & Run Pipeline", use_container_width=True):
                        st.session_state.raw_df = up_df
                        st.session_state.clean_df = None
                        run_pipeline_if_needed()
                        st.success("Uploaded CSV loaded and pipeline executed!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.markdown("---")
    st.markdown("### 📋 Quick Pipeline Summary")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        render_html(
            """
            <div class="clean-card">
                <strong style="color:#2563EB;">1. Ingestion & Quality Audit</strong>
                <p style="font-size:0.88rem; color:#475569; margin-top:0.4rem; margin-bottom:0;">
                    Automatic schema parsing, missing value checks, duplicate row identification, and anomaly profiling.
                </p>
            </div>
            """
        )
    with sum_col2:
        render_html(
            """
            <div class="clean-card">
                <strong style="color:#2563EB;">2. Automated Preprocessing</strong>
                <p style="font-size:0.88rem; color:#475569; margin-top:0.4rem; margin-bottom:0;">
                    Statistical median/mode imputation, time-series interpolation, and statistical IQR outlier capping.
                </p>
            </div>
            """
        )
    with sum_col3:
        render_html(
            """
            <div class="clean-card">
                <strong style="color:#2563EB;">3. Modeling & Forecasting</strong>
                <p style="font-size:0.88rem; color:#475569; margin-top:0.4rem; margin-bottom:0;">
                    Chronological 80/20 train/test evaluation, automatic lowest RMSE selection, and recursive forward projections.
                </p>
            </div>
            """
        )


# ==============================================================================
# PAGE 2: DATA & PREPROCESSING
# ==============================================================================
elif page == "Data & Preprocessing":
    render_timeline(1)

    render_html('<div class="badge-pill">Data Hygiene Engine</div>')
    render_html('<div class="hero-title">Data Inspection & Automated Preprocessing</div>')
    render_html('<div class="hero-subtitle">The core contribution: autonomous cleansing of raw transactional data to eliminate quality flaws without data leakage.</div>')

    raw_df = st.session_state.raw_df
    prep = AutomatedPreprocessor()
    inspection = prep.inspect(raw_df)

    # 1. Dataset Shape & Quality KPIs
    st.markdown("### 📊 Raw Dataset Profile")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Total Rows", f"{inspection['rows']:,}")
    with col_s2:
        st.metric("Features / Columns", f"{inspection['columns']}")
    with col_s3:
        st.metric("Missing Values", f"{inspection['missing_values']:,}")
    with col_s4:
        st.metric("Duplicate Rows", f"{inspection['duplicates']:,}")

    # Dataset Preview
    st.markdown("#### 🔍 Raw Dataset Sample (First 10 Records)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.markdown("---")

    # 2. Automated Preprocessing Steps
    st.markdown("### ⚙️ Automated Preprocessing Execution")
    render_html(
        """
        <div class="clean-card">
            <strong>Standardized Cleansing Sequence:</strong>
            <ol style="margin-top:0.5rem; margin-bottom:0; color:#475569; line-height:1.6;">
                <li><strong>Dataset Structure Detection:</strong> Verifies date and target columns automatically.</li>
                <li><strong>Duplicate Record Removal:</strong> Identifies and removes repeated transactional rows.</li>
                <li><strong>Timestamp Conversion:</strong> Converts date strings to datetime and chronologically sorts records.</li>
                <li><strong>Missing Value Imputation:</strong> Uses <em>median</em> for numeric attributes, <em>mode</em> for categories, and linear interpolation for demand series.</li>
                <li><strong>IQR Outlier Capping (Winsorization):</strong> Calculates Q1, Q3, and IQR, capping extreme spikes without discarding historical continuity.</li>
                <li><strong>Date Feature Extraction:</strong> Derives Year, Month, Day, DayOfWeek, Quarter, and IsWeekend.</li>
            </ol>
        </div>
        """
    )

    if st.button("Run Automated Preprocessing 🚀", type="primary"):
        with st.spinner("Executing automated preprocessing pipeline..."):
            clean_df, audit = prep.process(raw_df)
            st.session_state.clean_df = clean_df
            st.session_state.audit = audit

            feat_df, cols = engineer_features(clean_df)
            st.session_state.featured_df = feat_df
            st.session_state.feature_cols = cols

            tr_df, te_df, _ = chronological_split(feat_df)
            st.session_state.train_df = tr_df
            st.session_state.test_df = te_df

            comp_df, b_name, b_metrics, preds, b_obj = train_and_evaluate_models(tr_df, te_df, cols)
            st.session_state.comparison_df = comp_df
            st.session_state.best_model_name = b_name
            st.session_state.best_metrics = b_metrics
            st.session_state.predictions = preds
            st.session_state.best_model_obj = b_obj
            st.success("Automated Preprocessing executed successfully! Quality flaws resolved to 0.")

    # 3. Before vs After Statistics
    audit = st.session_state.audit
    clean_df = st.session_state.clean_df

    if audit is not None and clean_df is not None:
        st.markdown("### 📈 Quality Audit: Before vs After Preprocessing")
        comp_table = audit["comparison_table"]

        col_tbl, col_chips = st.columns([3, 2])
        with col_tbl:
            st.dataframe(comp_table, use_container_width=True, hide_index=True)

        with col_chips:
            render_html(
                f"""
                <div class="clean-card" style="border-left: 4px solid #10B981;">
                    <div style="font-weight:700; color:#065F46; font-size:1rem; margin-bottom:0.4rem;">✅ Automated Cleansing Audit</div>
                    <div style="font-size:0.9rem; color:#374151;">
                        • Missing values: <strong>{audit['missing_before']} → {audit['missing_after']}</strong><br>
                        • Duplicate rows: <strong>{audit['duplicates_before']} → {audit['duplicates_after']}</strong><br>
                        • Demand outliers: <strong>{audit['outliers_before']} → {audit['outliers_after']}</strong><br>
                        • Total clean records: <strong>{audit['rows_after']:,} rows</strong>
                    </div>
                </div>
                """
            )

        st.markdown("---")

        # 4. Clean Processed Dataset Preview & Download
        st.markdown("### 📁 Processed & Cleaned Dataset (First 10 Records)")
        st.dataframe(clean_df.head(10), use_container_width=True)

        buf = BytesIO()
        clean_df.to_csv(buf, index=False)
        st.download_button(
            label="📥 Download Processed Dataset (CSV)",
            data=buf.getvalue(),
            file_name="seasonal_demand_processed.csv",
            mime="text/csv",
        )


# ==============================================================================
# PAGE 3: ANALYSIS & MODEL
# ==============================================================================
elif page == "Analysis & Model":
    render_timeline(2)

    render_html('<div class="badge-pill">Exploratory Insights & Machine Learning</div>')
    render_html('<div class="hero-title">Seasonality Analysis & Model Evaluation</div>')
    render_html('<div class="hero-subtitle">Discovering seasonal trajectories and benchmarking Linear Regression, Random Forest, and Gradient Boosting on chronological test data.</div>')

    clean_df = st.session_state.clean_df
    if clean_df is None:
        st.warning("Please preprocess data first.")
    else:
        # 1. Seasonality Analysis
        st.markdown("### 🗓 Seasonal Drivers & Metrics")
        insights = calculate_seasonality_insights(clean_df)

        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        with col_i1:
            st.metric("Peak Demand Month", f"☀️ {insights['peak_month']}", f"{insights['peak_month_val']} avg units/day")
        with col_i2:
            st.metric("Lowest Demand Month", f"❄️ {insights['lowest_month']}", f"{insights['lowest_month_val']} avg units/day")
        with col_i3:
            st.metric("Top-Demand Product", f"🏆 {insights['highest_product']}", f"{insights['highest_product_demand']:,} total units")
        with col_i4:
            st.metric("Lowest-Demand Product", f"📉 {insights['lowest_product']}", f"{insights['lowest_product_demand']:,} total units")

        st.markdown("#### 📈 Visualizing Seasonal Trajectories")
        # Chart 1: Demand over time
        fig1 = create_demand_over_time_chart(clean_df)
        st.plotly_chart(fig1, use_container_width=True)

        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            # Chart 2: Monthly Demand
            fig2 = create_monthly_demand_chart(clean_df)
            st.plotly_chart(fig2, use_container_width=True)

        with col_ch2:
            # Chart 3: Product Demand
            fig3 = create_product_demand_chart(clean_df)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # 2. Machine Learning Modeling & Chronological Split
        st.markdown("### 🤖 Machine Learning Modeling")
        render_html(
            """
            <div class="clean-card">
                <strong>Strict Chronological Split (80% Train / 20% Unseen Test):</strong><br>
                <span style="color:#64748B; font-size:0.9rem;">
                    To prevent lookahead bias and temporal data leakage, data is not randomly shuffled.
                    Models are trained exclusively on past observations and validated against the future 20% test period.
                </span>
            </div>
            """
        )

        if st.button("Train Models ⚡", type="primary"):
            with st.spinner("Training Linear Regression, Random Forest, and Gradient Boosting..."):
                feat_df, cols = engineer_features(clean_df)
                st.session_state.featured_df = feat_df
                st.session_state.feature_cols = cols

                tr_df, te_df, _ = chronological_split(feat_df)
                st.session_state.train_df = tr_df
                st.session_state.test_df = te_df

                comp_df, b_name, b_metrics, preds, b_obj = train_and_evaluate_models(tr_df, te_df, cols)
                st.session_state.comparison_df = comp_df
                st.session_state.best_model_name = b_name
                st.session_state.best_metrics = b_metrics
                st.session_state.predictions = preds
                st.session_state.best_model_obj = b_obj
                st.success("All models trained and evaluated successfully!")

        comp_df = st.session_state.comparison_df
        b_name = st.session_state.best_model_name
        b_metrics = st.session_state.best_metrics
        test_df = st.session_state.test_df
        preds = st.session_state.predictions

        if comp_df is not None and b_metrics is not None:
            st.markdown("#### 🏆 Model Performance Comparison")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Best Model Champion Card
            render_html(
                f"""
                <div class="champion-card">
                    <div class="champion-title">🏆 BEST MODEL: {b_name.upper()}</div>
                    <div style="color:#1E3A8A; font-size:0.92rem;">
                        Automatically selected as the top-performing model with the <strong>lowest RMSE</strong> on the chronological test dataset.
                    </div>
                    <div class="champion-grid">
                        <div class="champion-stat">
                            <div class="stat-label">Root Mean Squared Error (RMSE)</div>
                            <div class="stat-value">{b_metrics['rmse']:.3f}</div>
                        </div>
                        <div class="champion-stat">
                            <div class="stat-label">Mean Absolute Error (MAE)</div>
                            <div class="stat-value">{b_metrics['mae']:.3f}</div>
                        </div>
                        <div class="champion-stat">
                            <div class="stat-label">Coefficient of Determination (R²)</div>
                            <div class="stat-value">{b_metrics['r2']:.4f}</div>
                        </div>
                    </div>
                </div>
                """
            )

            # Actual vs Predicted Graph
            st.markdown("#### 🎯 Actual vs Predicted Demand (Chronological Test Set)")
            fig_act_pred = create_actual_vs_predicted_chart(
                test_df=test_df,
                date_col="Date",
                demand_col="Demand",
                predictions=preds,
                model_name=b_name,
            )
            st.plotly_chart(fig_act_pred, use_container_width=True)


# ==============================================================================
# PAGE 4: FORECAST
# ==============================================================================
elif page == "Forecast":
    render_timeline(4)

    render_html('<div class="badge-pill">Predictive Planning</div>')
    render_html('<div class="hero-title">Future Demand Forecasting</div>')
    render_html('<div class="hero-subtitle">Recursive multi-step forecasting deployed using the optimal machine learning model without data leakage.</div>')

    clean_df = st.session_state.clean_df
    b_name = st.session_state.best_model_name
    b_obj = st.session_state.best_model_obj
    f_cols = st.session_state.feature_cols

    if clean_df is None or b_obj is None:
        st.warning("Please complete data preprocessing and model training first.")
    else:
        col_opt1, col_opt2 = st.columns([1, 2])
        with col_opt1:
            render_html(
                """
                <div class="clean-card" style="margin-bottom:0.75rem;">
                    <strong style="color:#0F172A;">Forecast Horizon Window:</strong>
                </div>
                """
            )
            horizon_choice = st.radio(
                "Select Horizon:",
                options=["7 days", "14 days", "30 days"],
                index=1,
                label_visibility="collapsed",
            )
            horizon_map = {"7 days": 7, "14 days": 14, "30 days": 30}
            horizon_days = horizon_map[horizon_choice]

        with col_opt2:
            render_html(
                f"""
                <div class="clean-card" style="margin-bottom:0.75rem;">
                    <div style="font-size:0.85rem; color:#64748B;">Active Forecast Engine:</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#1E40AF; margin-bottom:0.3rem;">{b_name}</div>
                    <div style="font-size:0.85rem; color:#475569;">
                        Utilizing <strong>{len(f_cols)}</strong> date, cyclical, lag, and rolling features iteratively updated day-by-day.
                    </div>
                </div>
                """
            )
            generate_clicked = st.button("Generate Forecast 🚀", type="primary", use_container_width=True)

        if generate_clicked or st.session_state.forecast_df is None:
            with st.spinner(f"Generating recursive {horizon_days}-day demand forecast..."):
                _, agg_fc = generate_future_forecast(
                    model=b_obj,
                    feature_cols=f_cols,
                    historical_df=clean_df,
                    horizon_days=horizon_days,
                )
                st.session_state.forecast_df = agg_fc

        agg_fc = st.session_state.forecast_df
        if agg_fc is not None:
            st.markdown("---")
            # 1. Forecast Graph
            st.markdown(f"### 📈 Historical Demand + Future Forecast Trajectory ({horizon_choice})")
            fig_forecast = create_forecast_chart(
                historical_df=clean_df,
                forecast_df=agg_fc,
                lookback_days=90,
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # 2. Forecast Table & Download
            st.markdown("### 📋 Predicted Demand Table")
            st.dataframe(agg_fc, use_container_width=True)

            fc_buf = BytesIO()
            agg_fc.to_csv(fc_buf, index=False)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=fc_buf.getvalue(),
                file_name="forecast.csv",
                mime="text/csv",
            )
