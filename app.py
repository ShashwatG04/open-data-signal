import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------------------------------------------------------
# 🎨 Page configuration and custom styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Open-Data Signals — Indian Equity Timing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS for consistent dark theme styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: "Inter", sans-serif;
        }
        h1, h2, h3, h4 {
            color: #00B4D8;
        }
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }
        .stPlotlyChart, .stLineChart {
            border-radius: 10px;
            background: #161A20;
            padding: 10px;
        }
        section[data-testid="stSidebar"] {
            background-color: #161A20;
        }
        .metric-label {
            color: #A9D6E5 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 🧠 Utility loader
# -----------------------------------------------------------------------------
def safe_load(path, kind="csv"):
    try:
        if not Path(path).exists():
            st.warning(f"⚠️ File not found: {path}")
            return None
        if kind == "csv":
            return pd.read_csv(path)
        elif kind == "parquet":
            return pd.read_parquet(path)
        elif kind == "pkl":
            return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None


# -----------------------------------------------------------------------------
# 🏗️ Load core data and models
# -----------------------------------------------------------------------------
DATA_PATH = Path("./data/features.parquet")
MODEL_PATH = Path("./models")
BACKTEST_PATH = Path("./data/backtest_results.csv")

st.title("📈 Open-Data Signals for Indian Equity Timing")
st.markdown(
    """
    ### Quantitative Strategy Simulator  
    _Modular signal engine for equity market feature extraction, ML-based forecasting, and portfolio backtesting._
    """
)

st.divider()

# -----------------------------------------------------------------------------
# 🧩 Load feature data
# -----------------------------------------------------------------------------
feats = safe_load(DATA_PATH, "parquet")
if feats is None or feats.empty:
    st.error("❌ Feature file not found or empty. Please run `python features.py` first.")
    st.stop()

st.subheader("🧩 Feature Sample")
st.dataframe(feats.head(), use_container_width=True)

# Check missing feature columns
required_cols = ["r_1w", "r_4w", "r_12w", "flow_z", "vol_4w"]
missing = [c for c in required_cols if c not in feats.columns]
if missing:
    st.warning(f"⚠️ Missing columns: {missing}")
    required_cols = [c for c in required_cols if c in feats.columns]

if not required_cols:
    st.error("No usable feature columns found. Please rebuild features.")
    st.stop()

X = feats[required_cols].values
y = feats["target"].values if "target" in feats.columns else np.zeros(len(feats))

# -----------------------------------------------------------------------------
# ⚙️ Model selection
# -----------------------------------------------------------------------------
elastic_model = safe_load(MODEL_PATH / "elasticnet.pkl", "pkl")
lgbm_model = safe_load(MODEL_PATH / "lgbm.pkl", "pkl")

if elastic_model is None and lgbm_model is None:
    st.error("⚠️ No trained models found. Please run `python train.py` first.")
    st.stop()

st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Select Model", ["ElasticNet", "LightGBM"])
model = elastic_model if model_choice == "ElasticNet" else lgbm_model

# -----------------------------------------------------------------------------
# 🤖 Generate predictions
# -----------------------------------------------------------------------------
if model is not None and hasattr(model, "predict"):
    try:
        preds = model.predict(X)
        feats["1w_fwd_predicted_return"] = preds
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader(f"📊 Predictions using {model_choice}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.line_chart(feats.set_index("Date")["1w_fwd_predicted_return"].tail(50), height=250, use_container_width=True)
    with col2:
        if "target" in feats.columns:
            corr = np.corrcoef(feats["target"], feats["1w_fwd_predicted_return"])[0, 1]
            st.metric("Correlation (Pred vs Target)", f"{corr:.2f}")

    st.markdown("#### 🔍 Latest Predictions")
    preview_cols = ["Date", "index", "1w_fwd_predicted_return"]
    preview_cols = [c for c in preview_cols if c in feats.columns]
    st.dataframe(feats[preview_cols].tail(10), use_container_width=True)

# -----------------------------------------------------------------------------
# 📉 Backtest results
# -----------------------------------------------------------------------------
st.divider()
st.subheader("💼 Backtest Performance")

backtest = safe_load("./data/backtest_portfolio_avg.csv", "csv")
if backtest is not None and not backtest.empty:
    st.line_chart(
        backtest.set_index("Date")[["Portfolio_Smooth"]],
        height=300,
        use_container_width=True
    )
    st.caption("Cumulative high-watermark (smoothed) portfolio value based on model signals.")
else:
    st.warning("⚠️ Run `python backtest.py` to generate results.")

# -----------------------------------------------------------------------------
# 🔁 Sidebar refresh
# -----------------------------------------------------------------------------
st.sidebar.divider()
if st.sidebar.button("🔁 Refresh / Rerun"):
    st.rerun()

st.sidebar.success("✅ App ready — all modules connected.")
