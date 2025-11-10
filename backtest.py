# backtest.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils import load_data, save_data

FEAT_PATH = "./data/features.parquet"
MODEL_PATH = "./models/lgbm.pkl"   # adjust if you want elasticnet
OUT_PATH = "./data/backtest_results.csv"
OUT_PORTFOLIO = "./data/backtest_portfolio_avg.csv"

# Tuneable params
EMA_SPAN = 8                # smoothing window
THRESHOLD_SCALE = 0.5       # threshold = ±THRESHOLD_SCALE * std(signal)
TRANSACTION_COST = 0.001    # per flip cost

# ---------------------------------------------------------------------
# Helper: robust model prediction
# ---------------------------------------------------------------------
def safe_predict(model, X):
    """Unified predict that covers sklearn estimators and LightGBM boosters."""
    try:
        return model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

# ---------------------------------------------------------------------
# Helper: improved regime persistence logic
# ---------------------------------------------------------------------
def build_position_for_group(df_group, upper, lower):
    """
    Persistent regime logic with hysteresis:
    - Enters long only when signal > upper
    - Enters short only when signal < lower
    - Stays in position until crossing opposite side
    - Neutral zone = hold
    """
    signal = df_group["signal_smooth"].values
    pos = np.zeros(len(signal))
    state = 0  # 0 = neutral, 1 = long, -1 = short

    for i in range(len(signal)):
        if state == 0:
            if signal[i] > upper:
                state = 1
            elif signal[i] < lower:
                state = -1
        elif state == 1 and signal[i] < lower:
            state = 0
        elif state == -1 and signal[i] > upper:
            state = 0
        pos[i] = state

    return pd.Series(pos, index=df_group.index)

# ---------------------------------------------------------------------
# Core Backtest Function
# ---------------------------------------------------------------------
def run_backtest():
    print("[backtest] Loading features...")
    df = load_data(FEAT_PATH)
    if df is None:
        raise FileNotFoundError(f"❌ features not found at {FEAT_PATH}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["index", "Date"]).reset_index(drop=True)

    # Feature selection
    feature_cols = ["r_1w", "r_4w", "r_12w", "flow_z", "vol_4w"]
    available_feats = [c for c in feature_cols if c in df.columns]
    if not available_feats:
        raise ValueError("❌ No feature columns available for prediction")

    df = df.dropna(subset=available_feats).reset_index(drop=True)

    # Load model
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
    model = joblib.load(model_path)
    if isinstance(model, dict):
        model = model.get("model", model)

    print("[backtest] Running model predictions...")
    X = df[available_feats].values
    preds = safe_predict(model, X)
    preds = np.asarray(preds).reshape(-1)
    df["predicted_return"] = preds

    # Smoothed signal
    df["signal_smooth"] = df.groupby("index")["predicted_return"].transform(
        lambda s: s.ewm(span=EMA_SPAN, adjust=False).mean()
    )

    # Position building
    df["position"] = 0
    for name, group in df.groupby("index"):
        s = group["signal_smooth"]
        std = s.std(ddof=0)
        threshold = THRESHOLD_SCALE * (std if np.isfinite(std) and std > 0 else 1e-6)
        upper, lower = threshold, -threshold
        pos_series = build_position_for_group(group, upper, lower)
        df.loc[group.index, "position"] = pos_series.shift(1).fillna(0).astype(int)

    # Strategy returns + cost
    if "r_1w" not in df.columns:
        raise ValueError("❌ Missing r_1w (weekly return) column.")
    df["turnover"] = df.groupby("index")["position"].diff().abs().fillna(0)
    df["strategy_return"] = df["position"] * df["r_1w"] - TRANSACTION_COST * df["turnover"]

    # Portfolio compounding per index
    df["Portfolio_Value"] = df.groupby("index")["strategy_return"].transform(
        lambda s: (1 + s).cumprod().fillna(1.0)
    )
    df["buy_hold"] = df.groupby("index")["r_1w"].transform(
        lambda s: (1 + s).cumprod().fillna(1.0)
    )

    # Rolling high-watermark (removes sawtooth)
    df["Portfolio_Smooth"] = df.groupby("index")["Portfolio_Value"].transform(
        lambda x: np.maximum.accumulate(x)
    )

    # Aggregate equal-weight portfolio across indices
    agg = (
        df.groupby("Date")["Portfolio_Value"]
        .mean()
        .reset_index(name="Portfolio_Avg")
    )
    agg["Portfolio_Smooth"] = np.maximum.accumulate(agg["Portfolio_Avg"])
    save_data(agg, OUT_PORTFOLIO)
    print(f"[backtest] ✅ Saved aggregate portfolio → {OUT_PORTFOLIO}")

    # Save full detailed results
    out_cols = [
        "Date", "index", "predicted_return", "signal_smooth", "position",
        "turnover", "strategy_return", "Portfolio_Value", "Portfolio_Smooth", "buy_hold"
    ]
    save_data(df[out_cols], OUT_PATH)
    print(f"[backtest] ✅ Saved detailed results → {OUT_PATH}")

    # Summary
    final_vals = df.groupby("index")["Portfolio_Value"].last()
    print("\n[backtest] 📊 Final Portfolio Value per Index:")
    print(final_vals.round(3))
    print(f"\n[backtest] Overall Average Final Portfolio: {agg['Portfolio_Avg'].iloc[-1]:.2f}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_backtest()
