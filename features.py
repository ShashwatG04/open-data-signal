import pandas as pd
import numpy as np
from utils import load_data, save_data

PANEL_PATH = "./data/weekly_panel.parquet"
FEAT_PATH = "./data/features.parquet"

def load_panel():
    panel = load_data(PANEL_PATH)
    if panel is None:
        raise FileNotFoundError(f"Panel file not found at {PANEL_PATH}")

    panel = panel.reset_index()

    # Rename columns for consistency
    renamed = []
    for c in panel.columns:
        if "return_1w" in c:
            renamed.append(c.replace("return_1w", "r_1w"))
        elif "flow_pressure" in c:
            renamed.append(c.replace("flow_pressure", "flow"))
        else:
            renamed.append(c)
    panel.columns = renamed

    panel["Date"] = pd.to_datetime(panel["Date"])
    print(f"[features] Loaded panel with {panel.shape[0]} rows and {panel.shape[1]} columns")
    return panel


def build_features(panel: pd.DataFrame):
    print("[features] Building features...")

    # Melt wide panel into long format
    long = panel.melt(id_vars=["Date"], var_name="metric", value_name="value")
    long["index"] = long["metric"].str.extract(r"_(Momentum|Quality|Value|SmallCap)")[0]
    long["metric_type"] = long["metric"].str.extract(r"^(price|flow|r_1w)")[0]
    long = long.dropna(subset=["index", "metric_type"])

    long = long.pivot_table(
        index=["Date", "index"],
        columns="metric_type",
        values="value",
        aggfunc="mean"
    ).reset_index()
    long = long.sort_values(["index", "Date"])

    # Compute rolling and z-score features
    def compute_group(df):
        df = df.copy()
        df["r_4w"] = df["r_1w"].rolling(4, min_periods=1).mean()
        df["r_12w"] = df["r_1w"].rolling(12, min_periods=1).mean()
        df["vol_4w"] = df["r_1w"].rolling(4, min_periods=1).std()
        df["flow_z"] = (df["flow"] - df["flow"].rolling(12, min_periods=1).mean()) / (
            df["flow"].rolling(12, min_periods=1).std() + 1e-6
        )
        return df

    long = long.groupby("index", group_keys=False).apply(compute_group).reset_index(drop=True)

    # Target = next week's return
    long["target"] = long.groupby("index")["r_1w"].shift(-1)

    # NEW 🔥: keep last rows even if they have no target (future prediction)
    long["is_future_prediction"] = long["target"].isna()
    long["prediction_date"] = long["Date"] + pd.to_timedelta(7, unit="D")

    # Fill NaNs safely for model use
    long = long.fillna(method="ffill").fillna(method="bfill")

    print(f"[features] Final feature set shape: {long.shape}")
    print(f"[features] Keeping {long['is_future_prediction'].sum()} future rows for next-week prediction.")
    return long


def save_features(feats):
    save_data(feats, FEAT_PATH)
    print(f"[features] Saved features → {FEAT_PATH}")


if __name__ == "__main__":
    panel = load_panel()
    feats = build_features(panel)
    save_features(feats)
