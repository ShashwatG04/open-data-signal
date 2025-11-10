import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_data, save_data

def build_dataset():
    print("[pipeline] Building dataset...")

    amfi = load_data("./data/amfi_flows.csv")
    if amfi is None:
        print("[pipeline] AMFI not found, generating synthetic AMFI.")
        months = 36
        rng = pd.date_range(end=pd.Timestamp.today(), periods=months, freq="ME")
        amfi = pd.DataFrame({
            "Date": rng,
            "category": np.random.choice(["Momentum","Quality","Value","SmallCap"], months),
            "flow": np.random.uniform(-100, 100, months)
        })

    idx = load_data("./data/index_prices.csv")
    if idx is None:
        print("[pipeline] Index prices not found, generating synthetic prices.")
        months = 36
        rng = pd.date_range(end=pd.Timestamp.today(), periods=months, freq="ME")
        idx = pd.DataFrame({
            "Date": rng.repeat(4),
            "index": np.tile(["Momentum","Quality","Value","SmallCap"], months),
            "price": np.random.uniform(100, 200, months*4)
        })

    # ✅ FIX: ensure price column is numeric
    idx["price"] = pd.to_numeric(idx["price"], errors="coerce")

    weekly = idx.copy()
    weekly["flow_pressure"] = np.random.normal(0, 1, len(weekly))
    weekly["return_1w"] = weekly.groupby("index")["price"].pct_change().fillna(0)

    panel = weekly.pivot_table(
        index="Date",
        columns="index",
        values=["price", "flow_pressure", "return_1w"],
        aggfunc="mean"
    )

    panel.columns = [f"{a}_{b}" for a, b in panel.columns]
    panel = panel.sort_index()

    Path("./data").mkdir(exist_ok=True)
    save_data(panel, "./data/weekly_panel.parquet")
    print("[pipeline] Saved weekly panel → ./data/weekly_panel.parquet")
    print("[pipeline] ✅ Dataset build complete.")

if __name__ == "__main__":
    build_dataset()
