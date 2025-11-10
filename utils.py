import pandas as pd
from pathlib import Path

def load_data(path: str):
    """Safely load a CSV or Parquet file if it exists, else return None."""
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"[utils] could not load {path}: file not found.")
            return None
        if path_obj.suffix in [".csv", ".txt"]:
            df = pd.read_csv(path_obj)
        elif path_obj.suffix in [".parquet"]:
            df = pd.read_parquet(path_obj)
        else:
            print(f"[utils] Unsupported file type for {path}")
            return None
        print(f"[utils] loaded {path} successfully ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"[utils] could not load {path}: {e}")
        return None


def save_data(df: pd.DataFrame, path: str):
    """Save DataFrame as Parquet or CSV, depending on extension."""
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(exist_ok=True)
        if path_obj.suffix == ".csv":
            df.to_csv(path_obj, index=False)
        elif path_obj.suffix == ".parquet":
            df.to_parquet(path_obj)
        else:
            df.to_csv(str(path_obj) + ".csv", index=False)
        print(f"[utils] saved → {path}")
    except Exception as e:
        print(f"[utils] could not save {path}: {e}")

import os, random, numpy as np

def ensure_data_dir(path="./data"):
    """Ensure data directory exists."""
    os.makedirs(path, exist_ok=True)

def load_csv_safe(path):
    """Safely load CSV or return None if not found."""
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            print(f"[utils] file not found: {path}")
            return None
    except Exception as e:
        print(f"[utils] could not load CSV {path}: {e}")
        return None

def seed_all(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
