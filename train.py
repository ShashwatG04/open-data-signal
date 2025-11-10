import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from utils import ensure_data_dir, seed_all, load_data
from lightgbm import early_stopping, log_evaluation

FEAT_PATH = "./data/features.parquet"
MODEL_DIR = "./models"

def load_feats():
    df = load_data(FEAT_PATH)
    if df is None:
        raise FileNotFoundError("features.parquet missing. Run features.py first.")
    return df

def prepare_xy(df):
    feat_cols = ["r_1w","r_4w","r_12w","flow_z","vol_4w"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].values
    y = df["target"].values
    return X, y, df[["Date","index"]]

def train_elastic(X, y):
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X, y)
    return model

def train_lgb(X, y):
    dtrain, dval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = lgb.Dataset(dtrain, label=ytrain)
    dval = lgb.Dataset(dval, label=yval)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain, dval],
        callbacks=[early_stopping(20), log_evaluation(0)]
    )
    return bst

def main():
    ensure_data_dir("./models")
    seed_all(42)

    df = load_feats()
    X, y, meta = prepare_xy(df)

    elastic = train_elastic(X, y)
    joblib.dump(elastic, f"{MODEL_DIR}/elasticnet.pkl")
    print("[train] saved elasticnet.")

    lgbm = train_lgb(X, y)
    joblib.dump(lgbm, f"{MODEL_DIR}/lgbm.pkl")
    print("[train] saved lgbm.")

if __name__ == "__main__":
    main()
