from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.transformers import (
    FraudFeatureEngineer,
    FutureTimestampGuard,
    LeakageFlagger,
)
from src.make_splits import create_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

PII_COLUMNS = ["customer_name", "email", "ip_address"]
TIMESTAMP_COLS = ["order_time", "account_created_at"]
NUMERIC_FEATURES = [
    "amount", "account_age_days", "order_hour",
    "high_amount_flag", "customer_txn_velocity_24h",
]
CATEGORICAL_FEATURES = ["payment_method", "merchant_category", "customer_country"]


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(rel_path):
    return PROJECT_ROOT / rel_path


def load_raw_data(cfg):
    return pd.read_csv(resolve(cfg["paths"]["raw_data"]), parse_dates=TIMESTAMP_COLS)


def remove_pii(df):
    return df.drop(columns=PII_COLUMNS)


def add_customer_velocity(df, window_hours=24):
    """Har order se pehle, pichhle 24 ghanton mein us customer ke kitne orders the.
    Sirf past dekhta hai, is liye split se pehle poore data par nikalna safe hai."""
    df = df.sort_values("order_time").reset_index(drop=True)
    velocity = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("customer_id"):
        t = g["order_time"].values
        left = np.searchsorted(t, t - np.timedelta64(window_hours, "h"), side="left")
        velocity[g.index.values] = np.arange(len(t)) - left
    df["customer_txn_velocity_24h"] = velocity
    return df


def clean_data(df, target="is_fraud"):
    df = df.drop_duplicates(subset="transaction_id")
    df = df.dropna(subset=TIMESTAMP_COLS + ["amount", target])
    df = df[df["amount"] > 0]
    df = df[df["account_created_at"] <= df["order_time"]]  # account order ke baad nahi ban sakta
    df = remove_pii(df)
    return add_customer_velocity(df)


def build_pipeline(reference_time, threshold=0.8):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return Pipeline(
        steps=[
            ("leakage_guard", FutureTimestampGuard(TIMESTAMP_COLS, reference_time)),
            ("feature_engineering", FraudFeatureEngineer()),
            ("preprocess", preprocessor),
            ("leakage_flagger", LeakageFlagger(threshold=threshold)),
        ]
    )


def run_etl(config_path=CONFIG_PATH):
    cfg = load_config(config_path)
    target = cfg["target_column"]
    split_cfg = cfg["split"]

    df = clean_data(load_raw_data(cfg), target)
    y = df[target]
    X = df.drop(columns=[target])

    train_idx, val_idx, test_idx = create_splits(
        y,
        split_cfg["train_size"], split_cfg["val_size"], split_cfg["test_size"],
        cfg["random_state"],
        resolve(cfg["paths"]["splits_dir"]),
    )

    # Pipeline sirf TRAIN par fit hoti hai
    pipeline = build_pipeline(
        reference_time=df["order_time"].max(),
        threshold=cfg["leakage"]["correlation_threshold"],
    )
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

    out_dir = resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        features = pipeline.transform(X.iloc[idx])
        features[target] = y.iloc[idx].values
        features.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"{name}: {features.shape}, fraud rate = {features[target].mean():.3f}")

    joblib.dump(pipeline, resolve(cfg["paths"]["pipeline_artifact"]))
    flagged = pipeline.named_steps["leakage_flagger"].flagged_cols_
    print("Leakage flagged columns:", flagged or "none")


if __name__ == "__main__":
    run_etl()