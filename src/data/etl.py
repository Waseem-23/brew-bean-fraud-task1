import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.features.transformers import FutureTimestampGuard, FraudFeatureEngineer

DATA_DIR = Path(__file__).parent
RAW_PATH = DATA_DIR / "raw_transactions.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
PIPELINE_PATH = DATA_DIR / "fraud_pipeline.joblib"

PII_COLUMNS = ["customer_name", "email", "ip_address"]
TIMESTAMP_COLS = ["order_time", "account_created_at"]
TEST_FRACTION = 0.2

NUMERIC_FEATURES = [
    "amount", "account_age_days", "order_hour",
    "high_amount_flag", "customer_txn_velocity_24h",
]
CATEGORICAL_FEATURES = ["payment_method", "merchant_category", "customer_country"]


def load_raw_data():
    return pd.read_csv(RAW_PATH, parse_dates=TIMESTAMP_COLS)


def remove_pii(df):
    return df.drop(columns=PII_COLUMNS)


def time_based_split(df, time_col="order_time", test_fraction=TEST_FRACTION):
    df = df.sort_values(time_col).reset_index(drop=True)
    split_point = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_point].copy()
    test_df = df.iloc[split_point:].copy()
    assert train_df[time_col].max() <= test_df[time_col].min()
    return train_df, test_df


def build_pipeline(reference_time):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("leakage_guard", FutureTimestampGuard(
                timestamp_cols=TIMESTAMP_COLS,
                reference_time=reference_time,
            )),
            ("feature_engineering", FraudFeatureEngineer()),
            ("preprocess", preprocessor),
        ]
    )


def run_etl():
    df = load_raw_data()
    df = remove_pii(df)

    train_df, test_df = time_based_split(df)

    X_train = train_df.drop(columns=["is_fraud"])
    X_test = test_df.drop(columns=["is_fraud"])

    reference_time = df[TIMESTAMP_COLS].max().max()
    pipeline = build_pipeline(reference_time)
    pipeline.fit(X_train)

    guard = pipeline.named_steps["leakage_guard"]
    engineer = pipeline.named_steps["feature_engineering"]

    train_out = engineer.transform(guard.transform(X_train))
    test_out = engineer.transform(guard.transform(X_test))
    train_out["is_fraud"] = train_df["is_fraud"].values
    test_out["is_fraud"] = test_df["is_fraud"].values

    train_out.to_csv(TRAIN_PATH, index=False)
    test_out.to_csv(TEST_PATH, index=False)
    joblib.dump(pipeline, PIPELINE_PATH)

    encoded_train = pipeline.transform(X_train)
    encoded_test = pipeline.transform(X_test)

    print("Train rows:", len(train_out), " Test rows:", len(test_out))
    print("Train ends :", train_out["order_time"].max())
    print("Test starts:", test_out["order_time"].min())
    print("Columns in train.csv:", list(train_out.columns))
    print("Encoded shape (train):", encoded_train.shape)
    print("Encoded shape (test): ", encoded_test.shape)
    print("Saved: train.csv, test.csv, fraud_pipeline.joblib")


if __name__ == "__main__":
    run_etl()