import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_PATH = DATA_DIR / "raw_transactions.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

PII_COLUMNS = ["customer_name", "email", "ip_address"]
TIMESTAMP_COLS = ["order_time", "account_created_at"]
TEST_FRACTION = 0.2


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


def run_etl():
    df = load_raw_data()
    print("Raw columns:", list(df.columns))

    df = remove_pii(df)
    print("After PII removal:", list(df.columns))

    train_df, test_df = time_based_split(df)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Train rows:", len(train_df), " Test rows:", len(test_df))
    print("Train ends :", train_df["order_time"].max())
    print("Test starts:", test_df["order_time"].min())


if __name__ == "__main__":
    run_etl()