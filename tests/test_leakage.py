import pandas as pd
import pytest
from datetime import timedelta

from src.data.etl import PII_COLUMNS, remove_pii, time_based_split
from src.features.transformers import FutureTimestampGuard, FraudFeatureEngineer


def make_df(n=20):
    times = pd.date_range(start="2026-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(n)],
        "customer_id": [1000 + (i % 3) for i in range(n)],
        "customer_name": [f"Name{i}" for i in range(n)],
        "email": [f"e{i}@x.com" for i in range(n)],
        "ip_address": ["1.2.3.4"] * n,
        "order_time": times,
        "account_created_at": times - timedelta(days=30),
        "amount": [10.0 + i for i in range(n)],
        "payment_method": ["wallet"] * n,
        "merchant_category": ["delivery"] * n,
        "customer_country": ["PK"] * n,
        "is_fraud": [0] * n,
    })


def test_pii_columns_are_removed():
    df = remove_pii(make_df())
    for col in PII_COLUMNS:
        assert col not in df.columns


def test_train_is_always_before_test():
    train_df, test_df = time_based_split(make_df(100))
    assert train_df["order_time"].max() <= test_df["order_time"].min()


def test_guard_raises_on_future_timestamp():
    df = make_df(10)
    cutoff = df["order_time"].max()
    df.loc[0, "order_time"] = cutoff + timedelta(days=5)

    guard = FutureTimestampGuard(
        timestamp_cols=["order_time", "account_created_at"],
        reference_time=cutoff,
    )
    with pytest.raises(ValueError):
        guard.transform(df)


def test_guard_passes_clean_data():
    df = make_df(10)
    guard = FutureTimestampGuard(
        timestamp_cols=["order_time", "account_created_at"],
        reference_time=df["order_time"].max(),
    )
    result = guard.transform(df)
    assert len(result) == len(df)


def test_four_engineered_features_exist():
    df = make_df(30)
    engineer = FraudFeatureEngineer().fit(df)
    out = engineer.transform(df)
    for col in ["account_age_days", "order_hour", "high_amount_flag", "customer_txn_velocity_24h"]:
        assert col in out.columns


def test_high_amount_threshold_comes_only_from_train():
    train_df = make_df(20)
    train_df["amount"] = list(range(1, 21))

    engineer = FraudFeatureEngineer().fit(train_df)
    threshold_before = engineer.threshold_

    test_df = make_df(5)
    test_df["amount"] = [1000, 2000, 3000, 4000, 5000]
    engineer.transform(test_df)

    assert engineer.threshold_ == threshold_before


def test_velocity_counts_only_past_orders():
    df = make_df(10)
    df["customer_id"] = 1000
    out = FraudFeatureEngineer().fit(df).transform(df)
    out = out.sort_values("order_time")
    assert out["customer_txn_velocity_24h"].tolist() == list(range(10))