from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    PII_COLUMNS, add_customer_velocity, build_pipeline, clean_data, remove_pii,
)
from src.features.transformers import (
    FraudFeatureEngineer, FutureTimestampGuard, LeakageFlagger,
)
from src.make_splits import create_splits


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
        "is_fraud": [1 if i % 10 == 0 else 0 for i in range(n)],
    })


def test_pii_columns_are_removed():
    df = remove_pii(make_df())
    for col in PII_COLUMNS:
        assert col not in df.columns


def test_clean_data_drops_account_created_after_order():
    df = make_df(10)
    df.loc[3, "account_created_at"] = df.loc[3, "order_time"] + timedelta(days=1)
    assert len(clean_data(df)) == 9


def test_guard_raises_on_future_timestamp():
    df = make_df(10)
    cutoff = df["order_time"].max()
    df.loc[0, "order_time"] = cutoff + timedelta(days=5)
    guard = FutureTimestampGuard(["order_time", "account_created_at"], cutoff)
    with pytest.raises(ValueError):
        guard.transform(df)


def test_guard_passes_clean_data():
    df = make_df(10)
    guard = FutureTimestampGuard(["order_time", "account_created_at"], df["order_time"].max())
    assert len(guard.transform(df)) == len(df)


def test_engineered_features_exist():
    df = make_df(30)
    out = FraudFeatureEngineer().fit(df).transform(df)
    for col in ["account_age_days", "order_hour", "high_amount_flag"]:
        assert col in out.columns


def test_high_amount_threshold_comes_only_from_train():
    train_df = make_df(20)
    train_df["amount"] = list(range(1, 21))
    engineer = FraudFeatureEngineer().fit(train_df)
    before = engineer.threshold_

    test_df = make_df(5)
    test_df["amount"] = [1000, 2000, 3000, 4000, 5000]
    engineer.transform(test_df)
    assert engineer.threshold_ == before


def test_velocity_counts_only_past_orders():
    df = make_df(10)
    df["customer_id"] = 1000
    out = add_customer_velocity(df)
    assert out["customer_txn_velocity_24h"].tolist() == list(range(10))


def test_splits_are_stratified_disjoint_and_reproducible(tmp_path):
    y = pd.Series([1 if i % 10 == 0 else 0 for i in range(200)])
    a = create_splits(y, 0.7, 0.15, 0.15, 42, tmp_path)
    b = create_splits(y, 0.7, 0.15, 0.15, 42, tmp_path)

    for x, z in zip(a, b):
        assert np.array_equal(x, z)                      # reproducible
    all_idx = np.concatenate(a)
    assert len(all_idx) == len(set(all_idx)) == 200      # overlap nahi, sab rows covered
    for idx in a:
        assert y.iloc[idx].mean() == pytest.approx(0.10, abs=0.02)  # stratified
    assert (tmp_path / "train_idx.csv").exists()


def test_leakage_flagger_flags_high_correlation():
    y = pd.Series([0, 1] * 50)
    X = pd.DataFrame({
        "leaky": y * 1.0,
        "noise": np.random.default_rng(0).normal(size=100),
    })
    with pytest.warns(UserWarning):
        flagger = LeakageFlagger(threshold=0.8).fit(X, y)
    assert flagger.flagged_cols_ == ["leaky"]


def test_pipeline_output_has_no_leaky_or_future_features():
    df = clean_data(make_df(200))
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])

    pipe = build_pipeline(reference_time=df["order_time"].max(), threshold=0.8)
    pipe.fit(X, y)
    out = pipe.transform(X)

    assert "is_fraud" not in out.columns
    assert pipe.named_steps["leakage_flagger"].flagged_cols_ == []
    assert out.corrwith(y).abs().dropna().max() <= 0.8
    assert not any(pd.api.types.is_datetime64_any_dtype(t) for t in out.dtypes)