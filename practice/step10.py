import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FutureTimestampGuard(BaseEstimator, TransformerMixin):
    def __init__(self, timestamp_cols, reference_time):
        self.timestamp_cols = timestamp_cols
        self.reference_time = reference_time

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ref = pd.to_datetime(self.reference_time)
        bad_rows = pd.Series(False, index=X.index)
        for col in self.timestamp_cols:
            bad_rows = bad_rows | (pd.to_datetime(X[col]) > ref)
        if bad_rows.any():
            raise ValueError(
                f"Leakage guard: {bad_rows.sum()} row(s) have a timestamp after {ref}"
            )
        return X

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.threshold_ = X["amount"].quantile(0.75)
        return self

    def transform(self, X):
        X = X.copy()
        order_time = pd.to_datetime(X["order_time"])
        account_created = pd.to_datetime(X["account_created_at"])

        X["account_age_days"] = (order_time - account_created).dt.total_seconds() / 86400
        X["order_hour"] = order_time.dt.hour
        X["high_amount_flag"] = (X["amount"] > self.threshold_).astype(int)

        velocity = []
        for idx in X.index:
            cust = X.loc[idx, "customer_id"]
            t = order_time.loc[idx]
            same_customer = X["customer_id"] == cust
            before_now = order_time < t
            in_window = order_time >= t - pd.Timedelta(hours=24)
            velocity.append((same_customer & before_now & in_window).sum())
        X["customer_txn_velocity_24h"] = velocity
        return X

numeric_features = [
    "amount", "account_age_days", "order_hour",
    "high_amount_flag", "customer_txn_velocity_24h",
]
categorical_features = ["payment_method", "merchant_category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("leakage_guard", FutureTimestampGuard(
            timestamp_cols=["order_time", "account_created_at"],
            reference_time="2026-06-30",
        )),
        ("feature_engineering", FraudFeatureEngineer()),
        ("preprocess", preprocessor),
    ]
)

df = pd.DataFrame({
    "customer_id": [1, 1, 2, 1, 3, 2, 3, 1, 2, 3],
    "order_time": [
        "2026-06-01 10:00:00", "2026-06-01 10:30:00", "2026-06-02 03:00:00",
        "2026-06-03 14:00:00", "2026-06-05 23:30:00", "2026-06-10 09:00:00",
        "2026-06-15 02:15:00", "2026-06-20 12:00:00", "2026-06-25 04:00:00",
        "2026-06-28 16:45:00",
    ],
    "account_created_at": [
        "2026-01-10 09:00:00", "2026-01-10 09:00:00", "2026-06-02 01:00:00",
        "2026-01-10 09:00:00", "2026-05-30 10:00:00", "2026-06-02 01:00:00",
        "2026-05-30 10:00:00", "2026-01-10 09:00:00", "2026-06-02 01:00:00",
        "2026-05-30 10:00:00",
    ],
    "amount": [500, 300, 1200, 450, 200, 350, 5000, 400, 900, 250],
    "payment_method": [
        "credit_card", "wallet", "credit_card", "wallet", "cash_on_delivery",
        "credit_card", "wallet", "debit_card", "credit_card", "wallet",
    ],
    "merchant_category": [
        "delivery", "delivery", "gift_card", "delivery", "subscription",
        "delivery", "gift_card", "delivery", "gift_card", "subscription",
    ],
    "is_fraud": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
})

df["order_time"] = pd.to_datetime(df["order_time"])
df = df.sort_values("order_time").reset_index(drop=True)

split_point = int(len(df) * 0.8)
train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

X_train = train_df.drop(columns=["is_fraud"])
X_test = test_df.drop(columns=["is_fraud"])

pipeline.fit(X_train)

train_out = pipeline.transform(X_train)
test_out = pipeline.transform(X_test)

columns = pipeline.named_steps["preprocess"].get_feature_names_out()

print("Train shape:", train_out.shape)
print("Test shape: ", test_out.shape)
print()
print("Columns:")
for c in columns:
    print(" -", c)