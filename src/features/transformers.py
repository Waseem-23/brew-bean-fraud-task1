import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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