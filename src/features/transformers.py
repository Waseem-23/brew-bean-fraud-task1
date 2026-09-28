import warnings

import numpy as np
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
    """Threshold sirf train se seekhta hai (fit), transform mein sirf apply karta hai."""

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
        return X


class LeakageFlagger(BaseEstimator, TransformerMixin):
    """Woh columns flag karta hai jin ka |correlation| target se threshold se zyada ho.

    Sirf fit() mein (train data par) correlation nikalta hai.
    drop=True ho to flagged columns transform mein hata deta hai.
    """

    def __init__(self, threshold=0.8, drop=False):
        self.threshold = threshold
        self.drop = drop

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        corr = X.apply(lambda col: col.astype(float).corr(y))
        self.correlations_ = corr
        self.flagged_cols_ = list(corr.index[corr.abs() > self.threshold])
        if self.flagged_cols_:
            warnings.warn(
                f"Possible target leakage (|corr| > {self.threshold}): {self.flagged_cols_}",
                UserWarning,
            )
        return self

    def transform(self, X):
        if self.drop and self.flagged_cols_:
            return X.drop(columns=self.flagged_cols_)
        return X