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


clean_df = pd.DataFrame({
    "order_time": ["2026-06-01", "2026-06-10", "2026-06-20"],
    "amount": [500, 300, 450],
})

bad_df = pd.DataFrame({
    "order_time": ["2026-06-01", "2026-12-25", "2026-06-20"],
    "amount": [500, 300, 450],
})

guard = FutureTimestampGuard(
    timestamp_cols=["order_time"],
    reference_time="2026-06-30",
)

print("CLEAN DATA:")
print(guard.fit(clean_df).transform(clean_df))

print("BAD DATA:")
try:
    guard.fit(bad_df).transform(bad_df)
except ValueError as e:
    print("Pakda gaya ->", e)