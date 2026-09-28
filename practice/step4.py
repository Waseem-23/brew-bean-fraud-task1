import pandas as pd

data = {
    "amount": [500, 1200, 90, 300],
    "order_time": [
        "2026-06-05 02:30:00",
        "2026-06-01 14:15:00",
        "2026-06-20 03:45:00",
        "2026-06-10 19:00:00",
    ],
    "account_created_at": [
        "2026-01-10 09:00:00",
        "2026-06-01 10:00:00",
        "2026-06-19 22:00:00",
        "2025-11-02 12:00:00",
    ],
    "is_fraud": [0, 0, 1, 0],
}

df = pd.DataFrame(data)

df["order_time"] = pd.to_datetime(df["order_time"])
df["account_created_at"] = pd.to_datetime(df["account_created_at"])

df["account_age_days"] = (
    df["order_time"] - df["account_created_at"]
).dt.total_seconds() / 86400

df["order_hour"] = df["order_time"].dt.hour

print(df[["order_time", "account_age_days", "order_hour", "is_fraud"]])