import pandas as pd

data = {
    "order_time": [
        "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05",
        "2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09", "2026-06-10",
    ],
    "amount": [500, 300, 450, 1200, 200, 350, 600, 400, 5000, 250],
    "is_fraud": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
}

df = pd.DataFrame(data)
df["order_time"] = pd.to_datetime(df["order_time"])
df = df.sort_values("order_time").reset_index(drop=True)

split_point = int(len(df) * 0.8)
train_df = df.iloc[:split_point].copy()
test_df = df.iloc[split_point:].copy()

threshold = train_df["amount"].quantile(0.75)
print("Threshold (sirf train se):", threshold)

train_df["high_amount_flag"] = (train_df["amount"] > threshold).astype(int)
test_df["high_amount_flag"] = (test_df["amount"] > threshold).astype(int)

print("TRAIN")
print(train_df)
print("TEST")
print(test_df)