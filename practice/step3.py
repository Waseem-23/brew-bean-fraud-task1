import pandas as pd

data = {
    "amount": [500, 1200, 90, 300, 700, 150, 2000, 80, 450, 900],
    "order_time": [
        "2026-06-05", "2026-06-01", "2026-06-20", "2026-06-10", "2026-06-15",
        "2026-06-25", "2026-06-03", "2026-06-30", "2026-06-12", "2026-06-18",
    ],
    "is_fraud": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
}

df = pd.DataFrame(data)

df["order_time"] = pd.to_datetime(df["order_time"])
df = df.sort_values("order_time").reset_index(drop=True)

split_point = int(len(df) * 0.7)
train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

print("TRAIN")
print(train_df)
print("TEST")
print(test_df)

print(train_df["order_time"].max() <= test_df["order_time"].min())