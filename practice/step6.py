import pandas as pd

data = {
    "customer_id": [1, 1, 1, 2, 1, 2],
    "order_time": [
        "2026-06-01 10:00:00",
        "2026-06-01 10:30:00",
        "2026-06-01 11:00:00",
        "2026-06-01 11:15:00",
        "2026-06-03 09:00:00",
        "2026-06-03 09:30:00",
    ],
}

df = pd.DataFrame(data)
df["order_time"] = pd.to_datetime(df["order_time"])
df = df.sort_values("order_time").reset_index(drop=True)

velocity = []

for i in range(len(df)):
    current_customer = df.loc[i, "customer_id"]
    current_time = df.loc[i, "order_time"]
    window_start = current_time - pd.Timedelta(hours=24)

    past_orders = df.iloc[:i]

    same_customer = past_orders["customer_id"] == current_customer
    in_window = past_orders["order_time"] >= window_start

    count = (same_customer & in_window).sum()
    velocity.append(count)

df["customer_txn_velocity_24h"] = velocity

print(df)