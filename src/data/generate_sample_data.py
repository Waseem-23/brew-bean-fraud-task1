import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 2000

start = pd.Timestamp("2026-06-01")
end = pd.Timestamp("2026-09-20")
total_seconds = int((end - start).total_seconds())

order_time = start + pd.to_timedelta(rng.integers(0, total_seconds, size=n), unit="s")
order_time = order_time.sort_values()

account_age_days = rng.exponential(scale=120, size=n).clip(0, 900)
account_created_at = order_time - pd.to_timedelta(account_age_days, unit="D")

amount = (rng.gamma(2.0, 8.0, size=n) + 2).round(2)

customer_id = rng.integers(1000, 1300, size=n)

is_new_account = account_age_days < 2
is_night_order = order_time.hour < 5
is_high_amount = amount > np.percentile(amount, 95)

fraud_probability = (
    0.02
    + 0.15 * is_new_account
    + 0.06 * is_night_order
    + 0.10 * is_high_amount
)
is_fraud = rng.binomial(1, fraud_probability)

df = pd.DataFrame({
    "transaction_id": [f"TXN{100000 + i}" for i in range(n)],
    "customer_id": customer_id,
    "customer_name": [f"Customer {c}" for c in customer_id],
    "email": [f"customer{c}@example.com" for c in customer_id],
    "ip_address": [f"192.168.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(n)],
    "order_time": order_time,
    "account_created_at": account_created_at,
    "amount": amount,
    "payment_method": rng.choice(["credit_card", "debit_card", "wallet", "cash_on_delivery"], size=n),
    "merchant_category": rng.choice(["dine_in_topup", "delivery", "subscription", "gift_card"], size=n),
    "customer_country": rng.choice(["PK", "AE", "US", "GB", "SA"], size=n),
    "is_fraud": is_fraud,
})

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

output_path = PROJECT_ROOT / cfg["paths"]["raw_data"]
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Rows:", len(df))
print("Fraud rate:", round(df["is_fraud"].mean() * 100, 2), "%")
print("Saved to:", output_path)