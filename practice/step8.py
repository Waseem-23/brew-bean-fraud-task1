import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

train_df = pd.DataFrame({
    "amount": [500, 300, 450, 1200],
    "account_age_days": [145.0, 0.2, 30.0, 400.0],
    "payment_method": ["credit_card", "wallet", "cash_on_delivery", "credit_card"],
    "merchant_category": ["delivery", "delivery", "gift_card", "subscription"],
})

test_df = pd.DataFrame({
    "amount": [600, 5000],
    "account_age_days": [2.0, 90.0],
    "payment_method": ["wallet", "debit_card"],
    "merchant_category": ["gift_card", "delivery"],
})

numeric_features = ["amount", "account_age_days"]
categorical_features = ["payment_method", "merchant_category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

preprocessor.fit(train_df)

train_out = preprocessor.transform(train_df)
test_out = preprocessor.transform(test_df)

columns = preprocessor.get_feature_names_out()

print("TRAIN")
print(pd.DataFrame(train_out, columns=columns).round(2))
print("TEST")
print(pd.DataFrame(test_out, columns=columns).round(2))