import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train_df = pd.DataFrame({
    "payment_method": ["credit_card", "wallet", "cash_on_delivery", "credit_card"],
    "merchant_category": ["delivery", "delivery", "gift_card", "subscription"],
})

test_df = pd.DataFrame({
    "payment_method": ["wallet", "debit_card"],
    "merchant_category": ["gift_card", "delivery"],
})

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train_df)

train_encoded = encoder.transform(train_df)
test_encoded = encoder.transform(test_df)

columns = encoder.get_feature_names_out()

print("TRAIN")
print(pd.DataFrame(train_encoded, columns=columns))
print("TEST")
print(pd.DataFrame(test_encoded, columns=columns))