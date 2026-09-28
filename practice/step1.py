import pandas as pd

data = {
    "customer_name": ["Ali", "Sara", "Bilal"],
    "email": ["ali@mail.com", "sara@mail.com", "bilal@mail.com"],
    "amount": [500, 1200, 90],
    "is_fraud": [0, 0, 1],
}

df = pd.DataFrame(data)
print(df)