# Brew & Bean Cafe: Fraud Detection (Task 1)

A leakage-safe data pipeline with feature engineering for cafe order fraud
detection.

## What it does

1. Loads sample transaction data (synthetic, made by `generate_sample_data.py`).
2. Removes personal information: `customer_name`, `email`, `ip_address`.
3. Splits the data by time: the oldest 80% of orders are train, the newest
   20% are test. No random shuffling.
4. Builds a scikit-learn `Pipeline`:
   - `FutureTimestampGuard`: a custom transformer that raises an error if any
     timestamp is later than the dataset's reference time (leakage guard).
   - `FraudFeatureEngineer`: a custom transformer that adds 4 features:
     `account_age_days`, `order_hour`, `high_amount_flag`,
     `customer_txn_velocity_24h`.
   - `ColumnTransformer`: `StandardScaler` for numeric columns and
     `OneHotEncoder` for categorical columns.
5. Saves `train.csv`, `test.csv` and the fitted pipeline
   (`fraud_pipeline.joblib`) in `src/data/`.

The reasoning behind each choice is in
[docs/decision_log.md](docs/decision_log.md).

## Project structure

```
brew_bean_fraud/
├── README.md
├── requirements.txt
├── docs/
│   └── decision_log.md
├── src/
│   ├── data/
│   │   ├── generate_sample_data.py
│   │   └── etl.py
│   └── features/
│       └── transformers.py
├── tests/
│   └── test_leakage.py
└── practice/            (step-by-step learning scripts)
```

## Setup

```
pip install -r requirements.txt
```

## Run

Run these from the project root folder:

```
python src/data/generate_sample_data.py
python -m src.data.etl
```

This creates `raw_transactions.csv`, `train.csv`, `test.csv` and
`fraud_pipeline.joblib` inside `src/data/`.

## Test

```
python -m pytest tests -v
```

The 7 tests check that:
- PII columns are removed
- every train timestamp is before every test timestamp
- the leakage guard raises an error on future timestamps and passes clean data
- the 4 engineered features exist
- the high-amount threshold comes only from train data
- the velocity feature counts only past orders

## Note on the data

Real cafe data was not available, so the data is synthetic. It has the same
kind of columns we expect in production, and the code should work the same
once real data is placed in `src/data/raw_transactions.csv`.