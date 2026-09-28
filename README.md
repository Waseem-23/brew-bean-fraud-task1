# Brew & Bean Café: Fraud Detection (Task 1)

Data integrity and leakage-prevention pipeline for the Brew & Bean Café
transaction dataset. It loads raw transactions, cleans them, engineers
features, checks for target leakage, and writes reproducible train/val/test
parquet files.

## Project structure

```
brew_bean_fraud/
├── config/config.yaml          # all paths and split parameters
├── data/
│   ├── raw/                    # raw_transactions.csv (input)
│   ├── processed/              # train/val/test .parquet + fitted pipeline
│   └── splits/                 # saved split indices (committed)
├── docs/decision_log.md        # design decisions
├── src/
│   ├── data_pipeline.py        # main pipeline
│   ├── make_splits.py          # stratified split + saved indices
│   ├── features/transformers.py# guard, feature engineer, leakage flagger
│   └── data/generate_sample_data.py  # how the sample data was generated
├── tests/test_pipeline.py      # pytest unit tests
└── requirements.txt
```

## Dependencies

Python 3.12, pandas, numpy, scikit-learn, pyyaml, pyarrow, joblib, pytest.

```
pip install -r requirements.txt
```

## Configuration

Everything is set in `config/config.yaml`; no paths are hard-coded.

| Key | Meaning |
|---|---|
| `paths.raw_data` | input CSV |
| `paths.processed_dir` | where parquet files are written |
| `paths.splits_dir` | where split indices are saved |
| `paths.pipeline_artifact` | fitted pipeline (joblib) |
| `target_column` | label column (`is_fraud`) |
| `random_state` | seed for reproducible splits |
| `split.*` | train / val / test fractions (0.70 / 0.15 / 0.15) |
| `leakage.correlation_threshold` | flag features with abs(corr) above this (0.8) |

## Pipeline steps

1. **Load** the raw CSV (path from config), parsing timestamps.
2. **Clean**: drop duplicate `transaction_id`s, rows with missing values or
   `amount <= 0`, and rows where `account_created_at > order_time`.
3. **Remove PII**: `customer_name`, `email`, `ip_address`.
4. **Velocity feature**: number of the customer's orders in the previous 24 h
   (looks only at the past).
5. **Stratified split** 70/15/15; indices saved to `data/splits/`.
6. **Fit the sklearn Pipeline on the training split only**:
   - `FutureTimestampGuard`: raises if any timestamp is after the reference time
   - `FraudFeatureEngineer`: account age, order hour, high-amount flag
     (threshold learned from train only)
   - `ColumnTransformer`: scale numeric, one-hot encode categorical
   - `LeakageFlagger`: warns for features with abs(corr) > 0.8 to the target
7. **Transform** train / val / test and save as parquet in `data/processed/`.

## How to rerun

```
pip install -r requirements.txt
python -m src.data_pipeline
python -m pytest -v
```

Outputs: `data/processed/{train,val,test}.parquet`,
`data/processed/fraud_pipeline.joblib`, `data/splits/{train,val,test}_idx.csv`.

## Leakage checks

- Scaler, encoder, amount threshold and leakage flagger are fitted on train only.
- Future timestamps are rejected by `FutureTimestampGuard`.
- Accounts created after the order are dropped during cleaning.
- `LeakageFlagger` flags any feature with abs(corr) > 0.8 to the target.
- Tests verify all of the above; see `docs/decision_log.md` for limitations.