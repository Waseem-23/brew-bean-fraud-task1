from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(y, train_size, val_size, test_size, random_state, splits_dir):
    """Stratified train/val/test split. Indices splits_dir mein save hote hain."""
    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size 1.0 hona chahiye")

    y = np.asarray(y)
    idx = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        idx, train_size=train_size, stratify=y, random_state=random_state
    )
    val_share = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_share, stratify=y[temp_idx], random_state=random_state
    )

    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        pd.DataFrame({"row_index": np.sort(arr)}).to_csv(
            splits_dir / f"{name}_idx.csv", index=False
        )
    return train_idx, val_idx, test_idx