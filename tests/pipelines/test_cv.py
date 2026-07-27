"""Validate the temporal cross-validation preserves chronological order.

No fold may mix/shuffle data: within every fold the training indices must come
strictly before the validation indices, and validation windows must move
forward in time across folds (no index from a later fold appears before one
from an earlier fold).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import app, evaluate_model

DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "brasaland_sales.csv"
)


def test_time_series_cv_preserves_chronological_order():
    n = 96  # the Brasaland training window (8 years of months)
    X = pd.DataFrame({"t": np.arange(n)})
    y = pd.Series(np.arange(n, dtype=float))

    cv = evaluate_model.make_time_series_cv(n_splits=5)
    folds = list(cv.split(X, y))

    # At least 5 folds, as required by the brief.
    assert len(folds) == 5

    prev_val_max = -1
    prev_train_max = -1
    for train_idx, val_idx in folds:
        train_idx = list(train_idx)
        val_idx = list(val_idx)

        # Indices are in ascending order within each fold (no shuffling).
        assert train_idx == sorted(train_idx)
        assert val_idx == sorted(val_idx)

        # Training is strictly before validation (no future leaks into the past).
        assert max(train_idx) < min(val_idx)

        # Validation windows advance in time across folds; no overlap backwards.
        assert min(val_idx) > prev_val_max
        prev_val_max = max(val_idx)

        # Expanding window: each fold trains on at least as much history as the last.
        assert max(train_idx) >= prev_train_max
        prev_train_max = max(train_idx)


def test_time_series_cv_no_index_from_later_fold_precedes_earlier_fold():
    """Concatenated validation indices are strictly increasing across folds."""
    n = 96
    X = pd.DataFrame({"t": np.arange(n)})

    cv = evaluate_model.make_time_series_cv(n_splits=5)
    val_indices = [idx for _, val in cv.split(X) for idx in val]

    assert val_indices == sorted(val_indices)
    assert len(val_indices) == len(set(val_indices))  # no repeats across folds


@pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason=f"Provided dataset not found at {DATASET_PATH}.",
)
def test_time_series_cv_preserves_month_order_on_brasaland_data():
    """On the real Brasaland training data, each fold's train `month`s precede its validation `month`s."""
    df = app.load_and_prepare_data().sort_values("month").reset_index(drop=True)
    train_df, _ = app.split_data(df)
    months = train_df["month"].reset_index(drop=True)

    cv = evaluate_model.make_time_series_cv(n_splits=5)
    prev_val_max_month = None
    for train_idx, val_idx in cv.split(train_df):
        train_months = months.iloc[train_idx]
        val_months = months.iloc[val_idx]
        # Every training month is strictly earlier than every validation month.
        assert train_months.max() < val_months.min()
        # Validation windows advance in calendar time across folds.
        if prev_val_max_month is not None:
            assert val_months.min() > prev_val_max_month
        prev_val_max_month = val_months.max()
