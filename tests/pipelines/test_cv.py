"""Validate the temporal cross-validation preserves chronological order.

No fold may mix/shuffle data: within every fold the training indices must come
strictly before the validation indices, and validation windows must move
forward in time across folds (no index from a later fold appears before one
from an earlier fold).
"""

import numpy as np
import pandas as pd

from src import evaluate_model


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
