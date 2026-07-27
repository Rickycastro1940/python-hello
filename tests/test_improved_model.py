"""Tests for the corrective-action model (`src/improved_model.py`).

Locks in the two properties that motivated the fix: the trend+RF hybrid
(1) can extrapolate beyond the training target range, and (2) beats the plain
Random Forest baseline on the held-out test years.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import root_mean_squared_error

from src import app, evaluate_model
from src.improved_model import TrendResidualRegressor

DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "brasaland_sales.csv"
)


def _train_test():
    df = app.load_and_prepare_data().sort_values("month").reset_index(drop=True)
    train_df, test_df = app.split_data(df)
    return (
        train_df[app.FEATURES],
        train_df[app.TARGET],
        test_df[app.FEATURES],
        test_df[app.TARGET],
    )


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="provided dataset not present")
def test_hybrid_beats_baseline_on_test_rmse():
    X_train, y_train, X_test, y_test = _train_test()

    baseline_pred = (
        evaluate_model.build_estimator().fit(X_train, y_train).predict(X_test)
    )
    hybrid_pred = TrendResidualRegressor().fit(X_train, y_train).predict(X_test)

    baseline_rmse = root_mean_squared_error(y_test, baseline_pred)
    hybrid_rmse = root_mean_squared_error(y_test, hybrid_pred)

    assert hybrid_rmse < baseline_rmse


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="provided dataset not present")
def test_hybrid_extrapolates_but_baseline_cannot():
    X_train, y_train, X_test, y_test = _train_test()
    train_max = float(np.asarray(y_train).max())

    baseline_pred = (
        evaluate_model.build_estimator().fit(X_train, y_train).predict(X_test)
    )
    hybrid_pred = TrendResidualRegressor().fit(X_train, y_train).predict(X_test)

    # A tree model can never predict above the largest training target...
    assert baseline_pred.max() <= train_max + 1e-6
    # ...but the trend component lets the hybrid extrapolate past it.
    assert hybrid_pred.max() > train_max
