"""Unit tests for the Brasaland sales forecasting pipeline (``src/app.py``)."""

import numpy as np
import pandas as pd
import pytest

from src import app


def test_calculate_psi_identical_distribution_is_near_zero():
    """PSI of a distribution against itself should be ~0 (no drift)."""
    rng = np.random.default_rng(0)
    sample = rng.normal(100, 15, size=1000)
    assert app.calculate_psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_calculate_psi_detects_distribution_change():
    """A distribution with a clearly different shape yields a much higher PSI.

    This implementation bins each input over its own min/max range, so it
    responds to changes in distribution *shape*; a re-shaped (bimodal) actual
    distribution should score far above the same-distribution baseline.
    """
    rng = np.random.default_rng(1)
    expected = rng.normal(100, 10, size=5000)
    same = rng.normal(100, 10, size=5000)
    bimodal = np.concatenate(
        [rng.normal(60, 5, 2500), rng.normal(140, 5, 2500)]
    )

    psi_same = app.calculate_psi(expected, same)
    psi_changed = app.calculate_psi(expected, bimodal)

    assert psi_changed > psi_same
    assert psi_changed > 1.0


def test_normalized_gini_perfect_prediction_is_one():
    """Perfectly ranked predictions yield a normalized Gini of 1.0."""
    actual = np.array([10.0, 50.0, 20.0, 80.0, 40.0])
    assert app.normalized_gini(actual, actual) == pytest.approx(1.0)


def test_normalized_gini_reversed_prediction_is_negative():
    """Perfectly inverted rankings give a negative normalized Gini."""
    actual = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    reversed_pred = actual[::-1]
    assert app.normalized_gini(actual, reversed_pred) < 0


def test_gini_assertion_on_length_mismatch():
    """gini() must reject mismatched-length inputs."""
    with pytest.raises(AssertionError):
        app.gini([1, 2, 3], [1, 2])


def _synthetic_frame() -> pd.DataFrame:
    months = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
    rng = np.random.default_rng(7)
    covers = rng.integers(2500, 6000, size=len(months))
    ticket = rng.uniform(50, 70, size=len(months)).round(2)
    return pd.DataFrame(
        {
            "month": months.strftime("%Y-%m-%d"),
            "covers_served": covers,
            "avg_ticket_usd": ticket,
            "revenue_usd": (covers * ticket).round(2),
        }
    )


def test_split_is_strict_eight_two_years():
    """Training window is 2016-2023 (8y); test window is 2024-2025 (2y)."""
    prepared = _synthetic_frame()
    prepared["month"] = pd.to_datetime(prepared["month"])
    prepared["year"] = prepared["month"].dt.year
    prepared["month_num"] = prepared["month"].dt.month

    train_df, test_df = app.split_data(prepared)

    assert train_df["year"].min() == 2016
    assert train_df["year"].max() == 2023
    assert test_df["year"].min() == 2024
    assert test_df["year"].max() == 2025
    assert len(train_df) == 8 * 12
    assert len(test_df) == 2 * 12
    # No leakage between the two windows.
    assert set(train_df["year"]).isdisjoint(set(test_df["year"]))


def test_end_to_end_pipeline_produces_metrics():
    """With the provided dataset present, train and yield finite metrics.

    Skips (rather than fails) when the real dataset has not been supplied, since
    the pipeline never falls back to generated/simulated data.
    """
    try:
        df = app.load_and_prepare_data()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    assert {"year", "month_num"}.issubset(df.columns)

    train_df, test_df = app.split_data(df)
    X_train, y_train = train_df[app.FEATURES], train_df[app.TARGET]
    X_test, y_test = test_df[app.FEATURES], test_df[app.TARGET]

    model = app.train_model(X_train, y_train)
    results = app.evaluate(model, X_test, y_test, y_train)

    assert len(results["y_pred"]) == len(test_df)
    assert np.isfinite(results["mse"]) and results["mse"] > 0
    assert np.isfinite(results["psi"])
    assert np.isfinite(results["gini"])
    assert -1.0 <= results["k2_tau"] <= 1.0
