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


def test_scale_features_uses_train_stats_only():
    """Scaler is fit on train only; train is standardized, test uses train stats."""
    X_train = pd.DataFrame({"a": [0.0, 10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0, 4.0]})
    X_test = pd.DataFrame({"a": [40.0, 50.0], "b": [5.0, 6.0]})

    X_train_s, X_test_s, _ = app.scale_features(X_train, X_test)

    # Column names and index are preserved.
    assert list(X_train_s.columns) == ["a", "b"]

    # Training features are standardized (mean ~0, std ~1 with ddof=0).
    assert X_train_s["a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert X_train_s["a"].std(ddof=0) == pytest.approx(1.0)

    # Test rows are transformed with TRAIN statistics (no test leakage).
    expected_a = (X_test["a"] - X_train["a"].mean()) / X_train["a"].std(ddof=0)
    assert np.allclose(X_test_s["a"].to_numpy(), expected_a.to_numpy())


def test_train_model_and_feature_importances():
    """train_model returns a fitted sklearn RF and importances cover all features."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {
            "year": rng.integers(2016, 2024, size=60),
            "month_num": rng.integers(1, 13, size=60),
            "covers_served": rng.integers(30000, 80000, size=60),
            "avg_ticket_usd": rng.uniform(11, 13, size=60),
        }
    )
    y = X["covers_served"] * X["avg_ticket_usd"]

    model = app.train_model(X, y)
    assert isinstance(model, RandomForestRegressor)

    importances = app.feature_importances(model, X.columns)
    assert set(importances) == set(X.columns)
    assert all(v >= 0 for v in importances.values())
    assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)


def test_null_and_empty_values_are_dropped(tmp_path):
    """Rows with null/empty/whitespace in required columns are removed."""
    csv = tmp_path / "sales.csv"
    csv.write_text(
        "month,revenue_usd,covers_served,avg_ticket_usd,market\n"
        "2016-01-01,100000,2000,50.0,consolidated\n"   # valid
        "2016-02-01,,2100,50.0,consolidated\n"          # empty revenue
        "2016-03-01,120000,,51.0,consolidated\n"        # missing covers
        "2016-04-01,130000,2200,   ,consolidated\n"     # whitespace ticket
        "2016-05-01,140000,2300,52.0,consolidated\n"    # valid
    )

    df = app.load_and_prepare_data(csv)

    assert len(df) == 2  # only the two fully-populated rows survive
    assert (
        df[["revenue_usd", "covers_served", "avg_ticket_usd"]].notna().all().all()
    )


def test_split_has_no_test_year_leakage(tmp_path):
    """The training window must never contain any test-year (>=2024) row."""
    prepared = _synthetic_frame()
    prepared["month"] = pd.to_datetime(prepared["month"])
    prepared["year"] = prepared["month"].dt.year
    prepared["month_num"] = prepared["month"].dt.month

    train_df, test_df = app.split_data(prepared)

    assert train_df["year"].max() < app.TEST_START_YEAR
    assert test_df["year"].min() > app.TRAIN_END_YEAR
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
    X_train, X_test, _ = app.scale_features(X_train, X_test)

    model = app.train_model(X_train, y_train)
    results = app.evaluate(model, X_test, y_test, y_train)

    assert len(results["y_pred"]) == len(test_df)
    # The four required metrics (+ RMSE/MAPE) are all reported on the test set.
    for key in ("mse", "rmse", "mape", "psi", "gini", "k2_tau"):
        assert key in results
        assert np.isfinite(results[key])
    assert results["mse"] > 0
    assert results["rmse"] == pytest.approx(np.sqrt(results["mse"]))
    assert results["mape"] >= 0
    assert -1.0 <= results["k2_tau"] <= 1.0
