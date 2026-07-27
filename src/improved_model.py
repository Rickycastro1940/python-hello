"""Corrective action from the evaluation: linear-trend + RF-residual hybrid.

The evaluation (`data/eval/evaluation_report.md`) diagnosed the plain Random
Forest as overfitting **and** biased against the upward trend (a tree model
cannot extrapolate beyond the training target range, so it under-forecasts
future months — notably the December peaks). The recommended fix was to make the
target extrapolable.

This module implements that fix as a `TrendResidualRegressor`:
1. Fit a **linear trend** on time (extrapolates the growth into 2024-2025).
2. Train the Random Forest on the **residual** (trend removed), so it only has
   to learn seasonality / covers-driven variation — a stationary target.
3. Predict = linear-trend(time) + RF(residual features).

It then validates the fix against the baseline with the same time-aware CV and
on the real 2024-2025 test set. Run: `uv run python src/improved_model.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import app, evaluate_model  # noqa: E402

FIGURE_PATH = PROJECT_ROOT / "data" / "eval" / "improved_forecast.png"
RANDOM_STATE = 42


class TrendResidualRegressor(BaseEstimator, RegressorMixin):
    """Linear trend on time + Random Forest on the detrended residual.

    Expects `X` to contain the standard `app.FEATURES` columns (which include
    `year` and `month_num`, used to build a monotonic time index).
    """

    def __init__(self, n_estimators: int = 100, random_state: int = RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _time_index(self, X):
        # Absolute monthly ordinal so fit/predict share the same origin.
        return (
            X["year"].to_numpy() * 12 + (X["month_num"].to_numpy() - 1)
        ).reshape(-1, 1)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.trend_ = LinearRegression().fit(self._time_index(X), y)
        residual = y - self.trend_.predict(self._time_index(X))
        self.residual_model_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                    ),
                ),
            ]
        ).fit(X[app.FEATURES], residual)
        return self

    def predict(self, X):
        return self.trend_.predict(self._time_index(X)) + self.residual_model_.predict(
            X[app.FEATURES]
        )


def _cv_val_rmse(estimator, X, y) -> dict:
    raw = cross_validate(
        estimator,
        X,
        y,
        cv=evaluate_model.make_time_series_cv(5),
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
    )
    return {
        "train_rmse": float(-raw["train_score"].mean()),
        "val_rmse_mean": float(-raw["test_score"].mean()),
        "val_rmse_std": float(raw["test_score"].std()),
    }


def _test_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def main() -> None:
    df = app.load_and_prepare_data().sort_values("month").reset_index(drop=True)
    train_df, test_df = app.split_data(df)
    X_train, y_train = train_df[app.FEATURES], train_df[app.TARGET]
    X_test, y_test = test_df[app.FEATURES], test_df[app.TARGET]

    baseline = evaluate_model.build_estimator()
    hybrid = TrendResidualRegressor()

    # 1) Time-aware CV comparison
    base_cv = _cv_val_rmse(baseline, X_train, y_train)
    hybrid_cv = _cv_val_rmse(hybrid, X_train, y_train)
    print("Time-aware CV (validation RMSE, mean ± std):")
    print(
        f"  baseline RF : {base_cv['val_rmse_mean']:,.0f} ± "
        f"{base_cv['val_rmse_std']:,.0f}"
    )
    print(
        f"  hybrid      : {hybrid_cv['val_rmse_mean']:,.0f} ± "
        f"{hybrid_cv['val_rmse_std']:,.0f}"
    )

    # 2) Held-out test set (2024-2025)
    base_pred = baseline.fit(X_train, y_train).predict(X_test)
    hybrid_pred = hybrid.fit(X_train, y_train).predict(X_test)
    base_test = _test_metrics(y_test, base_pred)
    hybrid_test = _test_metrics(y_test, hybrid_pred)
    print("\nTest set 2024-2025:")
    for name, m in (("baseline RF", base_test), ("hybrid", hybrid_test)):
        print(
            f"  {name:12s} RMSE {m['rmse']:,.0f} | MAE {m['mae']:,.0f} | "
            f"MAPE {m['mape']:.2f}%"
        )

    # December-specific error (the peaks the baseline missed)
    dec = test_df["month_num"].to_numpy() == 12
    if dec.any():
        y_dec = np.asarray(y_test)[dec]
        base_dec = np.mean(np.abs((y_dec - base_pred[dec]) / y_dec)) * 100
        hybrid_dec = np.mean(np.abs((y_dec - hybrid_pred[dec]) / y_dec)) * 100
        print(
            f"  December MAPE — baseline {base_dec:.2f}% vs hybrid {hybrid_dec:.2f}%"
        )

    # 3) Plot actual vs baseline vs hybrid on the test years
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["month"], y_test, "o-", color="blue", label="Actual")
    plt.plot(
        test_df["month"], base_pred, "--", color="red", label="Baseline RF"
    )
    plt.plot(
        test_df["month"], hybrid_pred, "--", color="green", label="Hybrid (trend+RF)"
    )
    plt.title("Corrective action: trend+RF hybrid vs baseline (test 2024-2025)")
    plt.xlabel("Date")
    plt.ylabel("Revenue (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH)
    plt.close()
    print(f"\nSaved comparison plot to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
