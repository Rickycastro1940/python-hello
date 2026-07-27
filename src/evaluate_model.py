"""Evaluate the Brasaland sales regressor (module: Evaluating a Regression Model).

Provides time-aware cross-validation (`TimeSeriesSplit`), a learning curve, and
MAE/RMSE for train vs validation, used to diagnose underfitting / overfitting /
good fit before promoting the model to staging. Reuses the training pipeline
from `src/app.py`.

Run with: `uv run python src/evaluate_model.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend so the figure saves without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import (  # noqa: E402
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Make the project root importable when run as a script (uv run python src/...).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import app  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
LEARNING_CURVE_PATH = EVAL_DIR / "learning_curve.png"
METRICS_PATH = EVAL_DIR / "metrics.json"

N_SPLITS = 5
RANDOM_STATE = 42


def build_estimator() -> Pipeline:
    """StandardScaler + RandomForest, mirroring the training pipeline.

    Wrapping the scaler in a Pipeline keeps cross-validation and the learning
    curve leakage-safe: the scaler is refit on each fold's training portion only.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def make_time_series_cv(n_splits: int = N_SPLITS) -> TimeSeriesSplit:
    """Expanding-window temporal CV: no shuffling, chronological order preserved."""
    return TimeSeriesSplit(n_splits=n_splits)


def verify_chronological_folds(n_samples: int, n_splits: int = N_SPLITS) -> dict:
    """Explicitly confirm no fold shuffles/mixes data (chronological order kept).

    For every fold: indices are ascending, all training indices come strictly
    before the validation indices, and validation windows advance in time.
    Raises AssertionError if any fold violates temporal order.
    """
    cv = make_time_series_cv(n_splits)
    indices = np.arange(n_samples).reshape(-1, 1)
    prev_val_max = -1
    for train_idx, val_idx in cv.split(indices):
        assert list(train_idx) == sorted(train_idx), "training fold is shuffled"
        assert list(val_idx) == sorted(val_idx), "validation fold is shuffled"
        assert max(train_idx) < min(val_idx), "train not strictly before validation"
        assert min(val_idx) > prev_val_max, "validation window went backwards"
        prev_val_max = max(val_idx)
    return {"n_splits": n_splits, "chronological_order_preserved": True}


def load_training_frame():
    """Return the chronologically-sorted training features/target (2016-2023)."""
    df = app.load_and_prepare_data().sort_values("month").reset_index(drop=True)
    train_df, _ = app.split_data(df)
    X_train = train_df[app.FEATURES].reset_index(drop=True)
    y_train = train_df[app.TARGET].reset_index(drop=True)
    return X_train, y_train


def time_series_cross_validate(X_train, y_train, n_splits: int = N_SPLITS) -> dict:
    """Run time-aware CV and return per-fold + mean/std RMSE and MAE.

    Uses `TimeSeriesSplit` (no shuffling) so each validation fold is strictly
    later in time than its training portion.
    """
    cv = make_time_series_cv(n_splits)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    raw = cross_validate(
        build_estimator(),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )

    summary = {"n_splits": n_splits}
    for metric in ("rmse", "mae"):
        val = -raw[f"test_{metric}"]  # sklearn returns negated errors
        train = -raw[f"train_{metric}"]
        summary[metric] = {
            "val_per_fold": [round(float(v), 2) for v in val],
            "val_mean": float(val.mean()),
            "val_std": float(val.std()),
            "train_mean": float(train.mean()),
            "train_std": float(train.std()),
        }
    return summary


def generate_learning_curve(
    X_train,
    y_train,
    path: Path = LEARNING_CURVE_PATH,
    val_months: int = 24,
    n_points: int = 6,
):
    """Chronological expanding-window learning curve; save the figure.

    We hold out the last `val_months` of the training window as a FIXED future
    validation block (chronologically after every training subset), then train
    on growing prefixes of history and record training RMSE (on the prefix) and
    validation RMSE (on the fixed future block). This preserves temporal order
    (training is always strictly before validation) and, unlike
    `learning_curve` + `TimeSeriesSplit`, explores realistic training sizes.
    """
    n = len(X_train)
    val_start = n - val_months
    X_val = X_train.iloc[val_start:]
    y_val = y_train.iloc[val_start:]

    sizes = np.unique(np.linspace(12, val_start, n_points).astype(int))
    train_rmse, val_rmse = [], []
    for k in sizes:
        X_k = X_train.iloc[:k]
        y_k = y_train.iloc[:k]
        estimator = build_estimator().fit(X_k, y_k)
        train_rmse.append(root_mean_squared_error(y_k, estimator.predict(X_k)))
        val_rmse.append(root_mean_squared_error(y_val, estimator.predict(X_val)))

    train_rmse = np.array(train_rmse)
    val_rmse = np.array(val_rmse)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_rmse, "o-", color="tab:blue", label="Training RMSE")
    plt.plot(sizes, val_rmse, "o-", color="tab:red", label="Validation RMSE")
    plt.title(
        f"Learning Curve — Brasaland RF (fixed {val_months}-month future "
        "validation block)"
    )
    plt.xlabel("Training set size (months of history)")
    plt.ylabel("RMSE (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return sizes, train_rmse, val_rmse


def resubstitution_metrics(X_train, y_train) -> dict:
    """Train-set MAE/RMSE (fit and predict on the full training window)."""
    estimator = build_estimator()
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_train)
    return {
        "mae": float(mean_absolute_error(y_train, pred)),
        "rmse": float(root_mean_squared_error(y_train, pred)),
    }


def diagnose(cv: dict, target_mean: float) -> dict:
    """Classify fit as well-fitted / underfitting / overfitting from CV errors."""
    train_rmse = cv["rmse"]["train_mean"]
    val_rmse = cv["rmse"]["val_mean"]
    val_pct = val_rmse / target_mean * 100
    train_pct = train_rmse / target_mean * 100
    gap_ratio = val_rmse / max(train_rmse, 1e-9)

    if val_pct >= 15 and gap_ratio < 2:
        label = "underfitting"
    elif gap_ratio >= 2 and train_pct < 5:
        label = "overfitting"
    else:
        label = "well fitted"

    return {
        "label": label,
        "train_rmse_pct_of_mean": round(train_pct, 2),
        "val_rmse_pct_of_mean": round(val_pct, 2),
        "val_over_train_ratio": round(gap_ratio, 2),
    }


def main() -> None:
    X_train, y_train = load_training_frame()
    target_mean = float(y_train.mean())
    print(f"Training window: {len(X_train)} months | mean revenue {target_mean:,.0f} USD")

    verify_chronological_folds(len(X_train))
    print(
        f"\nChronological check: all {N_SPLITS} TimeSeriesSplit folds preserve "
        "temporal order (train strictly before validation, no shuffling)."
    )

    cv = time_series_cross_validate(X_train, y_train)
    print(f"\nTime-aware CV ({cv['n_splits']} folds, TimeSeriesSplit, no shuffle):")
    for metric in ("rmse", "mae"):
        m = cv[metric]
        print(
            f"  {metric.upper()}  validation = {m['val_mean']:,.0f} ± "
            f"{m['val_std']:,.0f}  |  train = {m['train_mean']:,.0f} ± "
            f"{m['train_std']:,.0f}"
        )
        print(f"        val per fold: {m['val_per_fold']}")

    train_sizes, train_rmse, val_rmse = generate_learning_curve(X_train, y_train)
    print(f"\nLearning curve saved to {LEARNING_CURVE_PATH}")
    print(f"  train sizes: {[int(s) for s in train_sizes]}")
    print(f"  train RMSE : {[round(float(r)) for r in train_rmse]}")
    print(f"  val   RMSE : {[round(float(r)) for r in val_rmse]}")

    resub = resubstitution_metrics(X_train, y_train)
    print(
        f"\nTrain-set (resubstitution) — MAE {resub['mae']:,.0f} | "
        f"RMSE {resub['rmse']:,.0f}"
    )

    diagnosis = diagnose(cv, target_mean)
    print(f"\nDiagnosis: {diagnosis['label'].upper()} — {diagnosis}")

    metrics = {
        "target_mean_usd": target_mean,
        "cross_validation": cv,
        "resubstitution_train": resub,
        "learning_curve": {
            "train_sizes": [int(s) for s in train_sizes],
            "train_rmse": [float(r) for r in train_rmse],
            "val_rmse": [float(r) for r in val_rmse],
        },
        "diagnosis": diagnosis,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
