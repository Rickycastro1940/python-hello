"""Brasaland sales forecasting (Finance RFI).

Trains a Random Forest regressor on 8 years of monthly sales (2016-2023) and
evaluates it on the 2 most recent years (2024-2025). Reports MSE, PSI, a
normalized Gini and a Kendall-tau (K2 proxy) score, and plots the forecast with
a per-tree prediction-variability band.

The helper functions (``calculate_psi``, ``gini``, etc.) are defined at module
level so they can be imported and unit-tested without running the full
pipeline; the end-to-end run is guarded by ``if __name__ == "__main__"``.
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the figure renders/saves in headless envs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402

# Resolve paths relative to this file so the script runs from any working dir.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The historical sales dataset is *provided*, never generated/simulated. It
# lives under one of the two canonical locations (checked in this order):
#   - reference repository: content/contexts/sales-forecasting/<company>/<company>_sales.csv
#   - monorepo:             data/raw/<company>_sales.csv
COMPANY = "brasaland"
DATASET_CANDIDATES = (
    PROJECT_ROOT
    / "content"
    / "contexts"
    / "sales-forecasting"
    / COMPANY
    / f"{COMPANY}_sales.csv",
    PROJECT_ROOT / "data" / "raw" / f"{COMPANY}_sales.csv",
)
FIGURE_PATH = PROJECT_ROOT / "reports" / "sales_forecast_variability.png"


def resolve_dataset_path() -> Path:
    """Return the first existing provided-dataset path, or raise if missing.

    We never fall back to generated/simulated data — if the provided dataset is
    absent, the caller must supply the real file at one of the canonical paths.
    """
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n  - ".join(str(c) for c in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Provided historical sales dataset not found. Add the real file "
        "(do not generate or simulate it) at one of:\n  - " + searched
    )

FEATURES = ["year", "month_num", "covers_served", "avg_ticket_usd"]
TARGET = "revenue_usd"

TRAIN_END_YEAR = 2023  # strict 8-year training window: 2016-2023
TEST_START_YEAR = 2024  # 2 most recent years: 2024-2025


# ==========================================
# 1. DATA PREPARATION & SPLIT
# ==========================================
def load_and_prepare_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset and engineer the time features used for regression."""
    if path is None:
        path = resolve_dataset_path()
    df = pd.read_csv(path)

    # Handle empty values (if any).
    df = df.dropna()

    # Convert 'month' to datetime and extract time features.
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month

    return df


def split_data(df: pd.DataFrame):
    """Split strictly into 8 training years (<=2023) and 2 test years (>=2024)."""
    train_df = df[df["year"] <= TRAIN_END_YEAR]
    test_df = df[df["year"] >= TEST_START_YEAR]
    return train_df, test_df


# ==========================================
# 2. MODEL TRAINING & JUSTIFICATION
# ==========================================
"""
JUSTIFICATION FOR RANDOM FOREST:
We chose Random Forest over XGBoost for this Finance RFI because of its high
explainability. While XGBoost trains sequentially and might squeeze out
slightly better accuracy, it acts more like a black box. Random Forest trains
multiple independent decision trees on different subsets of the data and
averages them. This makes it much easier to explain the prediction variability
and feature importance to business stakeholders.
"""


def train_model(X_train, y_train) -> RandomForestRegressor:
    """Fit the Random Forest regressor on the training window."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# ==========================================
# 3. EVALUATION METRICS
# ==========================================
def calculate_psi(expected, actual, buckets: int = 10) -> float:
    """Population Stability Index between an expected and actual distribution."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    expected_pct = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=buckets)[0] / len(actual)

    # Avoid division by zero / log(0).
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def gini(actual, pred) -> float:
    """Raw Gini coefficient of predictions vs actuals."""
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    assert len(actual) == len(pred)

    all_data = np.asarray(
        np.c_[actual, pred, np.arange(len(actual))], dtype=float
    )
    all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
    total_losses = all_data[:, 0].sum()
    gini_sum = all_data[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.0
    return gini_sum / len(actual)


def normalized_gini(actual, pred) -> float:
    """Gini normalized by the best-possible (perfect-ranking) Gini."""
    return gini(actual, pred) / gini(actual, actual)


def evaluate(model, X_test, y_test, y_train) -> dict:
    """Compute the RFI metric suite for the fitted model on the test set."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    psi_score = calculate_psi(y_train, y_pred)
    gini_score = normalized_gini(np.asarray(y_test), y_pred)
    tau, k2_p_value = stats.kendalltau(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "mse": mse,
        "psi": psi_score,
        "gini": gini_score,
        "k2_tau": tau,
        "k2_p_value": k2_p_value,
    }


# ==========================================
# 4. VISUALIZATION (PREDICTIONS + VARIABILITY)
# ==========================================
def plot_forecast(model, test_df, X_test, y_test, y_pred, path: Path = FIGURE_PATH):
    """Plot actual vs predicted revenue with a 90% per-tree variability band."""
    # Extract predictions from all trees to build the variability band.
    tree_preds = np.array(
        [tree.predict(X_test.values) for tree in model.estimators_]
    )
    lower_bound = np.percentile(tree_preds, 5, axis=0)
    upper_bound = np.percentile(tree_preds, 95, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(
        test_df["month"],
        y_test,
        label="Actual Revenue (2024-2025)",
        color="blue",
        marker="o",
    )
    plt.plot(
        test_df["month"],
        y_pred,
        label="Predicted Revenue",
        color="red",
        linestyle="--",
    )
    plt.fill_between(
        test_df["month"],
        lower_bound,
        upper_bound,
        color="red",
        alpha=0.2,
        label="90% Prediction Variability",
    )

    plt.title("Brasaland Sales Forecasting (Test Set: 2024-2025)")
    plt.xlabel("Date")
    plt.ylabel("Revenue (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    return path


def main() -> None:
    df = load_and_prepare_data()
    train_df, test_df = split_data(df)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print(
        f"Train: {len(train_df)} months ({train_df['year'].min()}-"
        f"{train_df['year'].max()})  |  "
        f"Test: {len(test_df)} months ({test_df['year'].min()}-"
        f"{test_df['year'].max()})"
    )

    model = train_model(X_train, y_train)
    results = evaluate(model, X_test, y_test, y_train)
    y_pred = results["y_pred"]

    print("--- Model Evaluation Metrics ---")
    print(f"MSE: {results['mse']:,.2f}")
    print(f"PSI: {results['psi']:.4f}")
    print(f"Gini: {results['gini']:.4f}")
    print(f"K2 Score (Kendall Tau proxy): {results['k2_tau']:.4f}")

    figure_path = plot_forecast(model, test_df, X_test, y_test, y_pred)
    print(f"Saved forecast plot to {figure_path}")


if __name__ == "__main__":
    main()
