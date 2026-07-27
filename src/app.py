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
from sklearn.preprocessing import StandardScaler  # noqa: E402

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

# Columns exactly as described in CONTEXT-brasaland ("2. Data structure").
EXPECTED_COLUMNS = (
    "month",
    "revenue_usd",
    "covers_served",
    "avg_ticket_usd",
    "market",
)
# The model target is revenue_usd from the "consolidated" market row.
CONSOLIDATED_MARKET = "consolidated"

FEATURES = ["year", "month_num", "covers_served", "avg_ticket_usd"]
TARGET = "revenue_usd"

TRAIN_END_YEAR = 2023  # strict 8-year training window: 2016-2023
TEST_START_YEAR = 2024  # 2 most recent years: 2024-2025


# ==========================================
# 1. DATA PREPARATION & SPLIT
# ==========================================
def load_and_prepare_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset and engineer the time features used for regression.

    Validates that the file carries exactly the columns documented in
    CONTEXT-brasaland, then keeps the "consolidated" market rows (the CONTEXT
    designates them as the model's main rows / target source).
    """
    if path is None:
        path = resolve_dataset_path()
    df = pd.read_csv(path)

    # Verify the schema matches the CONTEXT-company description.
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing CONTEXT-specified columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # --- Handle null / empty values before training ---
    # Treat blank or whitespace-only cells as missing.
    df = df.replace(r"^\s*$", np.nan, regex=True)
    # Coerce numeric columns so non-numeric / empty entries become NaN.
    for col in ("revenue_usd", "covers_served", "avg_ticket_usd"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop any row missing a required column so the model trains on clean data.
    rows_before = len(df)
    df = df.dropna(subset=list(EXPECTED_COLUMNS)).reset_index(drop=True)
    dropped = rows_before - len(df)
    if dropped:
        print(f"Handled null/empty values: dropped {dropped} incomplete row(s).")

    # CONTEXT: use the "consolidated" market row as the main model row.
    df = df[df["market"] == CONSOLIDATED_MARKET].copy()

    # Convert 'month' to datetime and extract time features.
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month

    return df


def split_data(df: pd.DataFrame):
    """Split strictly into the first 8 years (train) and last 2 years (test).

    Train = years <= 2023, Test = years >= 2024. The two windows are disjoint by
    construction, so the model never sees any test-year row during training; we
    assert this explicitly to guard against accidental data leakage.
    """
    train_df = df[df["year"] <= TRAIN_END_YEAR]
    test_df = df[df["year"] >= TEST_START_YEAR]

    overlap = set(train_df["year"]).intersection(set(test_df["year"]))
    assert not overlap, f"Train/test leakage detected across years: {overlap}"

    return train_df, test_df


# ==========================================
# 2. MODEL TRAINING & JUSTIFICATION
# ==========================================
"""
MODEL CHOICE — Random Forest (scikit-learn `RandomForestRegressor`) over XGBoost:

1. Prediction-variability band (a CONTEXT deliverable): Random Forest is an
   ensemble of *independent* trees, so we can read a prediction interval
   straight from the spread of per-tree predictions (see `plot_forecast`).
   XGBoost's trees are sequential/additive and don't give this per-tree spread
   for free.
2. Explainability for Finance: bagged independent trees + `feature_importances_`
   are straightforward to explain to Mariana/Felipe; XGBoost's boosted residual
   fitting behaves more like a black box.
3. Small, low-noise dataset (120 monthly rows): Random Forest is robust and
   needs almost no tuning here, whereas XGBoost's edge shows mostly on larger,
   noisier data and would need careful regularization to avoid overfitting.
"""


def scale_features(X_train, X_test):
    """Standardize features using statistics learned ONLY from the training set.

    The features live on very different magnitudes (e.g. `covers_served` in the
    tens of thousands vs `avg_ticket_usd` ~ 12 and `month_num` 1-12).
    Standardizing puts them on a comparable scale so magnitude alone can't skew
    feature-importance / distance comparisons or a future gradient-based model.
    Fitting the scaler on the training split only avoids leaking test-set
    statistics into training. Column names and index are preserved.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train) -> RandomForestRegressor:
    """Fit a scikit-learn Random Forest regressor on the training window.

    See the MODEL CHOICE note above for why Random Forest is preferred over
    XGBoost for this forecasting task.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def feature_importances(model, feature_names=None) -> dict:
    """Map feature names to the fitted Random Forest's importances (explainability)."""
    names = list(feature_names) if feature_names is not None else list(FEATURES)
    return {
        name: float(importance)
        for name, importance in zip(names, model.feature_importances_)
    }


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

    # Standardize features (scaler fit on train only) to avoid faulty
    # magnitude comparisons between columns.
    X_train, X_test, _ = scale_features(X_train, X_test)

    print(
        f"Train: {len(train_df)} months ({train_df['year'].min()}-"
        f"{train_df['year'].max()})  |  "
        f"Test: {len(test_df)} months ({test_df['year'].min()}-"
        f"{test_df['year'].max()})"
    )
    print(
        "Leakage check: train/test year sets are disjoint — "
        "the model never sees the test years during training."
    )
    print("Scaled features (StandardScaler, fit on train only): " + ", ".join(FEATURES))

    model = train_model(X_train, y_train)
    results = evaluate(model, X_test, y_test, y_train)
    y_pred = results["y_pred"]

    print("Model: scikit-learn RandomForestRegressor (n_estimators=100)")
    importances = feature_importances(model)
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    print(
        "Feature importances: "
        + ", ".join(f"{name}={imp:.4f}" for name, imp in ranked)
    )

    print("--- Model Evaluation Metrics ---")
    print(f"MSE: {results['mse']:,.2f}")
    print(f"PSI: {results['psi']:.4f}")
    print(f"Gini: {results['gini']:.4f}")
    print(f"K2 Score (Kendall Tau proxy): {results['k2_tau']:.4f}")

    figure_path = plot_forecast(model, test_df, X_test, y_test, y_pred)
    print(f"Saved forecast plot to {figure_path}")


if __name__ == "__main__":
    main()
