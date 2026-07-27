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
MODEL CHOICE — Random Forest (scikit-learn `RandomForestRegressor`) over XGBoost.

Decision criteria used to pick the algorithm:

- DATA SIZE: the dataset is small and low-noise (120 monthly `consolidated`
  rows; 96 train / 24 test). Random Forest is robust and generalizes well at
  this size; XGBoost's accuracy edge typically appears on much larger, noisier
  datasets and it is easier to overfit small data without heavy regularization.
- NEED FOR EXPLAINABILITY: this is a Finance RFI, so stakeholders (Mariana,
  Felipe) must be able to trust and follow the model. Bagging independent trees
  and reading `feature_importances_` is easy to explain; XGBoost's sequential
  residual boosting behaves more like a black box.
- TIME AVAILABLE FOR TUNING: little/none. Random Forest gives strong results
  with essentially default hyper-parameters, while XGBoost needs careful tuning
  (learning rate, depth, regularization, early stopping) to be worthwhile.

Bonus (a CONTEXT deliverable): Random Forest's *independent* trees let us read a
prediction-variability band straight from the per-tree spread (see
`plot_forecast`); XGBoost's additive trees don't provide that for free.
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
# What each metric measures, and why a low MSE alone is not enough (see README
# "Why a low MSE alone isn't enough" for the Finance-facing version):
#   - MSE  : average squared error magnitude (report RMSE/MAPE for readability).
#            Blind to bias, distribution drift, ranking quality and direction.
#   - PSI  : distribution shift of predictions vs the training target -> catches
#            drift/bias that a low MSE would hide (high PSI => retrain).
#   - Gini : ranking power (can the model separate strong vs weak months?) ->
#            two models with equal MSE can rank very differently.
#   - K2   : Kendall-tau dependency -> confirms predictions move in the same
#            direction as actuals, not just close on average.
# A low MSE only says "close on average"; PSI + Gini + K2 confirm the model is
# stable, unbiased and directionally trustworthy for planning.
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
    """Compute the required metrics on the TEST set (2024-2025).

    Reports at least MSE, PSI, Gini and K2 (Kendall-tau). MSE is also translated
    into RMSE (USD) and MAPE (%) because, per the CONTEXT, that is how Finance
    actually reads the error.
    """
    y_pred = model.predict(X_test)
    y_test_arr = np.asarray(y_test, dtype=float)

    # MSE (USD^2) on the test set, plus human-readable RMSE / MAPE.
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_test_arr - y_pred) / y_test_arr)) * 100)

    # PSI: prediction distribution vs the training target (drift signal).
    psi_score = calculate_psi(y_train, y_pred)
    # Gini: ranking power on the test set.
    gini_score = normalized_gini(y_test_arr, y_pred)
    # K2 Score: Kendall-tau dependency between predictions and actuals.
    tau, k2_p_value = stats.kendalltau(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "psi": psi_score,
        "gini": gini_score,
        "k2_tau": tau,
        "k2_p_value": k2_p_value,
    }


# ==========================================
# 4. VISUALIZATION (PREDICTIONS + VARIABILITY)
# ==========================================
def prediction_interval(model, X_test, lower_pct: float = 5, upper_pct: float = 95):
    """Return (lower, upper) per-month bounds from the spread of the RF trees."""
    X = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    return (
        np.percentile(tree_preds, lower_pct, axis=0),
        np.percentile(tree_preds, upper_pct, axis=0),
    )


def build_comparison(test_df, y_test, y_pred, lower=None, upper=None) -> pd.DataFrame:
    """Build a month-by-month comparison of predictions vs the real test data.

    Columns: month, actual_revenue, predicted_revenue, error, abs_pct_error and
    (when bounds are given) the 90% band and whether the actual falls inside it.
    """
    y_test_arr = np.asarray(y_test, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    comparison = pd.DataFrame(
        {
            "month": pd.to_datetime(test_df["month"]).dt.strftime("%Y-%m").values,
            "actual_revenue": y_test_arr,
            "predicted_revenue": y_pred_arr,
        }
    )
    comparison["error"] = comparison["predicted_revenue"] - comparison["actual_revenue"]
    comparison["abs_pct_error"] = (
        comparison["error"].abs() / comparison["actual_revenue"] * 100
    )

    if lower is not None and upper is not None:
        comparison["lower_90"] = np.asarray(lower, dtype=float)
        comparison["upper_90"] = np.asarray(upper, dtype=float)
        comparison["within_90_band"] = (
            comparison["actual_revenue"] >= comparison["lower_90"]
        ) & (comparison["actual_revenue"] <= comparison["upper_90"])

    return comparison


def plot_forecast(
    model,
    test_df,
    X_test,
    y_test,
    y_pred,
    path: Path = FIGURE_PATH,
    metrics=None,
    lower_bound=None,
    upper_bound=None,
):
    """Plot the model's prediction and its 90% variability area vs the real data.

    The variability area is the 5th-95th percentile band of the individual tree
    predictions (a Random-Forest prediction interval), drawn against the actual
    revenue of the 2 test years. If `metrics` is provided, a small summary box
    is annotated on the chart.
    """
    # Reuse provided bounds, or derive the band from the per-tree spread.
    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound = prediction_interval(model, X_test)

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
        label="Predicted Revenue (RF mean)",
        color="red",
        linestyle="--",
        marker="x",
    )
    plt.fill_between(
        test_df["month"],
        lower_bound,
        upper_bound,
        color="red",
        alpha=0.2,
        label="90% Prediction Variability (per-tree 5th-95th pct)",
    )

    plt.title("Brasaland Sales Forecasting (Test Set: 2024-2025)")
    plt.xlabel("Date")
    plt.ylabel("Revenue (USD)")
    plt.legend(loc="upper left")
    plt.grid(True)

    if metrics is not None:
        summary = (
            f"RMSE: {metrics['rmse']:,.0f} USD\n"
            f"MAPE: {metrics['mape']:.2f}%\n"
            f"PSI: {metrics['psi']:.3f}\n"
            f"Gini: {metrics['gini']:.3f}\n"
            f"K2: {metrics['k2_tau']:.3f}"
        )
        plt.gca().text(
            0.99,
            0.03,
            summary,
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

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

    print("--- Model Evaluation Metrics (test set: 2024-2025) ---")
    print(
        f"MSE:  {results['mse']:,.2f} USD^2  "
        f"(RMSE: {results['rmse']:,.2f} USD  |  MAPE: {results['mape']:.2f}%)"
    )
    print(f"PSI:  {results['psi']:.4f}")
    print(f"Gini: {results['gini']:.4f}")
    print(f"K2 Score (Kendall Tau proxy): {results['k2_tau']:.4f}")

    # --- Compare predictions with the real data from the 2 test years ---
    lower_bound, upper_bound = prediction_interval(model, X_test)
    comparison = build_comparison(test_df, y_test, y_pred, lower_bound, upper_bound)

    print("\n--- Predicted vs Actual (2 test years: 2024-2025) ---")
    display = comparison.copy()
    for col in ("actual_revenue", "predicted_revenue", "error"):
        display[col] = display[col].map(lambda v: f"{v:,.0f}")
    display["abs_pct_error"] = display["abs_pct_error"].map(lambda v: f"{v:.2f}%")
    print(
        display[
            ["month", "actual_revenue", "predicted_revenue", "error",
             "abs_pct_error", "within_90_band"]
        ].to_string(index=False)
    )

    coverage = comparison["within_90_band"].mean() * 100
    total_actual = comparison["actual_revenue"].sum()
    total_pred = comparison["predicted_revenue"].sum()
    print(
        f"\nMonths inside the 90% band: "
        f"{int(comparison['within_90_band'].sum())}/{len(comparison)} "
        f"({coverage:.0f}%)"
    )
    print(
        f"Totals over the 2 test years — actual: {total_actual:,.0f} USD  |  "
        f"predicted: {total_pred:,.0f} USD  |  "
        f"diff: {(total_pred / total_actual - 1) * 100:+.2f}%"
    )

    comparison_path = PROJECT_ROOT / "reports" / "test_predictions_vs_actual.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    print(f"Saved comparison table to {comparison_path}")

    figure_path = plot_forecast(
        model,
        test_df,
        X_test,
        y_test,
        y_pred,
        metrics=results,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    print(f"Saved forecast plot to {figure_path}")


if __name__ == "__main__":
    main()
