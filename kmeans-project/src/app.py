"""K-Means house grouping — California Housing tutorial.

4Geeks: https://github.com/4GeeksAcademy/k-means-project-tutorial

Step 1: load housing.csv, keep MedInc/Latitude/Longitude, train/test split.
Later steps (2–5): K-Means clusters, test predictions, supervised model, save.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "housing.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES = ROOT / "figures"

DATASET_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/"
    "k-means-project-tutorial/main/housing.csv"
)
FEATURE_COLS = ["MedInc", "Latitude", "Longitude"]
TEST_SIZE = 0.2
RANDOM_SEED = 42
N_CLUSTERS = 6


def load_housing(csv_path: Path = RAW_CSV, url: str = DATASET_URL) -> pd.DataFrame:
    """Load the California Housing dataset from disk or the tutorial URL."""
    if csv_path.exists():
        print(f"Loading dataset from {csv_path}")
        data = pd.read_csv(csv_path)
    else:
        print(f"Local file missing — loading from {url}")
        data = pd.read_csv(url)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv_path, index=False)
        print(f"Saved a local copy to {csv_path}")
    return data


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only MedInc, Latitude, and Longitude (tutorial Step 1)."""
    missing = [c for c in FEATURE_COLS if c not in data.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    return data[FEATURE_COLS].copy()


def split_train_test(
    features: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test for unsupervised fit + later cluster prediction."""
    X_train, X_test = train_test_split(
        features,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True)


def save_splits(X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: Path = PROCESSED) -> None:
    """Persist train/test feature tables under data/processed/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "housing_train.csv"
    test_path = out_dir / "housing_test.csv"
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)
    print(f"Saved train split: {train_path} ({len(X_train)} rows)")
    print(f"Saved test split:  {test_path} ({len(X_test)} rows)")


def step1_load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Step 1 of the tutorial and return train/test feature frames."""
    print("=== Step 1: Loading the dataset ===")
    total_data = load_housing()
    print(f"Full dataset shape: {total_data.shape}")
    print(f"Columns: {list(total_data.columns)}")
    print(total_data.head())

    print("\nKeeping only:", FEATURE_COLS)
    X = select_features(total_data)
    print(X.describe().round(3))

    X_train, X_test = split_train_test(X)
    print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")
    print("Train head:")
    print(X_train.head())

    save_splits(X_train, X_test)
    return X_train, X_test


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    step1_load_and_split()
    print(
        "\nStep 1 complete. Next: Step 2 — build KMeans "
        f"with n_clusters={N_CLUSTERS}."
    )


if __name__ == "__main__":
    main()
