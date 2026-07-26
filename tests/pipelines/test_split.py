from pathlib import Path

import pandas as pd
import pytest

# Resolve from the repo root so the test works regardless of the working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "brasaland_sales.csv"


@pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason=(
        f"Provided dataset not found at {DATASET_PATH}. Add the real file "
        "(do not generate or simulate it) to run this integrity check."
    ),
)
def test_train_test_split_integrity():
    """
    Validates that the dataset is split strictly into:
    - Train: First 8 years (<= 2023)
    - Test: Last 2 years (>= 2024)
    And ensures no data leakage between the two.
    """
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year

    # Perform split
    train_df = df[df['year'] <= 2023]
    test_df = df[df['year'] >= 2024]

    # 1. Assert chronological correctness
    assert train_df['year'].max() == 2023, "Training set should end in 2023."
    assert test_df['year'].min() == 2024, "Testing set should begin in 2024."

    # 2. Assert no data leakage (no overlapping dates)
    train_dates = set(train_df['month'])
    test_dates = set(test_df['month'])
    intersection = train_dates.intersection(test_dates)

    assert len(intersection) == 0, f"Data leakage detected! Overlapping dates: {intersection}"

    # 3. Assert total length matches original
    assert len(train_df) + len(test_df) == len(df), "Row counts do not match after split."
