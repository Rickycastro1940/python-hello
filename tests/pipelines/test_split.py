from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import app

# Resolve from the repo root so the test works regardless of the working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "brasaland_sales.csv"


def _ten_years_monthly() -> pd.DataFrame:
    """A deterministic 2016-01..2025-12 monthly frame shaped like the loader output."""
    months = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
    return pd.DataFrame(
        {
            "month": months,
            "revenue_usd": np.arange(1, len(months) + 1, dtype=float),
            "covers_served": np.arange(len(months)) + 100,
            "avg_ticket_usd": 12.0,
            "market": "consolidated",
            "year": months.year,
            "month_num": months.month,
        }
    )


def test_split_data_respects_8_2_rule_and_has_no_leakage():
    """`split_data` must give 8 training years / 2 test years with no leakage."""
    df = _ten_years_monthly()

    train_df, test_df = app.split_data(df)

    # 8-year / 2-year rule: exact year membership and month counts.
    assert sorted(train_df["year"].unique()) == list(range(2016, 2024))
    assert sorted(test_df["year"].unique()) == [2024, 2025]
    assert len(train_df) == 8 * 12
    assert len(test_df) == 2 * 12

    # No data leakage: disjoint years AND months, with a clean chronological cut.
    assert set(train_df["year"]).isdisjoint(set(test_df["year"]))
    assert set(train_df["month"]).isdisjoint(set(test_df["month"]))
    assert train_df["month"].max() < test_df["month"].min()

    # Nothing dropped or duplicated across the split.
    assert len(train_df) + len(test_df) == len(df)
    recombined = (
        pd.concat([train_df, test_df]).sort_values("month").reset_index(drop=True)
    )
    assert recombined["month"].equals(
        df.sort_values("month").reset_index(drop=True)["month"]
    )


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
