from pathlib import Path

import pandas as pd
import pytest

# Resolve from the repo root so the test works regardless of the working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "brasaland_sales.csv"

# Columns exactly as described in CONTEXT-brasaland ("2. Data structure").
CONTEXT_COLUMNS = [
    "month",
    "revenue_usd",
    "covers_served",
    "avg_ticket_usd",
    "market",
]
CONTEXT_MARKETS = {"colombia", "florida", "consolidated"}


@pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason=f"Provided dataset not found at {DATASET_PATH}.",
)
def test_dataset_columns_match_context():
    """The dataset columns must match those documented in CONTEXT-brasaland."""
    df = pd.read_csv(DATASET_PATH)

    assert list(df.columns) == CONTEXT_COLUMNS, (
        f"Columns {list(df.columns)} do not match the CONTEXT schema "
        f"{CONTEXT_COLUMNS}."
    )

    # `market` values must be within the documented set.
    assert set(df["market"].unique()).issubset(CONTEXT_MARKETS)

    # CONTEXT guarantees 120 consolidated monthly rows (2016-01 .. 2025-12).
    consolidated = df[df["market"] == "consolidated"].copy()
    assert len(consolidated) == 120

    # CONTEXT §5 business constraints.
    assert (consolidated["revenue_usd"] > 0).all()  # all revenue positive

    months = pd.to_datetime(consolidated["month"]).sort_values()
    assert (months.dt.day == 1).all()  # month is the first day of the month
    # No missing months across the full 2016-01 .. 2025-12 range.
    expected = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
    assert list(months.to_numpy()) == list(expected.to_numpy())
