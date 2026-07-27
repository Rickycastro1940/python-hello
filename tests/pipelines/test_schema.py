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
    consolidated = df[df["market"] == "consolidated"]
    assert len(consolidated) == 120
    assert (consolidated["revenue_usd"] > 0).all()
