"""Generate a synthetic monthly sales dataset for Brasaland (2016-2025).

The dataset backs the forecasting model in ``src/app.py``. It intentionally
covers 10 full years so the pipeline can use the strict 8-year training window
(2016-2023) and the 2 most recent years (2024-2025) as the test set.

Revenue is modeled as ``covers_served * avg_ticket_usd`` plus a small amount of
operational noise, which mirrors how a restaurant's monthly revenue actually
arises (guests served times the average check). Both drivers carry a long-term
growth trend plus month-of-year seasonality (busier in the Dec holidays and
mid-year, quieter in the Feb/Sep shoulder months).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "brasaland_sales.csv"
)

# Multiplicative seasonal factors by calendar month (Jan..Dec).
SEASONALITY = np.array(
    [0.95, 0.88, 0.98, 1.00, 1.06, 1.10, 1.12, 1.08, 0.96, 1.02, 1.05, 1.30]
)


def generate(seed: int = 42) -> pd.DataFrame:
    """Build the monthly Brasaland sales dataframe for 2016-2025."""
    rng = np.random.default_rng(seed)

    months = pd.date_range(start="2016-01-01", end="2025-12-01", freq="MS")
    records = []
    for period_index, month in enumerate(months):
        # ~5% annual growth in guest volume, compounded monthly.
        trend = (1.05) ** (period_index / 12.0)
        season = SEASONALITY[month.month - 1]

        base_covers = 3200 * trend * season
        covers_served = int(
            max(500, rng.normal(base_covers, base_covers * 0.05))
        )

        # Average check drifts up with inflation (~3%/yr) plus seasonality.
        base_ticket = 52.0 * (1.03) ** (period_index / 12.0)
        avg_ticket_usd = round(
            max(20.0, rng.normal(base_ticket * (0.97 + 0.05 * season), 1.5)),
            2,
        )

        # Revenue is covers * ticket with a little operational noise.
        revenue_usd = round(
            covers_served * avg_ticket_usd * rng.normal(1.0, 0.02), 2
        )

        records.append(
            {
                "month": month.strftime("%Y-%m-%d"),
                "covers_served": covers_served,
                "avg_ticket_usd": avg_ticket_usd,
                "revenue_usd": revenue_usd,
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(df.head())
    print(df.tail())
