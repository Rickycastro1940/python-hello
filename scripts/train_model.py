"""CONTEXT-brasaland training script (Expected deliverable, section 6).

Loads `data/raw/brasaland_sales.csv`, splits the first 8 years (2016-2023) as
training and the last 2 (2024-2025) as test, trains the Random Forest, reports
MSE/PSI/Gini/K2 on the test set, compares predictions against the real test
years, and saves the prediction + variability plot.

This is a thin entrypoint over the reusable pipeline in `src/app.py`; run it
with `uv run python scripts/train_model.py`.
"""

import sys
from pathlib import Path

# Make the project root importable so `src.app` resolves when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import main  # noqa: E402

if __name__ == "__main__":
    main()
