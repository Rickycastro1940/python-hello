# AGENTS.md

## Cursor Cloud specific instructions

This is a small, single Python repository (no monorepo, no containers, no database, no CI). The primary deliverable on `main` is the **WeLoveReviews sentiment analysis** batch script; two older 4Geeks-template artifacts also live in the tree.

### Environment
- Python 3.12 is available as the system `python3`. Dependencies are installed into the system/user site-packages via `pip install -r requirements.txt` (the startup update script). There is no `.venv` — `python3 -m venv` fails here because `ensurepip` (the `python3-venv` package) is not installed, so run scripts with the system `python3` directly.
- Deps are heavy (`torch`, `transformers`, etc.); a clean install pulls large CUDA wheels even though inference runs on CPU.

### Products / how to run
- Brasaland sales forecasting: `python3 src/app.py`. Loads `data/raw/brasaland_sales.csv` (120 monthly rows), does a strict 8-year train (2016-2023) / 2-year test (2024-2025) split, trains a `RandomForestRegressor`, prints MSE/PSI/Gini/Kendall-tau metrics, and saves a forecast + variability plot to `reports/sales_forecast_variability.png`. Regenerate the (seeded, deterministic) dataset with `python3 scripts/generate_brasaland_sales.py`. Tests: `python3 -m pytest tests/`.
- Sentiment analysis: `python3 sentiment_analysis.py`. Loads `data/reviews.csv` (500 rows), downloads/caches the pinned Hugging Face model `prajjwal1/bert-mini` under `~/.cache/huggingface` (needs outbound network on first run), writes `data/reviews_with_sentiment.csv`, and prints a sentiment breakdown + star-rating comparison. One-shot batch job, not a long-running service.
- Todo CLI: `python3 main.py` (stdlib only, interactive; persists to `todos.csv`).
- Flask boilerplate (optional, unrelated to the other work): `python3 server.py` serves on port 3000. Flask is intentionally NOT in `requirements.txt`; install it separately if needed.

### Non-obvious notes
- `src/app.py` forces the matplotlib `Agg` backend (headless) so the plot renders/saves without a display. `calculate_psi` bins each input over its own min/max range, so it responds to distribution *shape* changes rather than pure location shifts (see `tests/test_app.py`). The Random Forest cannot extrapolate the upward revenue trend beyond the 2016-2023 target range, so it underpredicts the 2024-2025 peaks — this is an expected model limitation the metrics (high PSI) are meant to surface.
- The sentiment model's classification head is randomly initialized (`bert-mini` ships no fine-tuned sentiment head), so `predicted_label`/`predicted_score` are effectively arbitrary and will differ run-to-run in spirit. This is expected behavior of the exercise, not a bug — do not "fix" it.
- Tests live in `tests/` and run with `python3 -m pytest tests/` (only cover the forecasting pipeline). No linter is configured.
