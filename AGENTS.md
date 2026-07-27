# AGENTS.md

## Cursor Cloud specific instructions

This is a small, single Python repository (no monorepo, no containers, no database, no CI). The primary deliverable on `main` is the **WeLoveReviews sentiment analysis** batch script; two older 4Geeks-template artifacts also live in the tree.

### Environment
- Dependency management uses **`uv`** (canonical: `pyproject.toml` + `uv.lock`). `uv` is installed to `~/.local/bin` and the startup update script runs `uv sync`, which creates a project `.venv` (gitignored). Do NOT use `pip`/`pipenv` to add deps — use `uv add <pkg>` (and `uv add --dev <pkg>` for test/dev tools). `requirements.txt` is kept only as a pre-merge fallback and may lag `pyproject.toml`.
- Run everything through the uv venv with `uv run ...` (e.g. `uv run python src/app.py`, `uv run pytest`). If `uv` is not on `PATH`, call it as `~/.local/bin/uv`. Python is pinned to 3.12 via `.python-version`.
- Deps are heavy (`torch`, `transformers`, etc.); a clean `uv sync` pulls large CUDA wheels even though inference runs on CPU.
- Version pins that matter: `transformers==4.57.1` / `torch==2.7.1` are pinned on purpose. `transformers` 5.x breaks the sentiment pipeline because `prajjwal1/bert-mini`'s `config.json` has no `model_type` key (4.x tolerated it). Do not bump these without re-testing `sentiment_analysis.py`.

### Products / how to run
- Brasaland sales forecasting: `uv run python src/app.py`. Does a strict 8-year train (2016-2023) / 2-year test (2024-2025) split, trains a `RandomForestRegressor`, prints MSE/PSI/Gini/Kendall-tau metrics, and saves a forecast + variability plot to `reports/sales_forecast_variability.png`. Tests: `uv run pytest tests/`.
- Dataset is **provided, never generated/simulated**. The real file is committed at `data/raw/brasaland_sales.csv` (120 monthly `consolidated` rows, 2016-2025). `src/app.py` (`resolve_dataset_path`) loads the first that exists: `content/contexts/sales-forecasting/brasaland/brasaland_sales.csv` (reference repo) then `data/raw/brasaland_sales.csv` (monorepo). If neither is present it raises `FileNotFoundError` instead of falling back to synthetic data, and the data-dependent tests skip.
- Dataset spec lives in `content/contexts/sales-forecasting/brasaland/CONTEXT-brasaland.en.md` (mirrored from the 4Geeks ai-engineering-syllabus). Columns (validated in `load_and_prepare_data` against `EXPECTED_COLUMNS`, and by `tests/pipelines/test_schema.py`): `month, revenue_usd, covers_served, avg_ticket_usd, market`. The model uses the `market == "consolidated"` rows; target is `revenue_usd`; features are `covers_served, avg_ticket_usd` + engineered `year, month_num`.
- Sentiment analysis: `uv run python sentiment_analysis.py`. Loads `data/reviews.csv` (500 rows), downloads/caches the pinned Hugging Face model `prajjwal1/bert-mini` under `~/.cache/huggingface` (needs outbound network on first run), writes `data/reviews_with_sentiment.csv`, and prints a sentiment breakdown + star-rating comparison. One-shot batch job, not a long-running service. Source of `data/reviews.csv` (WeLoveReviews project): https://github.com/4GeeksAcademy/ai-engineering-syllabus/blob/main/content/projects/existing-model-sentiment-analysis-reviews/reviews.csv (the committed copy already matches this source exactly).
- Todo CLI: `uv run python main.py` (stdlib only, interactive; persists to `todos.csv`).
- Flask boilerplate (optional, unrelated to the other work): `uv run python server.py` serves on port 3000. Flask is intentionally NOT a project dependency; `uv add flask` first if you need it.

### Non-obvious notes
- `src/app.py` forces the matplotlib `Agg` backend (headless) so the plot renders/saves without a display. `calculate_psi` bins each input over its own min/max range, so it responds to distribution *shape* changes rather than pure location shifts (see `tests/test_app.py`).
- The sentiment model's classification head is randomly initialized (`bert-mini` ships no fine-tuned sentiment head), so `predicted_label`/`predicted_score` are effectively arbitrary and will differ run-to-run in spirit. This is expected behavior of the exercise, not a bug — do not "fix" it.
- Tests live in `tests/` and run with `uv run pytest tests/` (only cover the forecasting pipeline). No linter is configured.
