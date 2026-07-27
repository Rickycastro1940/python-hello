# Brasaland Sales Model — Technical Evaluation Report (WORK IN PROGRESS)

Evaluation of the `RandomForestRegressor` sales forecaster (see `src/app.py`)
requested by the tech-lead ticket before promotion to staging.

Reference: `content/contexts/sales-forecasting/brasaland/CONTEXT-brasaland.en.md`.

---

## 1. Business cost of errors (from CONTEXT-brasaland)

The model feeds purchasing decisions: **Felipe (Operations)** anticipates
ingredient purchases from the expected trend, and **Lucía (Procurement)**
buys ahead of **meat price fluctuations** based on projected volume. Brasaland
is a churrascaria, so its dominant input is **perishable meat**.

| Error type | What it means | Business consequence |
|---|---|---|
| **Under-forecast** (predicted < actual) | We expect fewer sales than happen | Stockouts during peak demand (e.g. December), lost revenue and guest dissatisfaction, and forced reactive meat purchases at higher spot prices |
| **Over-forecast** (predicted > actual) | We expect more sales than happen | Over-purchasing perishable meat → spoilage/waste and tied-up capital |

Both are costly, but the **largest single misses are the most damaging** — a big
miss in a high-revenue month (December) means either a major stockout or a major
perishable overstock. Because the model's biggest errors are concentrated at the
December peaks, the metric we prioritize should **penalize large errors more**.

### Primary metric choice: RMSE (with MAE reported alongside)
- **RMSE** penalizes large errors quadratically → it reflects Brasaland's real
  cost, where a few big misses (holiday peaks) dominate the operational risk.
- **MAE** (linear) is reported for interpretability (average USD miss) but is
  less aligned with the "large errors hurt disproportionately" reality here.

_(This justification is required by the module; final numbers below.)_

---

## 2. Time-aware cross-validation  _(TODO — to implement)_

- Strategy: `TimeSeriesSplit`, ≥5 folds over the training window (2016-2023),
  chronological order preserved (no shuffling).
- Result: report chosen metric as **mean ± std** across folds.

> _Results pending implementation (`src/evaluate_model.py`)._

---

## 3. Learning curve  _(TODO — to implement)_

- Plot training error vs validation error as the training set grows.
- Image saved to `data/eval/learning_curve.png`.

> _Interpretation pending._

---

## 4. Metrics (train vs validation)  _(TODO — to implement)_

| Metric | Train | Validation |
|---|---|---|
| MAE | _pending_ | _pending_ |
| RMSE | _pending_ | _pending_ |

---

## 5. Diagnosis & corrective action  _(TODO — to implement)_

- Classification: **well fitted / underfitting / overfitting** — backed by the
  learning curve and CV spread.
- Concrete, root-cause corrective action (not a generic "add more data").

> _Diagnosis pending evidence from sections 2-4._
