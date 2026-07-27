# Brasaland Sales Model — Technical Evaluation Report

Formal evaluation of the `RandomForestRegressor` sales forecaster (`src/app.py`),
requested by the tech-lead ticket before promoting the model to staging.
Generated with `uv run python src/evaluate_model.py` (metrics in
`data/eval/metrics.json`, learning curve in `data/eval/learning_curve.png`).

Reference: `content/contexts/sales-forecasting/brasaland/CONTEXT-brasaland.en.md`.

---

## 1. Business cost of errors & metric choice (from CONTEXT-brasaland)

The model feeds purchasing: **Felipe (Operations)** anticipates ingredient
purchases from the expected trend, and **Lucía (Procurement)** buys ahead of
**meat price fluctuations** based on projected volume. Brasaland is a
grilled-food restaurant chain (14 locations across Colombia and Florida), and
**meat** is its primary perishable input.

| Error type | Consequence |
|---|---|
| **Under-forecast** (pred < actual) | Stockouts in peak months, lost revenue, and reactive meat purchases at higher spot prices |
| **Over-forecast** (pred > actual) | Over-buying perishable meat → spoilage/waste and tied-up capital |

The most damaging errors are the **large single misses** (e.g. the December
holiday peaks): a big miss there is either a major stockout or a major
perishable overstock.

### Which direction is more costly? → **Over-estimation**
The CONTEXT/briefing don't state an explicit policy, so we reason from the
domain (perishable-inventory "newsvendor" logic):

- **Over-estimating sales** → over-order perishable meat → it **spoils**. This
  is an **unrecoverable full-cost loss** (we paid for meat we throw away) plus
  tied-up working capital. Cost ≈ the wasted meat's COGS.
- **Under-estimating sales** → under-order → **stockout** → we lose the
  **margin** on unserved covers (we never bought that meat, so no COGS was
  sunk), partially mitigable by substitution — a smaller per-unit loss than
  wasted premium meat.

Because wasted premium meat (full COGS) generally exceeds the lost margin of an
unserved cover, **over-estimation is the more costly error direction** for
Brasaland, and the forecast should lean slightly conservative.

> ⚠️ **Caveat that matters here:** our model's actual bias is the *opposite* —
> it systematically **under-forecasts**, especially the December peaks. Most
> months that under-bias is the "safer" (anti-spoilage) direction, but a
> December **stockout in the highest-revenue month** is a serious brand hit for
> a grill chain whose promise is a fast, consistent kitchen. So the model's
> single most business-relevant flaw is the peak under-forecast — reinforcing
> the corrective actions in §5.

**Primary metric: RMSE** — it penalizes large errors quadratically, so it
reflects Brasaland's real risk where a few big misses dominate. **MAE** is
reported alongside as the average USD miss (interpretability), but it is less
aligned with the "large errors hurt disproportionately" reality here. If the
business confirms the spoilage-vs-stockout asymmetry, an **asymmetric loss**
(penalizing over-prediction more) could be layered on top later.

---

## 2. Time-aware cross-validation (`TimeSeriesSplit`, 5 folds, no shuffle)

Cross-validation over the 96-month training window (2016-2023). Folds are
chronological (expanding window); no shuffling — each validation fold is
strictly later in time than its training portion.

| Metric | Validation (mean ± std) | Train (mean ± std) |
|---|---|---|
| **RMSE** | **33,688 ± 6,984 USD** | 7,678 ± 1,892 USD |
| **MAE** | **25,768 ± 6,477 USD** | 5,471 ± 1,262 USD |

Validation RMSE per fold: `[25,119, 42,634, 30,219, 29,167, 41,300]`.
Reference scale: mean monthly training revenue ≈ **605,468 USD** (so validation
RMSE ≈ 5.6% of the monthly mean; training RMSE ≈ 1.3%).

---

## 3. Learning curve

![Learning curve](learning_curve.png)

Chronological expanding-window curve: train on a growing prefix of history,
validate on a **fixed 24-month future block** (the last 2 training years).

| Train size (months) | 12 | 24 | 36 | 48 | 60 | 72 |
|---|---|---|---|---|---|---|
| Training RMSE (USD) | 13,206 | 8,216 | 5,829 | 6,636 | 7,822 | 6,936 |
| Validation RMSE (USD) | 161,313 | 129,135 | 114,503 | 70,303 | 59,688 | 37,653 |

**Pattern:** training error is consistently **low and flat** (~6–13k), while
validation error is **much higher but drops steeply** as more history is added
(161k → 38k). A **wide gap persists** between the two lines even at the largest
training size, but it is **shrinking** as data grows.

---

## 4. Metrics (train vs validation)

| Metric | Train (resubstitution) | Train (CV) | Validation (CV) |
|---|---|---|---|
| MAE  | 4,981 | 5,471 ± 1,262 | **25,768 ± 6,477** |
| RMSE | 7,268 | 7,678 ± 1,892 | **33,688 ± 6,984** |

---

## 5. Diagnosis: OVERFITTING (high variance)

The evidence points to **overfitting**, i.e. high variance:

- **Wide train↔validation gap.** Validation RMSE (33,688) is **~4.4×** the
  training RMSE (7,678); training error is only ~1.3% of the monthly mean while
  validation error is ~5.6%. Low training error + substantially higher
  validation error is the signature of a model that memorizes the training
  months rather than generalizing.
- **Learning curve confirms it.** The gap between the low, flat training curve
  and the high validation curve is large and persistent (not converging at a
  common low error → not well-fitted; training error is low, not high → not
  underfitting). The validation curve is still **declining** at 72 months,
  meaning the variance is data-limited.

This is **not underfitting** (training error is low, so the model captures the
in-sample pattern) and **not a good fit** (the validation gap is too wide).

### Root cause (two components, isolated by experiment)
The Random Forest uses scikit-learn defaults — `max_depth=None` and
`min_samples_leaf=1` — so its 100 unbounded trees drive each leaf toward a
single month, memorizing training noise (the variance / low-train-error part).
But the **dominant** driver of the validation error is **structural bias**: the
strongest feature (`covers_served`) and the target both grow over time, and tree
models cannot extrapolate beyond the training target range, so future months are
systematically under-predicted (the same effect behind the high PSI and the
December under-forecast on the real test set).

I isolated the two components by re-running the **same time-aware CV** with a
regularized RF (`max_depth=6`, `min_samples_leaf=4`):

| Model | Train RMSE | Validation RMSE | val/train gap |
|---|---|---|---|
| Default RF (`max_depth=None`, `min_samples_leaf=1`) | 7,678 | 33,688 | 4.39 |
| Regularized RF (`max_depth=6`, `min_samples_leaf=4`) | 21,800 | 46,219 | 2.12 |

Regularization **narrows the gap** (4.39 → 2.12) — confirming a real variance
component — **but validation RMSE gets worse** (33,688 → 46,219). That is the
key evidence: if the problem were only variance, reducing complexity would help
validation; instead it hurts, because shallower trees track the rising level
even less well. So the bottleneck is **trend-extrapolation bias**, not variance.

### Corrective action (specific, root-cause-justified — reproduce via `regularization_experiment`)
1. **Primary — make the target extrapolable (fix the dominant bias).** Model a
   **detrended / relative** target instead of raw USD: e.g. predict
   **revenue-per-cover** (then multiply by a covers forecast), or **year-over-year
   growth**, or fit an explicit **linear trend and have the RF predict the
   residual**. Justification: the learning-curve validation error keeps falling
   as history grows (bias from incomplete trend coverage), and the experiment
   above shows plain regularization *worsens* validation — both point to trend
   bias, which detrending removes so the model can extrapolate the growth.
2. **Secondary — light regularization, but only after detrending.** A modest
   `min_samples_leaf`/`max_depth`, **tuned on the same `TimeSeriesSplit` CV**,
   trims the residual variance gap. It must come *after* (1): applied to the raw
   target it degrades validation (shown above), so it is not a standalone fix.

> Explicitly **rejected** generic answers:
> - *"Just add more data"* — we already use all 8 years of history and cannot
>   buy more past months; the gap is driven by trend bias, not sample count.
> - *"Just increase complexity"* — the RF already memorizes the training set
>   (train RMSE ≈ 1.3% of mean); more complexity worsens variance, not the bias.

---

## 6. Recommendation

**Do not promote as-is** — the model is overfitting (high variance) **and**, more
importantly, biased against the upward trend. Fix the target so it is
extrapolable (corrective action 1), then add light, CV-tuned regularization
(action 2), and re-run this exact evaluation (`uv run python src/evaluate_model.py`)
to confirm both the val/train gap and the validation RMSE improve before
promotion to staging.
