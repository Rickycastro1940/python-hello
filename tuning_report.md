# StreamLoop Churn — Hyperparameter Tuning Report

Tuning of the churn classifier for StreamLoop (data publicly modeled after the
IBM Telco Customer Churn dataset). Reproduced by `notebooks/streamloop_churn_eda.ipynb`
(Step 11); run with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/streamloop_churn_eda.ipynb`.

## 1. Chosen classifier

`GradientBoostingClassifier` (scikit-learn) — the strongest model in the Step 9
comparison (ROC-AUC ~0.846, ahead of Random Forest and tied with balanced
Logistic Regression).

## 2. Scoring metric — business priority, not the default

**`scoring='roc_auc'`.** StreamLoop's goal is to *rank customers by churn risk*
so retention offers reach the right people. The target is imbalanced (~26.5%
churn), so the sklearn default (accuracy) is misleading — a model that predicts
"no churn" for everyone scores ~73.5% accuracy while catching zero churners.
ROC-AUC measures how well the model ranks churners above non-churners
independent of the decision threshold, which matches the retention use-case.

## 3. Search strategy

Broad random search first, then a focused grid search around the winning region.
Both use 5-fold cross-validation, `n_jobs=-1`, and `refit=True`.

### 3a. `RandomizedSearchCV` (broad)

`RandomizedSearchCV(GradientBoostingClassifier(random_state=0), n_iter=20, cv=5, scoring='roc_auc', n_jobs=-1, random_state=0, refit=True)`

Search space (only parameters `GradientBoostingClassifier` supports):

| Parameter | Distribution / values |
|---|---|
| `n_estimators` | `randint(100, 350)` |
| `learning_rate` | `loguniform(1e-2, 3e-1)` |
| `max_depth` | `randint(2, 5)` |
| `min_samples_leaf` | `randint(1, 30)` |
| `subsample` | `[0.7, 0.9, 1.0]` |
| `max_features` | `['sqrt', 'log2', None]` |

**Result:** best CV ROC-AUC **0.8488**, best params
`{learning_rate: 0.0790, max_depth: 2, max_features: 'log2', min_samples_leaf: 11, n_estimators: 184, subsample: 0.7}`.

### 3b. `GridSearchCV` (refine)

The space was narrowed around the random-search winner (`n_estimators` ±75,
`learning_rate` ×/÷2, `max_depth` ±1; the remaining winners held fixed) and
refined with `GridSearchCV(..., cv=5, scoring='roc_auc', n_jobs=-1, refit=True)`.

**Result:** best CV ROC-AUC **0.8486**, best params
`{learning_rate: 0.079, max_depth: 2, max_features: 'log2', min_samples_leaf: 11, n_estimators: 184, subsample: 0.7}`
(the random-search region was already near-optimal, so the grid confirmed it).

## 4. Refit

`refit=True` (the default) means the search re-fits the best configuration on the
full training set automatically. We evaluate `grid_search.best_estimator_`
directly — no manual re-fit.

## 5. Test-set performance (tuned `best_estimator_`)

| Metric | Untuned Gradient Boosting | Tuned Gradient Boosting |
|---|---|---|
| ROC-AUC | 0.846 | **0.850** |
| Accuracy | 0.800 | 0.805 |

Classification report (tuned, default 0.5 threshold):

```
              precision    recall  f1-score   support
      Stayed       0.84      0.91      0.87      1552
     Churned       0.67      0.53      0.59       561
    accuracy                           0.80      2113
```

## 6. Takeaways

- Tuning gave a small but real lift in the business metric (test ROC-AUC
  0.846 → 0.850); the model was already near its ceiling for this feature set.
- The winner is a **shallow** ensemble (`max_depth=2`, `learning_rate≈0.08`,
  `n_estimators=184`, `subsample=0.7`) — mild regularization generalizes best.
- ROC-AUC (ranking) is strong, but churn **recall at the default threshold is
  only 0.53**. For retention, the next step is to **lower the decision threshold**
  (or use `class_weight`/resampling) to trade some precision for the higher
  recall StreamLoop needs to catch more at-risk customers.
