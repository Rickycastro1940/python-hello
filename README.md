# Python Hello

The most basic boilerplate to start a Python project at 4Geeks is to start your very first Python project from scratch.

## What to do next?

Open the `main.py` file and start writing your code.

Execute your code by typing the following command on your terminal:

```bash
$ python main.py
```

You can create and include as many python files (a.k.a. modules) as you want using the import statements.

## Requirements

Make sure you have Python installed in your computer. We strongly recommend [installing Python through Pyenv ](https://4geeks.com/how-to/what-is-pyenv-and-how-to-install-pyenv) to avoid version conflicts in the future.

## Model Choice: Random Forest (scikit-learn)

The sales forecaster (`src/app.py`) trains a scikit-learn `RandomForestRegressor`. We chose Random Forest over XGBoost against three criteria:

* **Data size.** Small, low-noise dataset (120 monthly rows). Random Forest is robust and generalizes well at this size; XGBoost's edge appears mainly on larger, noisier data and overfits small data without heavy regularization.
* **Need for explainability.** This is a Finance RFI — bagged independent trees plus `feature_importances_` are easy to explain to stakeholders, whereas boosting behaves more like a black box.
* **Time available for tuning.** Little to none. Random Forest performs well with near-default hyper-parameters; XGBoost needs careful tuning (learning rate, depth, regularization, early stopping) to be worthwhile.

Bonus: Random Forest's independent trees also give the required 90% prediction-variability band directly from the per-tree spread, which XGBoost's additive trees don't provide out of the box.

## Metric Explanations for the Finance Team
* **MSE (Mean Squared Error):** Measures the average squared difference between our forecasted revenue and the actual revenue. *Why low MSE isn't enough:* It heavily penalizes large outliers but doesn't tell us if our model's underlying distribution of predictions remains stable over time or if it's biased in one direction.
* **PSI (Population Stability Index):** Measures how much the distribution of our predicted sales shifts compared to the historical training data. A low PSI (< 0.1) confirms the model is stable and hasn't drifted.
* **Gini Coefficient:** Measures the predictive ranking power of the model. A higher Gini means the model is excellent at differentiating between high-revenue months and low-revenue months.
* **K2 Score:** Evaluates the structural dependency and correlation strength between the model's predictions and actuals.

We also report **RMSE** (the error back in plain USD) and **MAPE** (the average error as a percentage), because those are the numbers Finance actually reads.

### Why a low MSE alone isn't enough

MSE only summarizes the *average magnitude* of the errors — it is blind to several failure modes that matter for a Finance forecast, which is exactly why we report PSI, Gini and K2 alongside it:

* **Distribution drift / bias (→ PSI).** A model can have a low average error yet systematically under- or over-predict, or see the sales distribution shift structurally between train and test (new locations, a market change). MSE won't flag this; a high **PSI** will, telling us the model needs retraining.
* **Ranking ability (→ Gini).** Two models can share the same MSE while one is far better at telling a strong month from a weak one. Mariana needs to spot underperforming months *in advance*, so ranking power (**Gini**) matters as much as absolute error.
* **Monotonic dependency (→ K2).** MSE says nothing about whether predictions consistently move in the same direction as actuals. **K2** (Kendall-tau) confirms the predictions track the real ups and downs, not just land close on average.

In short: a low MSE says "we're close on average," but only PSI + Gini + K2 confirm the model is **stable, unbiased, and directionally trustworthy** for planning decisions.

### Contributors

This template was built as part of the [4Geeks Python Resources](https://4geeks.com/technology/python) for learning at [4Geeks.com](https://4geeks.com) by [Alejandro Sanchez](https://twitter.com/alesanchezr) and [many other contributors](https://github.com/4GeeksAcademy/python-hello/graphs/contributors).
