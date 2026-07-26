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

## Metric Explanations for the Finance Team
* **MSE (Mean Squared Error):** Measures the average squared difference between our forecasted revenue and the actual revenue. *Why low MSE isn't enough:* It heavily penalizes large outliers but doesn't tell us if our model's underlying distribution of predictions remains stable over time or if it's biased in one direction.
* **PSI (Population Stability Index):** Measures how much the distribution of our predicted sales shifts compared to the historical training data. A low PSI (< 0.1) confirms the model is stable and hasn't drifted.
* **Gini Coefficient:** Measures the predictive ranking power of the model. A higher Gini means the model is excellent at differentiating between high-revenue months and low-revenue months.
* **K2 Score:** Evaluates the structural dependency and correlation strength between the model's predictions and actuals.

### Contributors

This template was built as part of the [4Geeks Python Resources](https://4geeks.com/technology/python) for learning at [4Geeks.com](https://4geeks.com) by [Alejandro Sanchez](https://twitter.com/alesanchezr) and [many other contributors](https://github.com/4GeeksAcademy/python-hello/graphs/contributors).
