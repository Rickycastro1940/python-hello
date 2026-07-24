# K-Means Project Tutorial — House grouping system

Repository: https://github.com/Rickycastro1940/kmeans-project  
Tutorial: https://github.com/4GeeksAcademy/k-means-project-tutorial

## Goal

Classify California Housing block groups by **region** (`Latitude`, `Longitude`) and **median income** (`MedInc`) using K-Means, then train a supervised model on the resulting cluster labels.

## Steps

| Step | Status | Description |
|------|--------|-------------|
| 1 | Done | Load `housing.csv`, keep 3 features, train/test split |
| 2 | Next | Fit K-Means (`n_clusters=6`), add `cluster`, plot |
| 3 | Pending | Predict clusters for the test set and overlay on the plot |
| 4 | Pending | Train a supervised classifier on cluster labels |
| 5 | Pending | Save both models under `models/` |

## Run Step 1

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Dataset path: `data/raw/housing.csv`  
(or auto-download from the tutorial URL if missing)

Processed splits: `data/processed/housing_train.csv`, `housing_test.csv`
