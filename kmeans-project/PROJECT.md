# K-Means Project Tutorial — House grouping system

Repository: https://github.com/Rickycastro1940/kmeans-project  
Tutorial: https://github.com/4GeeksAcademy/k-means-project-tutorial

## Goal

Classify California Housing block groups by **region** (`Latitude`, `Longitude`) and **median income** (`MedInc`) using K-Means, then train a supervised model on the resulting cluster labels.

## Steps

| Step | Status | Description |
|------|--------|-------------|
| 1 | Done | Load `housing.csv`, keep 3 features, train/test split |
| 2 | Done | Fit K-Means (`n_clusters=6`), add `cluster`, plot |
| 3 | Done | Predict clusters for the test set and overlay on the plot |
| 4 | Done | Train Decision Tree on cluster labels |
| 5 | Done | Save both models under `models/` |

## Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Outputs

| Path | Description |
|------|-------------|
| `data/raw/housing.csv` | Source dataset |
| `data/processed/housing_*_clustered.csv` | Labeled train/test |
| `figures/kmeans_train_clusters.png` | Step 2 clusters |
| `figures/kmeans_test_overlay.png` | Step 3 test overlay |
| `figures/decision_tree.png` | Step 4 tree view |
| `models/k-means_default_42.sav` | Unsupervised model |
| `models/decision_tree_classifier_default_42.sav` | Supervised model |
