"""K-Means house grouping — California Housing tutorial.

4Geeks: https://github.com/4GeeksAcademy/k-means-project-tutorial

Step 1: load housing.csv, keep MedInc/Latitude/Longitude, train/test split.
Step 2: fit K-Means (6 clusters), store cluster labels, plot.
Step 3: predict clusters on the test set and overlay on the plot.
Step 4: train a Decision Tree on the unsupervised labels.
Step 5: save both models under models/.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "housing.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES = ROOT / "figures"

DATASET_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/"
    "k-means-project-tutorial/main/housing.csv"
)
FEATURE_COLS = ["MedInc", "Latitude", "Longitude"]
TEST_SIZE = 0.2
RANDOM_SEED = 42
N_CLUSTERS = 6

KMEANS_PATH = MODELS_DIR / "k-means_default_42.sav"
TREE_PATH = MODELS_DIR / "decision_tree_classifier_default_42.sav"
TRAIN_CLUSTERS_PNG = FIGURES / "kmeans_train_clusters.png"
TEST_OVERLAY_PNG = FIGURES / "kmeans_test_overlay.png"
CONFUSION_PNG = FIGURES / "decision_tree_confusion.png"
TREE_PNG = FIGURES / "decision_tree.png"


def load_housing(csv_path: Path = RAW_CSV, url: str = DATASET_URL) -> pd.DataFrame:
    """Load the California Housing dataset from disk or the tutorial URL."""
    if csv_path.exists():
        print(f"Loading dataset from {csv_path}")
        data = pd.read_csv(csv_path)
    else:
        print(f"Local file missing — loading from {url}")
        data = pd.read_csv(url)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv_path, index=False)
        print(f"Saved a local copy to {csv_path}")
    return data


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only MedInc, Latitude, and Longitude (tutorial Step 1)."""
    missing = [c for c in FEATURE_COLS if c not in data.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    return data[FEATURE_COLS].copy()


def split_train_test(
    features: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test for unsupervised fit + later cluster prediction."""
    X_train, X_test = train_test_split(
        features,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True)


def save_splits(X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: Path = PROCESSED) -> None:
    """Persist train/test feature tables under data/processed/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "housing_train.csv"
    test_path = out_dir / "housing_test.csv"
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)
    print(f"Saved train split: {train_path} ({len(X_train)} rows)")
    print(f"Saved test split:  {test_path} ({len(X_test)} rows)")


def step1_load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Step 1 of the tutorial and return train/test feature frames."""
    print("=== Step 1: Loading the dataset ===")
    total_data = load_housing()
    print(f"Full dataset shape: {total_data.shape}")
    print(f"Columns: {list(total_data.columns)}")
    print(total_data.head())

    print("\nKeeping only:", FEATURE_COLS)
    X = select_features(total_data)
    print(X.describe().round(3))

    X_train, X_test = split_train_test(X)
    print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")
    print("Train head:")
    print(X_train.head())

    save_splits(X_train, X_test)
    return X_train, X_test


def plot_cluster_scatter(
    data: pd.DataFrame,
    out_path: Path,
    title: str,
    alpha: float = 0.7,
    marker: str = "o",
) -> None:
    """Three scatter views colored by cluster label."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)
    pairs = [
        ("Latitude", "Longitude"),
        ("Latitude", "MedInc"),
        ("Longitude", "MedInc"),
    ]
    for ax, (x_col, y_col) in zip(axes, pairs):
        sns.scatterplot(
            ax=ax,
            data=data,
            x=x_col,
            y=y_col,
            hue="cluster",
            palette="deep",
            alpha=alpha,
            marker=marker,
        )
        ax.set_title(f"{y_col} vs {x_col}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def step2_build_kmeans(X_train: pd.DataFrame) -> tuple[KMeans, pd.DataFrame, list[int]]:
    """Fit K-Means with 6 clusters and attach labels to the train set."""
    print("\n=== Step 2: Build a K-Means ===")
    model_unsup = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=RANDOM_SEED)
    model_unsup.fit(X_train[FEATURE_COLS])

    y_train = list(model_unsup.labels_)
    X_train = X_train.copy()
    X_train["cluster"] = y_train
    # Categorical-friendly dtype for plotting / reporting
    X_train["cluster"] = X_train["cluster"].astype("category")

    print("Cluster counts (train):")
    print(X_train["cluster"].value_counts().sort_index())
    print(X_train.head())

    plot_cluster_scatter(
        X_train,
        TRAIN_CLUSTERS_PNG,
        title="Train houses colored by K-Means cluster",
    )
    print(
        "Description: clusters separate California geography (lat/long) and "
        "income bands — coastal / inland regions and higher-MedInc pockets "
        "tend to form distinct groups."
    )
    return model_unsup, X_train, y_train


def step3_predict_test(
    model_unsup: KMeans,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    """Predict cluster for test houses and overlay them on the train plot."""
    print("\n=== Step 3: Predict with the test set ===")
    y_test = list(model_unsup.predict(X_test[FEATURE_COLS]))
    X_test = X_test.copy()
    X_test["cluster"] = y_test
    X_test["cluster"] = X_test["cluster"].astype("category")
    print("Cluster counts (test):")
    print(X_test["cluster"].value_counts().sort_index())
    print(X_test.head())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Test predictions (+) over train clusters", fontsize=14)
    pairs = [
        ("Latitude", "Longitude"),
        ("Latitude", "MedInc"),
        ("Longitude", "MedInc"),
    ]
    for ax, (x_col, y_col) in zip(axes, pairs):
        sns.scatterplot(
            ax=ax,
            data=X_train,
            x=x_col,
            y=y_col,
            hue="cluster",
            palette="deep",
            alpha=0.15,
        )
        sns.scatterplot(
            ax=ax,
            data=X_test,
            x=x_col,
            y=y_col,
            hue="cluster",
            palette="deep",
            marker="+",
            s=60,
            legend=False,
        )
        ax.set_title(f"{y_col} vs {x_col}")
        ax.legend([], [], frameon=False)
    fig.tight_layout()
    TEST_OVERLAY_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TEST_OVERLAY_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {TEST_OVERLAY_PNG}")
    print(
        "Test points (+) land inside the same geographic/income regions as "
        "the train clusters — the K-Means assignment generalizes well."
    )
    return X_test, y_test


def step4_supervised_model(
    X_train: pd.DataFrame,
    y_train: list[int],
    X_test: pd.DataFrame,
    y_test: list[int],
) -> DecisionTreeClassifier:
    """Train a Decision Tree to recover K-Means labels from the 3 features."""
    print("\n=== Step 4: Train a supervised classification model ===")
    print(
        "Why DecisionTreeClassifier?\n"
        "  - Cluster boundaries here are geographic + income (non-linear),\n"
        "    so a tree fits axis-aligned regions without assuming linearity.\n"
        "  - Features are on different scales (income vs lat/long); trees do\n"
        "    not require standardization.\n"
        "  - Easy to inspect which splits recover the unsupervised labels.\n"
        "This is the usual unlabeled-data flow: K-Means creates labels,\n"
        "then a supervised model learns to predict them for new points."
    )

    # Fit on features only (do not leak the cluster column into X)
    model_sup = DecisionTreeClassifier(random_state=RANDOM_SEED)
    model_sup.fit(X_train[FEATURE_COLS], y_train)

    y_pred = model_sup.predict(X_test[FEATURE_COLS])
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nDecision Tree accuracy vs K-Means test labels: {acc:.2%}")
    print(f"Tree depth={model_sup.get_depth()}, leaves={model_sup.get_n_leaves()}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (rows=true K-Means cluster, cols=tree prediction):")
    print(cm)

    # Confusion-matrix heatmap
    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
        xticklabels=[str(i) for i in range(N_CLUSTERS)],
        yticklabels=[str(i) for i in range(N_CLUSTERS)],
    )
    ax_cm.set_xlabel("Predicted cluster")
    ax_cm.set_ylabel("K-Means (true) cluster")
    ax_cm.set_title(f"Decision Tree vs K-Means labels (acc={acc:.2%})")
    fig_cm.tight_layout()
    CONFUSION_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig_cm.savefig(CONFUSION_PNG, dpi=120)
    plt.close(fig_cm)
    print(f"Saved figure: {CONFUSION_PNG}")

    # Tree overview (depth truncated for readability)
    fig = plt.figure(figsize=(18, 12))
    tree.plot_tree(
        model_sup,
        feature_names=FEATURE_COLS,
        class_names=[str(i) for i in range(N_CLUSTERS)],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    fig.suptitle("Decision Tree (depth truncated for display)", fontsize=14)
    fig.tight_layout()
    TREE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TREE_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {TREE_PNG}")

    print(
        "\nWhat we see:\n"
        f"  - Accuracy ~{acc:.1%} means the tree almost perfectly recovers the\n"
        "    K-Means categorization from MedInc/Latitude/Longitude alone.\n"
        "  - Most errors are on the smallest cluster (rare high-income pockets),\n"
        "    which is expected with fewer support samples.\n"
        "  - So unsupervised labeling + a simple supervised model works well\n"
        "    for assigning new houses to the same house-group system."
    )
    return model_sup


def step5_save_models(model_unsup: KMeans, model_sup: DecisionTreeClassifier) -> None:
    """Store both models in models/ (tutorial Step 5)."""
    print("\n=== Step 5: Save the models ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_unsup, KMEANS_PATH)
    joblib.dump(model_sup, TREE_PATH)
    print(f"Saved unsupervised model: {KMEANS_PATH}")
    print(f"Saved supervised model:   {TREE_PATH}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test = step1_load_and_split()
    model_unsup, X_train, y_train = step2_build_kmeans(X_train)
    X_test, y_test = step3_predict_test(model_unsup, X_train, X_test)
    model_sup = step4_supervised_model(X_train, y_train, X_test, y_test)
    step5_save_models(model_unsup, model_sup)

    # Persist labeled splits for inspection
    PROCESSED.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED / "housing_train_clustered.csv", index=False)
    X_test.to_csv(PROCESSED / "housing_test_clustered.csv", index=False)
    print("\nAll tutorial steps (1–5) complete.")


if __name__ == "__main__":
    main()
