"""Google Play review sentiment with Naive Bayes + stronger alternatives.

Dataset: playstore_reviews.csv (4Geeks Naive Bayes project tutorial)
https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv

Columns:
  - package_name: app id
  - review: review text
  - polarity: 0 = negative, 1 = positive

Why alternatives can beat Naive Bayes on text
---------------------------------------------
Naive Bayes assumes feature independence. Discriminative models that learn
P(y|x) on TF-IDF often do better:
  - Logistic Regression: strong regularized text baseline
  - Linear SVM: max-margin classifier for sparse high-dimensional text
  - Random Forest: non-linear, but usually needs more data / is weaker on sparse text
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = Path(__file__).resolve().parent / "playstore_reviews.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/"
    "naive-bayes-project-tutorial/main/playstore_reviews.csv"
)
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "playstore_sentiment_model.joblib"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PREFERRED_NB = "MultinomialNB"
LABEL_NAMES = ["negative", "positive"]


class DenseTfidfVectorizer(TfidfVectorizer):
    """TF-IDF that returns dense arrays for GaussianNB."""

    def transform(self, raw_documents):
        return super().transform(raw_documents).toarray()

    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(raw_documents, y).toarray()


def load_dataframe() -> pd.DataFrame:
    """Load playstore_reviews.csv; download it if the local file is missing."""
    if not DATA_PATH.exists():
        print(f"{DATA_PATH.name} missing — downloading from tutorial URL")
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    expected = {"package_name", "review", "polarity"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Only 3 raw variables: package_name, review (predictors) and polarity (label).
    # Sentiment depends on comment content, not which app it came from, so drop
    # package_name and keep review as the sole predictor.
    df = df.drop(columns=["package_name"]).copy()
    df["review"] = df["review"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["review", "polarity"])
    df["polarity"] = df["polarity"].astype(int)

    print(
        f"Loaded {DATA_PATH.name}: {len(df)} reviews "
        f"(neg={int((df['polarity'] == 0).sum())}, "
        f"pos={int((df['polarity'] == 1).sum())})"
    )
    print(
        "Dropped package_name — using review text only to predict polarity."
    )
    return df


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Fit a model and print hold-out metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Weighted F1: {f1:.2%}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=[0, 1]),
            index=LABEL_NAMES,
            columns=LABEL_NAMES,
        )
    )
    return {
        "model_name": name,
        "model": model,
        "accuracy": accuracy,
        "f1": f1,
        "y_pred": y_pred,
    }


def nb_pipelines() -> dict[str, Pipeline]:
    """Naive Bayes candidates with feature types matched to each algorithm."""
    return {
        "GaussianNB": Pipeline(
            [
                ("tfidf", DenseTfidfVectorizer(stop_words="english")),
                ("clf", GaussianNB()),
            ]
        ),
        "MultinomialNB": Pipeline(
            [
                ("counts", CountVectorizer(stop_words="english")),
                ("clf", MultinomialNB()),
            ]
        ),
        "BernoulliNB": Pipeline(
            [
                ("binary", CountVectorizer(stop_words="english", binary=True)),
                ("clf", BernoulliNB()),
            ]
        ),
    }


def alternative_pipelines() -> dict[str, Pipeline]:
    """Models that can overcome Naive Bayes on text classification."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=42,
                        dual="auto",
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def score_pipeline(name: str, pipeline: Pipeline, texts, y) -> dict:
    """Run stratified CV for one pipeline."""
    acc_scores = cross_val_score(pipeline, texts, y, cv=CV, scoring="accuracy")
    f1_scores = cross_val_score(
        pipeline, texts, y, cv=CV, scoring="f1_weighted"
    )
    print(
        f"{name}: accuracy={acc_scores.mean():.2%} ± {acc_scores.std():.2%}, "
        f"weighted F1={f1_scores.mean():.2%} ± {f1_scores.std():.2%}"
    )
    return {
        "model": name,
        "cv_accuracy_mean": acc_scores.mean(),
        "cv_accuracy_std": acc_scores.std(),
        "cv_f1_mean": f1_scores.mean(),
        "cv_f1_std": f1_scores.std(),
    }


def cross_validate_nb(texts: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Rank the three Naive Bayes implementations with stratified CV."""
    rows = []
    print("\n===== Naive Bayes: Stratified 5-Fold CV =====")
    for name, pipeline in nb_pipelines().items():
        rows.append(score_pipeline(name, pipeline, texts, y))
    return pd.DataFrame(rows).sort_values(
        ["cv_accuracy_mean", "cv_f1_mean"], ascending=False
    )


def select_best_nb(cv_summary: pd.DataFrame) -> str:
    """Pick the best NB; prefer MultinomialNB on ties."""
    best_score = cv_summary.iloc[0]["cv_accuracy_mean"]
    tied = cv_summary.loc[
        np.isclose(cv_summary["cv_accuracy_mean"], best_score), "model"
    ].tolist()
    if PREFERRED_NB in tied:
        return PREFERRED_NB
    return tied[0]


def print_alternatives_argument() -> None:
    """Explain which non-NB models can beat Naive Bayes and why."""
    print("\n===== Why explore alternatives to Naive Bayes? =====")
    print(
        "Naive Bayes is a strong generative baseline, but it assumes word\n"
        "independence. Discriminative models often overcome that limit on text:\n"
        "  • LogisticRegression — learns P(y|x) with L2 regularization; usually\n"
        "    the first model that beats MultinomialNB on TF-IDF review tasks.\n"
        "  • LinearSVC — max-margin separator in sparse high-dimensional space;\n"
        "    a classic top text classifier in applied ML courses.\n"
        "  • RandomForest — captures non-linear interactions, but is data-hungry\n"
        "    and often lags linear models on bag-of-words datasets.\n"
        "We train all three and compare them with the best Naive Bayes model."
    )


def train_alternatives(texts, y, texts_train, texts_test, y_train, y_test):
    """Train Logistic Regression, Linear SVM, and Random Forest."""
    print_alternatives_argument()
    print("\n===== Alternatives: Stratified 5-Fold CV =====")

    cv_rows = []
    holdout_rows = []
    fitted = {}

    for name, pipeline in alternative_pipelines().items():
        cv_rows.append(score_pipeline(name, pipeline, texts, y))
        holdout = evaluate_model(
            name,
            pipeline,
            texts_train,
            texts_test,
            y_train,
            y_test,
        )
        holdout_rows.append(holdout)
        fitted[name] = holdout["model"]

    cv_df = pd.DataFrame(cv_rows).sort_values(
        ["cv_accuracy_mean", "cv_f1_mean"], ascending=False
    )
    return cv_df, holdout_rows, fitted


def optimize_random_forest(texts: pd.Series, y: pd.Series) -> dict:
    """Optional extra tuning pass for Random Forest."""
    print("\n===== Random Forest fine-tuning (RandomizedSearchCV) =====")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            (
                "clf",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "tfidf__min_df": [1, 2],
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 3, 5],
            "clf__max_features": ["sqrt", "log2"],
        },
        n_iter=12,
        scoring="f1_weighted",
        cv=CV,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(texts, y)
    acc_scores = cross_val_score(
        search.best_estimator_, texts, y, cv=CV, scoring="accuracy"
    )
    f1_scores = cross_val_score(
        search.best_estimator_, texts, y, cv=CV, scoring="f1_weighted"
    )
    print(f"Best RF params: {search.best_params_}")
    print(
        f"Tuned RandomForest: accuracy={acc_scores.mean():.2%} ± "
        f"{acc_scores.std():.2%}, weighted F1={f1_scores.mean():.2%} ± "
        f"{f1_scores.std():.2%}"
    )
    return {
        "model_name": "RandomForestTuned",
        "estimator": search.best_estimator_,
        "cv_accuracy_mean": acc_scores.mean(),
        "cv_accuracy_std": acc_scores.std(),
        "cv_f1_mean": f1_scores.mean(),
        "cv_f1_std": f1_scores.std(),
    }


def save_confusion_matrix(y_true, y_pred, title, path):
    """Save a seaborn heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved plot: {path}")


def build_nb_pipeline(model_name: str) -> Pipeline:
    """Build an unfitted NB text pipeline."""
    return nb_pipelines()[model_name]


def save_model(pipeline: Pipeline, model_name: str, metrics: dict) -> Path:
    """Persist the fitted pipeline and metadata under models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
        "dataset": DATA_PATH.name,
        "label_map": {0: "negative", 1: "positive"},
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    return MODEL_PATH


def main() -> None:
    df = load_dataframe()
    texts = df["review"]
    y = df["polarity"]

    print("\nStep 1 — Train GaussianNB, MultinomialNB, and BernoulliNB")
    print(f"Dataset source: {DATA_URL}")

    count_vec = CountVectorizer(stop_words="english")
    binary_vec = CountVectorizer(stop_words="english", binary=True)
    tfidf_vec = TfidfVectorizer(stop_words="english")

    X_counts = count_vec.fit_transform(texts)
    X_binary = binary_vec.fit_transform(texts)
    X_tfidf = tfidf_vec.fit_transform(texts).toarray()

    idx_train, idx_test = train_test_split(
        np.arange(len(y)),
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    texts_train, texts_test = texts.iloc[idx_train], texts.iloc[idx_test]
    print(f"Train / test sizes: {len(y_train)} / {len(y_test)}")

    nb_holdout = [
        evaluate_model(
            "GaussianNB",
            GaussianNB(),
            X_tfidf[idx_train],
            X_tfidf[idx_test],
            y_train,
            y_test,
        ),
        evaluate_model(
            "MultinomialNB",
            MultinomialNB(),
            X_counts[idx_train],
            X_counts[idx_test],
            y_train,
            y_test,
        ),
        evaluate_model(
            "BernoulliNB",
            BernoulliNB(),
            X_binary[idx_train],
            X_binary[idx_test],
            y_train,
            y_test,
        ),
    ]

    print("\n===== Hold-out Split Summary (Naive Bayes) =====")
    print(
        pd.DataFrame(
            [
                {
                    "model": r["model_name"],
                    "holdout_accuracy": r["accuracy"],
                    "holdout_weighted_f1": r["f1"],
                }
                for r in nb_holdout
            ]
        )
        .sort_values(
            ["holdout_accuracy", "holdout_weighted_f1"], ascending=False
        )
        .to_string(index=False)
    )

    cv_summary = cross_validate_nb(texts, y)
    print("\n===== Cross-Validation Ranking (Naive Bayes) =====")
    print(cv_summary.to_string(index=False))

    best_nb_name = select_best_nb(cv_summary)
    best_nb_cv = cv_summary.loc[cv_summary["model"] == best_nb_name].iloc[0]
    best_nb_holdout = next(
        r for r in nb_holdout if r["model_name"] == best_nb_name
    )
    print(f"\nBest Naive Bayes: {best_nb_name}")

    alt_cv, alt_holdout, alt_fitted = train_alternatives(
        texts, y, texts_train, texts_test, y_train, y_test
    )
    print("\n===== Alternatives CV Ranking =====")
    print(alt_cv.to_string(index=False))

    rf_tuned = optimize_random_forest(texts, y)
    rf_holdout = evaluate_model(
        "RandomForestTuned",
        rf_tuned["estimator"],
        texts_train,
        texts_test,
        y_train,
        y_test,
    )

    comparison_rows = [
        {
            "model": best_nb_name,
            "cv_accuracy": best_nb_cv["cv_accuracy_mean"],
            "cv_f1": best_nb_cv["cv_f1_mean"],
            "holdout_accuracy": best_nb_holdout["accuracy"],
            "holdout_f1": best_nb_holdout["f1"],
        }
    ]
    for row in alt_cv.itertuples(index=False):
        hold = next(h for h in alt_holdout if h["model_name"] == row.model)
        comparison_rows.append(
            {
                "model": row.model,
                "cv_accuracy": row.cv_accuracy_mean,
                "cv_f1": row.cv_f1_mean,
                "holdout_accuracy": hold["accuracy"],
                "holdout_f1": hold["f1"],
            }
        )
    comparison_rows.append(
        {
            "model": "RandomForestTuned",
            "cv_accuracy": rf_tuned["cv_accuracy_mean"],
            "cv_f1": rf_tuned["cv_f1_mean"],
            "holdout_accuracy": rf_holdout["accuracy"],
            "holdout_f1": rf_holdout["f1"],
        }
    )

    comparison = pd.DataFrame(comparison_rows)
    comparison["is_alternative"] = comparison["model"] != best_nb_name
    comparison = comparison.sort_values(
        ["cv_accuracy", "cv_f1", "is_alternative", "holdout_accuracy"],
        ascending=[False, False, False, False],
    )
    print("\n===== Final Comparison vs Best Naive Bayes =====")
    print(comparison.drop(columns=["is_alternative"]).to_string(index=False))

    final_winner = comparison.iloc[0]["model"]
    print(f"\nFinal model: {final_winner}")

    if final_winner == best_nb_name:
        print(f"{best_nb_name} remains the strongest model on this dataset.")
        final_preds = best_nb_holdout["y_pred"]
        winner_metrics = {
            "cv_accuracy": float(best_nb_cv["cv_accuracy_mean"]),
            "cv_f1": float(best_nb_cv["cv_f1_mean"]),
            "holdout_accuracy": float(best_nb_holdout["accuracy"]),
            "holdout_f1": float(best_nb_holdout["f1"]),
        }
    elif final_winner == "RandomForestTuned":
        print("Tuned Random Forest overcame the best Naive Bayes result.")
        final_preds = rf_holdout["y_pred"]
        winner_metrics = {
            "cv_accuracy": float(rf_tuned["cv_accuracy_mean"]),
            "cv_f1": float(rf_tuned["cv_f1_mean"]),
            "holdout_accuracy": float(rf_holdout["accuracy"]),
            "holdout_f1": float(rf_holdout["f1"]),
        }
    else:
        nb_acc = float(best_nb_cv["cv_accuracy_mean"])
        alt_acc = float(comparison.iloc[0]["cv_accuracy"])
        if alt_acc > nb_acc and not np.isclose(alt_acc, nb_acc):
            print(
                f"{final_winner} overcame Naive Bayes on playstore review sentiment."
            )
        else:
            print(
                f"{final_winner} matches or leads and is selected as the "
                "preferred discriminative alternative."
            )
        hold = next(h for h in alt_holdout if h["model_name"] == final_winner)
        final_preds = hold["y_pred"]
        winner_row = comparison.iloc[0]
        winner_metrics = {
            "cv_accuracy": float(winner_row["cv_accuracy"]),
            "cv_f1": float(winner_row["cv_f1"]),
            "holdout_accuracy": float(winner_row["holdout_accuracy"]),
            "holdout_f1": float(winner_row["holdout_f1"]),
        }

    if final_winner == "RandomForestTuned":
        final_pipeline = rf_tuned["estimator"]
    elif final_winner in alternative_pipelines():
        final_pipeline = alternative_pipelines()[final_winner]
    else:
        final_pipeline = build_nb_pipeline(final_winner)
    final_pipeline.fit(texts, y)

    save_model(final_pipeline, final_winner, winner_metrics)
    save_confusion_matrix(
        y_test,
        final_preds,
        f"Final model: {final_winner}",
        "confusion_matrix.png",
    )

    demos = [
        "this app is amazing and so easy to use love it",
        "terrible update crashes all the time waste of space",
    ]
    loaded = joblib.load(MODEL_PATH)
    demo_preds = loaded["pipeline"].predict(demos)
    label_map = loaded["label_map"]
    print(f"\n--- Demo Predictions from saved model ({loaded['model_name']}) ---")
    for text, pred in zip(demos, demo_preds):
        print(f"[{label_map[int(pred)]}] {text}")


if __name__ == "__main__":
    main()
