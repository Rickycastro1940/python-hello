"""Spam/ham classification: Naive Bayes, then stronger alternatives.

Why other models can beat Naive Bayes on text
---------------------------------------------
Naive Bayes assumes feature independence and models P(x|y). That is fast and
works well on small bag-of-words data, but it is often beaten by discriminative
linear models that learn P(y|x) directly on high-dimensional TF-IDF features:

- Logistic Regression: strong, calibrated text baseline; regularized linear
  decision boundary often generalizes better than NB on TF-IDF.
- Linear SVM (LinearSVC): max-margin classifier; classic top performer for
  sparse high-dimensional text.
- Random Forest: can model non-linear word interactions, but usually needs
  more samples than we have here and is weaker on very sparse text.
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

DATA_PATH = Path(__file__).resolve().parent / "data.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PREFERRED_NB = "MultinomialNB"


class DenseTfidfVectorizer(TfidfVectorizer):
    """TF-IDF that returns dense arrays for GaussianNB."""

    def transform(self, raw_documents):
        return super().transform(raw_documents).toarray()

    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(raw_documents, y).toarray()


def load_dataframe() -> pd.DataFrame:
    """Load data.csv when present."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from {DATA_PATH.name} ({len(df)} rows)")
    return df


def evaluate_model(name, model, X_train, X_test, y_train, y_test, labels):
    """Fit a model and print hold-out metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Weighted F1: {f1:.2%}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=labels,
            columns=labels,
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
    """Models studied as stronger text baselines than Naive Bayes."""
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
        "pipeline": pipeline,
    }


def cross_validate_nb(texts: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Rank the three Naive Bayes implementations with stratified CV."""
    rows = []
    print("\n===== Naive Bayes: Stratified 5-Fold CV =====")
    for name, pipeline in nb_pipelines().items():
        row = score_pipeline(name, pipeline, texts, y)
        rows.append({k: v for k, v in row.items() if k != "pipeline"})
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
        "    the first model that beats MultinomialNB on TF-IDF spam/ham tasks.\n"
        "  • LinearSVC — max-margin separator in sparse high-dimensional space;\n"
        "    a classic top text classifier in applied ML courses.\n"
        "  • RandomForest — captures non-linear interactions, but is data-hungry\n"
        "    and often lags linear models on tiny bag-of-words datasets.\n"
        "We train all three and compare them with the best Naive Bayes model."
    )


def train_alternatives(texts, y, texts_train, texts_test, y_train, y_test, labels):
    """Train Logistic Regression, Linear SVM, and Random Forest."""
    print_alternatives_argument()
    print("\n===== Alternatives: Stratified 5-Fold CV =====")

    cv_rows = []
    holdout_rows = []
    fitted = {}

    for name, pipeline in alternative_pipelines().items():
        cv_row = score_pipeline(name, pipeline, texts, y)
        cv_rows.append({k: v for k, v in cv_row.items() if k != "pipeline"})

        holdout = evaluate_model(
            name,
            pipeline,
            texts_train,
            texts_test,
            y_train,
            y_test,
            labels,
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


def save_confusion_matrix(y_true, y_pred, labels, title, path):
    """Save a seaborn heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved plot: {path}")


def build_final_pipeline(model_name: str, estimator) -> Pipeline:
    """Return a text→label pipeline for the winning model."""
    # Pipelines from alternatives / tuned RF are already complete.
    if isinstance(estimator, Pipeline):
        return estimator

    builders = {
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
        "GaussianNB": Pipeline(
            [
                ("tfidf", DenseTfidfVectorizer(stop_words="english")),
                ("clf", GaussianNB()),
            ]
        ),
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model name: {model_name}")
    return builders[model_name]


def save_model(pipeline: Pipeline, model_name: str, metrics: dict) -> Path:
    """Persist the fitted pipeline and metadata under models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    return MODEL_PATH


def main() -> None:
    df = load_dataframe()
    labels = sorted(df["label"].unique())
    texts = df["text"]
    y = df["label"]

    print("\nStep 1 — Train GaussianNB, MultinomialNB, and BernoulliNB")

    count_vec = CountVectorizer(stop_words="english")
    binary_vec = CountVectorizer(stop_words="english", binary=True)
    tfidf_vec = TfidfVectorizer(stop_words="english")

    X_counts = count_vec.fit_transform(texts)
    X_binary = binary_vec.fit_transform(texts)
    X_tfidf = tfidf_vec.fit_transform(texts).toarray()

    idx_train, idx_test = train_test_split(
        np.arange(len(y)),
        test_size=0.3,
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
            labels,
        ),
        evaluate_model(
            "MultinomialNB",
            MultinomialNB(),
            X_counts[idx_train],
            X_counts[idx_test],
            y_train,
            y_test,
            labels,
        ),
        evaluate_model(
            "BernoulliNB",
            BernoulliNB(),
            X_binary[idx_train],
            X_binary[idx_test],
            y_train,
            y_test,
            labels,
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

    # Step 2 — alternatives that can overcome NB
    alt_cv, alt_holdout, alt_fitted = train_alternatives(
        texts, y, texts_train, texts_test, y_train, y_test, labels
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
        labels,
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
    # Prefer discriminative alternatives when CV scores tie the best NB.
    comparison["is_alternative"] = comparison["model"] != best_nb_name
    comparison = comparison.sort_values(
        ["cv_accuracy", "cv_f1", "is_alternative", "holdout_accuracy"],
        ascending=[False, False, False, False],
    )
    print("\n===== Final Comparison vs Best Naive Bayes =====")
    print(
        comparison.drop(columns=["is_alternative"]).to_string(index=False)
    )

    final_winner = comparison.iloc[0]["model"]
    print(f"\nFinal model: {final_winner}")

    if final_winner == best_nb_name:
        print(
            f"{best_nb_name} still leads on this small dataset. Linear models "
            "are the most promising challengers as more labeled text is added."
        )
        final_preds = best_nb_holdout["y_pred"]
        saved_estimator = best_nb_holdout["model"]
        winner_metrics = {
            "cv_accuracy": float(best_nb_cv["cv_accuracy_mean"]),
            "cv_f1": float(best_nb_cv["cv_f1_mean"]),
            "holdout_accuracy": float(best_nb_holdout["accuracy"]),
            "holdout_f1": float(best_nb_holdout["f1"]),
        }
    elif final_winner == "RandomForestTuned":
        print("Tuned Random Forest overcame the best Naive Bayes result.")
        final_preds = rf_holdout["y_pred"]
        saved_estimator = rf_tuned["estimator"]
        winner_metrics = {
            "cv_accuracy": float(rf_tuned["cv_accuracy_mean"]),
            "cv_f1": float(rf_tuned["cv_f1_mean"]),
            "holdout_accuracy": float(rf_holdout["accuracy"]),
            "holdout_f1": float(rf_holdout["f1"]),
        }
    else:
        nb_acc = float(best_nb_cv["cv_accuracy_mean"])
        alt_acc = float(comparison.iloc[0]["cv_accuracy"])
        if np.isclose(alt_acc, nb_acc):
            print(
                f"{final_winner} matches {best_nb_name} on CV and is selected "
                "as the preferred discriminative alternative for text."
            )
        else:
            print(
                f"{final_winner} overcame Naive Bayes — as expected for a "
                "discriminative linear text model on TF-IDF features."
            )
        hold = next(h for h in alt_holdout if h["model_name"] == final_winner)
        final_preds = hold["y_pred"]
        saved_estimator = alt_fitted[final_winner]
        winner_row = comparison.iloc[0]
        winner_metrics = {
            "cv_accuracy": float(winner_row["cv_accuracy"]),
            "cv_f1": float(winner_row["cv_f1"]),
            "holdout_accuracy": float(winner_row["holdout_accuracy"]),
            "holdout_f1": float(winner_row["holdout_f1"]),
        }

    # Always refit a fresh deployable pipeline on the full dataset before saving.
    if final_winner == "RandomForestTuned":
        final_pipeline = rf_tuned["estimator"]
    elif final_winner in alternative_pipelines():
        final_pipeline = alternative_pipelines()[final_winner]
    else:
        final_pipeline = build_final_pipeline(final_winner, saved_estimator)
    final_pipeline.fit(texts, y)

    save_model(final_pipeline, final_winner, winner_metrics)
    save_confusion_matrix(
        y_test,
        final_preds,
        labels,
        f"Final model: {final_winner}",
        "confusion_matrix.png",
    )

    demos = [
        "Congratulations! Claim your FREE prize now",
        "Are we still meeting for coffee later?",
    ]
    loaded = joblib.load(MODEL_PATH)
    demo_preds = loaded["pipeline"].predict(demos)
    print(f"\n--- Demo Predictions from saved model ({loaded['model_name']}) ---")
    for text, label in zip(demos, demo_preds):
        print(f"[{label}] {text}")


if __name__ == "__main__":
    main()
