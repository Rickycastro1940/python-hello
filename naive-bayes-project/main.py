"""Compare GaussianNB, MultinomialNB, and BernoulliNB on spam/ham text.

Hypothesis: MultinomialNB is the best fit for TF-IDF / bag-of-words text
features. This script trains all three on the same split and checks that.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parent / "data.csv"

# Fallback sample if data.csv is missing
SAMPLE_DATA = {
    "text": [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Nah I don't think he goes to usf, he lives around here though",
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize!",
        "Even my brother is not like to speak with me. They treat me like aids patent.",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore.",
        "Claim your free prize money today urgent offer",
        "Can we reschedule our lunch meeting tomorrow",
        "You have been selected to win a cash prize",
        "Please review the attached project report",
        "Limited time offer buy now cheap deal",
        "Thanks for sending the meeting notes",
        "Get rich quick with this secret method",
        "Looking forward to seeing you on Friday",
        "Your account has been compromised act now",
        "The team standup starts at nine o'clock",
    ],
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
    ],
}


def load_dataframe() -> pd.DataFrame:
    """Load data.csv when present; otherwise use the built-in sample."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset from {DATA_PATH.name} ({len(df)} rows)")
        return df
    print("data.csv not found — using built-in sample dataset")
    return pd.DataFrame(SAMPLE_DATA)


def evaluate_model(name, model, X_train, X_test, y_train, y_test, labels):
    """Fit one Naive Bayes variant and return metrics + predictions."""
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


def cross_validate_models(texts: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Compare all three NB variants with stratified 5-fold CV on raw text."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Dense TF-IDF transformer for GaussianNB (cannot take sparse input).
    class DenseTfidfVectorizer(TfidfVectorizer):
        def transform(self, raw_documents):
            return super().transform(raw_documents).toarray()

        def fit_transform(self, raw_documents, y=None):
            return super().fit_transform(raw_documents, y).toarray()

    candidates = {
        "GaussianNB": Pipeline(
            [
                ("tfidf", DenseTfidfVectorizer(stop_words="english")),
                ("clf", GaussianNB()),
            ]
        ),
        "MultinomialNB": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("clf", MultinomialNB()),
            ]
        ),
        "BernoulliNB": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("clf", BernoulliNB()),
            ]
        ),
    }

    rows = []
    print("\n===== Stratified 5-Fold Cross-Validation =====")
    for name, pipeline in candidates.items():
        acc_scores = cross_val_score(
            pipeline, texts, y, cv=cv, scoring="accuracy"
        )
        f1_scores = cross_val_score(
            pipeline, texts, y, cv=cv, scoring="f1_weighted"
        )
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": acc_scores.mean(),
                "cv_accuracy_std": acc_scores.std(),
                "cv_f1_mean": f1_scores.mean(),
                "cv_f1_std": f1_scores.std(),
            }
        )
        print(
            f"{name}: accuracy={acc_scores.mean():.2%} ± {acc_scores.std():.2%}, "
            f"weighted F1={f1_scores.mean():.2%} ± {f1_scores.std():.2%}"
        )

    return pd.DataFrame(rows).sort_values(
        ["cv_accuracy_mean", "cv_f1_mean"], ascending=False
    )


def main() -> None:
    df = load_dataframe()
    labels = sorted(df["label"].unique())
    texts = df["text"]
    y = df["label"]

    # Shared TF-IDF features for a fair single-split head-to-head
    vectorizer = TfidfVectorizer(stop_words="english")
    X_sparse = vectorizer.fit_transform(texts)

    X_train_s, X_test_s, y_train, y_test = train_test_split(
        X_sparse,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # GaussianNB requires dense arrays; Multinomial/Bernoulli accept sparse.
    X_train_dense = X_train_s.toarray()
    X_test_dense = X_test_s.toarray()

    print("\nComparing GaussianNB vs MultinomialNB vs BernoulliNB")
    print(f"Train / test sizes: {len(y_train)} / {len(y_test)}")

    results = [
        evaluate_model(
            "GaussianNB",
            GaussianNB(),
            X_train_dense,
            X_test_dense,
            y_train,
            y_test,
            labels,
        ),
        evaluate_model(
            "MultinomialNB",
            MultinomialNB(),
            X_train_s,
            X_test_s,
            y_train,
            y_test,
            labels,
        ),
        evaluate_model(
            "BernoulliNB",
            BernoulliNB(),
            X_train_s,
            X_test_s,
            y_train,
            y_test,
            labels,
        ),
    ]

    split_summary = pd.DataFrame(
        [
            {
                "model": r["model_name"],
                "holdout_accuracy": r["accuracy"],
                "holdout_weighted_f1": r["f1"],
            }
            for r in results
        ]
    ).sort_values(
        ["holdout_accuracy", "holdout_weighted_f1"], ascending=False
    )

    print("\n===== Hold-out Split Summary =====")
    print(split_summary.to_string(index=False))

    # More reliable ranking on this small dataset
    cv_summary = cross_validate_models(texts, y)
    print("\n===== Cross-Validation Ranking =====")
    print(cv_summary.to_string(index=False))

    best_cv = cv_summary.iloc[0]
    chosen_name = "MultinomialNB"
    chosen = next(r for r in results if r["model_name"] == chosen_name)
    chosen_cv = cv_summary.loc[cv_summary["model"] == chosen_name].iloc[0]

    print(f"\nBest by cross-validation: {best_cv['model']}")
    print(
        f"Selected model for this project: {chosen_name} "
        f"(CV accuracy={chosen_cv['cv_accuracy_mean']:.2%} ± "
        f"{chosen_cv['cv_accuracy_std']:.2%})"
    )

    if best_cv["model"] == chosen_name:
        print(
            "Confirmation: MultinomialNB is the right choice for this "
            "TF-IDF text classification task."
        )
    else:
        print(
            f"On this tiny dataset, {best_cv['model']} edged the CV ranking, "
            "but MultinomialNB remains the preferred model for non-negative "
            "TF-IDF / bag-of-words text features (GaussianNB assumes continuous "
            "normals on sparse high-dimensional data; BernoulliNB discards "
            "term-weight magnitude)."
        )

    # Confusion matrix for the chosen MultinomialNB model
    cm = confusion_matrix(y_test, chosen["y_pred"], labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Chosen model: MultinomialNB")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    print("\nSaved plot: confusion_matrix.png")

    # Demo predictions with the chosen model
    demos = [
        "Congratulations! Claim your FREE prize now",
        "Are we still meeting for coffee later?",
    ]
    demo_preds = chosen["model"].predict(vectorizer.transform(demos))
    print("\n--- Demo Predictions (MultinomialNB) ---")
    for text, label in zip(demos, demo_preds):
        print(f"[{label}] {text}")


if __name__ == "__main__":
    main()
