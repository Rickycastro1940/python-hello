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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

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


def main() -> None:
    df = load_dataframe()
    labels = sorted(df["label"].unique())

    # Shared TF-IDF features for a fair head-to-head comparison
    vectorizer = TfidfVectorizer(stop_words="english")
    X_sparse = vectorizer.fit_transform(df["text"])
    y = df["label"]

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

    summary = pd.DataFrame(
        [
            {
                "model": r["model_name"],
                "accuracy": r["accuracy"],
                "weighted_f1": r["f1"],
            }
            for r in results
        ]
    ).sort_values(["accuracy", "weighted_f1"], ascending=False)

    print("\n===== Comparison Summary =====")
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    chosen_name = "MultinomialNB"
    chosen = next(r for r in results if r["model_name"] == chosen_name)

    print(f"\nBest by metrics on this split: {best['model']}")
    print(
        f"Selected model for this project: {chosen_name} "
        f"(accuracy={chosen['accuracy']:.2%}, "
        f"weighted F1={chosen['f1']:.2%})"
    )

    if best["model"] == chosen_name:
        print(
            "Confirmation: MultinomialNB is the right choice for this "
            "TF-IDF text classification task."
        )
    else:
        print(
            f"Note: {best['model']} scored higher on this small split, "
            "but MultinomialNB remains the theoretically preferred model "
            "for non-negative TF-IDF / bag-of-words text features "
            "(GaussianNB assumes continuous normals; BernoulliNB ignores "
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
