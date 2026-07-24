"""Train and evaluate a Multinomial Naive Bayes text classifier."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = Path(__file__).resolve().parent / "data.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load labeled text data from CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place a CSV with columns 'text' and 'label' as data.csv."
        )

    data = pd.read_csv(path)
    expected = {"text", "label"}
    missing = expected - set(data.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    data = data.dropna(subset=["text", "label"])
    if data.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    return data


def train_and_evaluate(data: pd.DataFrame) -> None:
    """Vectorize text, train MultinomialNB, and print evaluation metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    vectorizer = CountVectorizer()
    x_train_counts = vectorizer.fit_transform(x_train)
    x_test_counts = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_counts, y_train)

    predictions = model.predict(x_test_counts)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Samples: {len(data)}")
    print(f"Train / test: {len(x_train)} / {len(x_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print()
    print(classification_report(y_test, predictions))

    # Demo a few predictions on held-out phrases.
    demos = [
        "Free cash prize click this link",
        "Please send the meeting agenda",
    ]
    demo_counts = vectorizer.transform(demos)
    demo_preds = model.predict(demo_counts)
    print("Demo predictions:")
    for text, label in zip(demos, demo_preds):
        print(f"  [{label}] {text}")


def main() -> None:
    data = load_data()
    train_and_evaluate(data)


if __name__ == "__main__":
    main()
