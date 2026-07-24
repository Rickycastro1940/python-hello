"""Spam/ham text classification with Naive Bayes.

Model choice: MultinomialNB
- GaussianNB: for continuous numeric features (normal distribution). Poor fit for
  sparse TF-IDF text vectors.
- BernoulliNB: for binary presence/absence features. Usable for text, but ignores
  term frequency / TF-IDF weight.
- MultinomialNB: for discrete count-like / TF-IDF features. Best match for bag-of-words
  text classification, so we use this implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Sample Dataset (Replace with your actual dataset/CSV)
data = {
    "text": [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Nah I don't think he goes to usf, he lives around here though",
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize!",
        "Even my brother is not like to speak with me. They treat me like aids patent.",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore.",
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham"],
}

df = pd.DataFrame(data)

# 2. Feature Extraction (non-negative TF-IDF weights → MultinomialNB)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Model Training — chosen implementation: MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predictions & Evaluation
y_pred = model.predict(X_test)
labels = sorted(y.unique())

print("Chosen model: MultinomialNB")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print()
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=labels)
print("--- Confusion Matrix ---")
print(pd.DataFrame(cm, index=labels, columns=labels))

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("MultinomialNB Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("\nSaved plot: confusion_matrix.png")

# 6. Demo predictions on new messages
demos = [
    "Congratulations! Claim your FREE prize now",
    "Are we still meeting for coffee later?",
]
demo_preds = model.predict(vectorizer.transform(demos))
print("\n--- Demo Predictions ---")
for text, label in zip(demos, demo_preds):
    print(f"[{label}] {text}")
