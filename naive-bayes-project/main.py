import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# 1. Sample Dataset (Replace with your actual dataset/CSV)
data = {
    "text": [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Nah I don't think he goes to usf, he lives around here though",
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize!",
        "Even my brother is not like to speak with me. They treat me like aids patent.",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore."
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# 2. Feature Extraction
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Model Training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predictions & Evaluation
y_pred = model.predict(X_test)

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))
