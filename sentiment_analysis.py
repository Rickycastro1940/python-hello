"""Load WeLoveReviews customer reviews for sentiment analysis."""

from pathlib import Path

import pandas as pd
from transformers import Pipeline, pipeline

DATA_PATH = Path(__file__).resolve().parent / "data" / "reviews.csv"
OUTPUT_PATH = (
    Path(__file__).resolve().parent / "data" / "reviews_with_sentiment.csv"
)
EXPECTED_COLUMNS = ("review_id", "rating", "review_text")
EXPECTED_COUNT = 500

# Pinned model identity — do not resolve to an unpinned "latest" alias.
MODEL_NAME = "prajjwal1/bert-mini"
MODEL_REVISION = "5e123abc2480f0c4b4cac186d3b3f09299c258fc"

# bert-mini exposes binary labels; map them to readable sentiment names.
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
}


def load_reviews(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the reviews dataset and validate its shape."""
    if not path.exists():
        raise FileNotFoundError(f"Reviews file not found: {path}")

    reviews = pd.read_csv(path)

    missing = [col for col in EXPECTED_COLUMNS if col not in reviews.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if len(reviews) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} reviews, found {len(reviews)}"
        )

    if reviews["review_text"].isna().any():
        raise ValueError("Found empty review_text values")

    return reviews


def load_sentiment_model(
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> Pipeline:
    """Load the sentiment pipeline once; reuse this object for all reviews.

    Weights are downloaded/cached under ~/.cache/huggingface on first run.
    Call this outside any per-review loop so the model is not re-created.
    """
    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        revision=revision,
    )


def predict_sentiments(
    reviews: pd.DataFrame,
    sentiment_model: Pipeline,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run inference on every review using one shared model instance.

    Stores predicted_label and predicted_score alongside each review.
    """
    texts = reviews["review_text"].tolist()
    # One pipeline call (batched) — model is not reloaded per review.
    raw_preds = sentiment_model(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=512,
    )

    if len(raw_preds) != len(reviews):
        raise ValueError(
            f"Expected {len(reviews)} predictions, got {len(raw_preds)}"
        )

    result = reviews.copy()
    result["predicted_label_raw"] = [pred["label"] for pred in raw_preds]
    result["predicted_label"] = [
        LABEL_MAP.get(pred["label"], pred["label"]) for pred in raw_preds
    ]
    result["predicted_score"] = [float(pred["score"]) for pred in raw_preds]
    return result


def calculate_sentiment_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall % positive / neutral / negative from predictions."""
    total = len(predictions)
    if total == 0:
        raise ValueError("Cannot calculate breakdown for an empty dataset")

    counts = (
        predictions["predicted_label"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
    )
    breakdown = pd.DataFrame(
        {
            "sentiment": counts.index,
            "count": counts.values.astype(int),
            "percent": (counts.values / total * 100).round(2),
        }
    )
    return breakdown


def compare_to_star_rating(
    predictions: pd.DataFrame,
    business_avg_claim: float = 4.5,
) -> dict:
    """Compare model sentiment breakdown against star ratings / 4.5 claim."""
    total = len(predictions)
    avg_rating = float(predictions["rating"].mean())
    breakdown = calculate_sentiment_breakdown(predictions)

    # Map stars to a sentiment proxy for an apples-to-apples view.
    star_proxy_counts = {
        "positive": int((predictions["rating"] >= 4).sum()),
        "neutral": int((predictions["rating"] == 3).sum()),
        "negative": int((predictions["rating"] <= 2).sum()),
    }
    star_proxy = {
        label: {
            "count": count,
            "percent": round(count / total * 100, 2),
        }
        for label, count in star_proxy_counts.items()
    }

    model_pct = {
        row.sentiment: float(row.percent) for row in breakdown.itertuples()
    }
    mismatches = {
        "low_stars_labeled_positive": int(
            (
                (predictions["rating"] <= 2)
                & (predictions["predicted_label"] == "positive")
            ).sum()
        ),
        "three_stars_labeled_positive": int(
            (
                (predictions["rating"] == 3)
                & (predictions["predicted_label"] == "positive")
            ).sum()
        ),
        "high_stars_labeled_negative": int(
            (
                (predictions["rating"] >= 4)
                & (predictions["predicted_label"] == "negative")
            ).sum()
        ),
    }

    lines_up = (
        abs(avg_rating - business_avg_claim) < 0.05
        and model_pct.get("positive", 0) >= 80
        and mismatches["low_stars_labeled_positive"] == 0
    )

    return {
        "business_avg_claim": business_avg_claim,
        "actual_avg_rating": round(avg_rating, 3),
        "rating_matches_claim": abs(avg_rating - business_avg_claim) < 0.05,
        "model_breakdown_pct": model_pct,
        "star_proxy_pct": {
            k: v["percent"] for k, v in star_proxy.items()
        },
        "star_proxy_counts": star_proxy_counts,
        "mismatches": mismatches,
        "lines_up": lines_up,
    }


def print_star_comparison(comparison: dict) -> None:
    """Print a readable comparison for the account manager."""
    print("\nComparison vs 4.5-star average:")
    print(
        f"  Claimed average: {comparison['business_avg_claim']}"
        f"  |  Actual average: {comparison['actual_avg_rating']}"
        f"  |  Match: {comparison['rating_matches_claim']}"
    )
    print(
        "  Star-proxy sentiment "
        "(4–5=positive, 3=neutral, 1–2=negative): "
        f"{comparison['star_proxy_pct']}"
    )
    print(f"  Model sentiment: {comparison['model_breakdown_pct']}")
    print(f"  Mismatches: {comparison['mismatches']}")
    print(
        "  Overall lineup: "
        + (
            "YES — model and stars tell a consistent story."
            if comparison["lines_up"]
            else "NO — model and stars diverge in important places."
        )
    )


if __name__ == "__main__":
    df = load_reviews()
    print(f"Loaded {len(df)} reviews from {DATA_PATH}")

    # Load once — reuse for all reviews below.
    sentiment_model = load_sentiment_model()
    print(
        f"Loaded model once: {MODEL_NAME} @ {MODEL_REVISION[:12]} "
        f"({type(sentiment_model.model).__name__})"
    )

    predictions = predict_sentiments(df, sentiment_model)
    predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(predictions)} predictions to {OUTPUT_PATH}")

    breakdown = calculate_sentiment_breakdown(predictions)
    print("\nOverall sentiment breakdown:")
    print(breakdown.to_string(index=False))

    comparison = compare_to_star_rating(predictions)
    print_star_comparison(comparison)

    print("\nSample:")
    print(
        predictions[
            ["review_id", "rating", "predicted_label", "predicted_score"]
        ]
        .head(5)
        .to_string(index=False)
    )
