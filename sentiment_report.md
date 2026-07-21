# WeLoveReviews — Customer Review Sentiment Report

**Client account:** Harbor House Café  
**Prepared for:** Account manager handoff  
**Reviews analyzed:** 500  
**Model used:** `prajjwal1/bert-mini` (Hugging Face, pinned revision)

---

## 1. Executive summary

The business’s **4.5 / 5** average star rating is **accurate** for this dataset — the 500 reviews average exactly **4.5 stars**.

However, the automated text-sentiment run returned **100% positive**. That does **not** line up with either the star distribution or a human reading of the written reviews. There is real dissatisfaction and lukewarm feedback in the text that the model failed to surface.

**Bottom line for the client:** The 4.5-star average is a fair headline number, but it should not be read as “every written review is glowing.” Roughly **1 in 7** reviews are 3 stars or below, and many of those texts are clearly critical or only mildly okay.

---

## 2. Total reviews analyzed

| Metric | Value |
|--------|------:|
| Total reviews | 500 |
| Fields used | `review_id`, `rating`, `review_text` |
| Actual average star rating | **4.5 / 5** |

### Star rating distribution

| Stars | Count | Share |
|------:|------:|------:|
| 5 | 360 | 72.0% |
| 4 | 72 | 14.4% |
| 3 | 38 | 7.6% |
| 2 | 18 | 3.6% |
| 1 | 12 | 2.4% |

---

## 3. Sentiment breakdown (model)

| Sentiment | Count | Share |
|-----------|------:|------:|
| Positive | 500 | 100.0% |
| Neutral | 0 | 0.0% |
| Negative | 0 | 0.0% |

The model assigned a positive label to **every** review.

---

## 4. Comparison to the 4.5-star average

### What lines up

- The claimed **4.5-star** average matches the data exactly.
- Most reviews are high-star (**86.4%** are 4–5★), so an overall positive story is reasonable.

### What does not line up

If we treat stars as a simple sentiment proxy (4–5 = positive, 3 = neutral, 1–2 = negative):

| Signal | Positive | Neutral | Negative |
|--------|--------:|--------:|---------:|
| Star ratings (proxy) | 86.4% | 7.6% | 6.0% |
| Model text sentiment | 100.0% | 0.0% | 0.0% |

The model is **more optimistic than the stars**. It reports zero criticism, while the ratings show **30** low-star reviews (1–2★) and **38** mid-star reviews (3★).

---

## 5. Discrepancies found (and why)

### What we checked by hand

We manually read **20** reviews across low, mid, and high star ratings (documented in `manual_inspection.md`).

| Group | Model label | Human read | Outcome |
|-------|-------------|------------|---------|
| 7 low-star (1–2★) reviews | All positive | Clearly negative / disappointed | **Wrong** |
| 6 mid-star (3★) reviews | All positive | Lukewarm / neutral | **Wrong** |
| 7 high-star (5★) reviews | All positive | Clearly positive | **Correct** |

Examples the model missed:

- **Review #123 (1★):** sticky tables, long wait for water, small portions — labeled positive  
- **Review #442 (1★):** burnt coffee, rude cashier — labeled positive  
- **Review #475 (3★):** “Wouldn't go out of my way, but wouldn't avoid it either” — labeled positive  

Across the full dataset, **68** reviews with 1–3★ stars were still labeled positive.

### Why this is happening

1. **The model is not a reliable sentiment classifier for this task.**  
   `prajjwal1/bert-mini` is a small general language model. When used here for sentiment classification, its classification head was not properly sentiment-trained for this use case (Hugging Face warns that classifier weights are newly initialized). In practice it collapsed almost everything to “positive.”

2. **Confidence scores stayed weak even on harsh complaints** (typically ~0.53–0.58), which is another sign the model is not cleanly separating sentiment.

3. **Star ratings and written tone are related but not identical.**  
   Even a correct sentiment model can diverge from stars (e.g., a 5★ review that mentions a wait). Here, though, the gap is mostly a **model failure**, not a subtle rating-vs-text nuance — clear complaints were labeled positive.

---

## 6. Recommendation for the client conversation

- Lead with the **validated 4.5-star average** — that number is solid.
- Do **not** claim that written sentiment is 100% positive.
- Acknowledge that about **14%** of reviews are 3★ or below, and that written comments in that band include real service, quality, and wait-time issues.
- Treat this model run as a **technical pilot that failed quality checks**, not as a client-ready sentiment metric. A sentiment-finetuned model (or a stronger pretrained sentiment model) should be used before quoting text-sentiment percentages externally.

---

*Supporting files in this repository: `data/reviews_with_sentiment.csv`, `manual_inspection.md`, `sentiment_analysis.py`.*
