# Manual Prediction Inspection — WeLoveReviews

Inspected **20 reviews** stratified by star rating (7 low / 6 mid / 7 high).
Model: `prajjwal1/bert-mini` (pinned). Every sampled review was labeled **positive**.

Human judgment key:
- **Agree** — model label matches how the text reads
- **Disagree** — model label looks wrong
- **Expected human label** — what the text actually expresses

---

## Low-star reviews (1–2★) — model said positive

| review_id | stars | model | human | notes |
|----------:|------:|-------|-------|-------|
| 123 | 1 | positive | **Disagree → negative** | Small portions, 25-min wait for water, sticky tables, unswept floor. Clear complaint. |
| 314 | 1 | positive | **Disagree → negative** | Cold eggs; nobody checked on them; "Not sure I'll be returning." |
| 442 | 1 | positive | **Disagree → negative** | Burnt coffee, rude cashier, cramped/chaotic. Explicitly won't return. |
| 465 | 1 | positive | **Disagree → negative** | Bland/overpriced food, order wrong twice, chaotic room. |
| 115 | 2 | positive | **Disagree → negative** | Stale pastry, long wait for water, "Disappointing experience." |
| 192 | 2 | positive | **Disagree → negative** | "Decent, nothing special" + "Just didn't love it enough to come back." Soft negative. |
| 199 | 2 | positive | **Disagree → negative** | Small portions, annoyed staff, "Expected a lot more." |

**Result:** 7/7 low-star samples look **wrong**. The model missed every clear complaint.

---

## Mid-star reviews (3★) — model said positive

| review_id | stars | model | human | notes |
|----------:|------:|-------|-------|-------|
| 43 | 3 | positive | **Disagree → neutral** | "Okay" coffee, "fine, nothing memorable," "fine for a quick stop." |
| 170 | 3 | positive | **Disagree → neutral** | Decent/nothing special; polite but slow; lukewarm. |
| 362 | 3 | positive | **Disagree → neutral** | Okay coffee, average service, no complaints — not enthusiasm. |
| 421 | 3 | positive | **Disagree → neutral** | "Nothing to write home about"; maybe another shot. |
| 475 | 3 | positive | **Disagree → neutral** | Wouldn't go out of the way, wouldn't avoid — classic neutral. |
| 493 | 3 | positive | **Disagree → neutral** | Same lukewarm pattern as 475. |

**Result:** 6/6 mid-star samples look **wrong** (should be neutral, not positive). Model has no useful neutral class in practice.

---

## High-star reviews (4–5★) — model said positive

| review_id | stars | model | human | notes |
|----------:|------:|-------|-------|-------|
| 37 | 5 | positive | **Agree → positive** | Warm staff, great sourdough, strong recommend. |
| 89 | 5 | positive | **Agree → positive** | Loved ambiance, fast friendly service, "go-to spot." |
| 216 | 5 | positive | **Agree → positive** | Caring team, sourdough worth the visit. |
| 321 | 5 | positive | **Agree → positive** | Mentions a 25-min wait, but says food made up for it; overall praise. |
| 453 | 5 | positive | **Agree → positive** | Birthday brunch praise; planning next visit. |
| 478 | 5 | positive | **Agree → positive** | Sourdough + great spot; go-to language. |
| 490 | 5 | positive | **Agree → positive** | Perfect mood, rich coffee, planning next visit. |

**Result:** 7/7 high-star samples look **correct** (positive text → positive label).

---

## Summary

- Inspected: **20** reviews
- Model labels that look wrong: **13 / 20** (all 7 low-star + all 6 mid-star)
- Model labels that look right: **7 / 20** (only the clearly glowing 5★ reviews)
- Confidence scores stayed ~0.53–0.58 even on harsh complaints — another red flag that the head is not truly discriminating sentiment

**Conclusion:** Do not trust this model's positive-only output for the client report. Star ratings and a human read of the text show real negativity and neutrality that the model flattens away. Root cause remains the uninitialized / non–sentiment-fine-tuned classifier head on `prajjwal1/bert-mini`.
