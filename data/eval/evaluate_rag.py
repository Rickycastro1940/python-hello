"""Milestone 7 RAG — Recall@3 evaluation and min_score tuning.

Runs the retriever against data/eval/test-queries.json and reports, for each
question, whether the expected source document appears among the top-3 results,
plus the top score (to tune the similarity threshold). Retrieval only — does
not load the generation model.

Run: uv run python -m data.eval.evaluate_rag
"""

from __future__ import annotations

import json
from pathlib import Path

from data.pipelines import rag as pipe

TEST_QUERIES = Path(__file__).resolve().parent / "test-queries.json"


def main() -> None:
    data = json.loads(TEST_QUERIES.read_text(encoding="utf-8"))
    queries = data["queries"]

    hits = 0
    top_scores = []
    print(f"Recall@3 over {len(queries)} test questions\n")
    for q in queries:
        # min_score=0 to measure pure retrieval recall (thresholding is separate)
        results = pipe.retrieve(q["question"], k=3, min_score=0.0)
        docs = [r["source_document"] for r in results]
        expected = q["expected_source_document"]
        ok = expected in docs
        hits += ok
        top = results[0]["score"] if results else 0.0
        top_scores.append(top)
        print(
            f"[{'HIT ' if ok else 'MISS'}] top={top:.3f} exp={expected:<16} "
            f"got={docs}  | {q['question']}"
        )

    recall = hits / len(queries)
    print(f"\nRecall@3: {hits}/{len(queries)} = {recall:.0%}")
    print(
        f"Top-score range: min={min(top_scores):.3f} "
        f"max={max(top_scores):.3f} mean={sum(top_scores)/len(top_scores):.3f}"
    )


if __name__ == "__main__":
    main()
