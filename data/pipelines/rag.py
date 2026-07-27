"""Milestone 7 RAG — Phase 2: retrieval + generation pipeline.

`query()` is the only function external consumers call; it is literally
`retrieve()` + `generate_answer()`, kept as separate functions so a later
LangGraph agent can reuse each step independently.

- `retrieve()`        : embed the question, search Qdrant, drop hits below
                        `min_score`, return payload dicts (never raw SDK objects).
- `generate_answer()` : assemble a salesperson-voice prompt from the retrieved
                        context and call the GENERATION model (not the embeddings
                        model).
- `query()`           : retrieve -> generate; honest fallback when nothing clears
                        the threshold (never invents company facts).
"""

from __future__ import annotations

from data.process import rag as kb

# Minimum cosine similarity a chunk must reach to be used as context.
# Tuned against data/eval/test-queries.json: correct matches score >= 0.56,
# off-topic questions score <= 0.29, so 0.40 separates them with margin.
DEFAULT_MIN_SCORE = 0.40

# Dedicated GENERATION model — different from the embeddings model in kb.embed().
GENERATION_MODEL = "google/flan-t5-base"
_generator = None

NO_CONTEXT_ANSWER = (
    "I'm sorry, I don't have that information in our Brasaland knowledge base. "
    "Please check with your operations lead so we can get you an accurate answer."
)


def retrieve(query: str, *, k: int = 5, min_score: float = DEFAULT_MIN_SCORE,
             client=None) -> list[dict]:
    """Return up to `k` retrieved chunks whose score clears `min_score`.

    Embeds the question with the same `kb.embed()` used at index time, searches
    Qdrant, filters out weak matches, and returns plain payload dicts (with the
    similarity `score` attached) — not raw Qdrant SDK objects. May return fewer
    than `k`, or an empty list when nothing clears the bar.
    """
    client = client or kb._get_client()
    vector = kb.embed(query)

    response = client.query_points(
        collection_name=kb.COLLECTION,
        query=vector,
        limit=k,
        with_payload=True,
    )

    results: list[dict] = []
    for point in response.points:
        if point.score is None or point.score < min_score:
            continue
        payload = dict(point.payload or {})
        payload["score"] = float(point.score)
        results.append(payload)
    return results


def _get_generator():
    global _generator
    if _generator is None:
        from transformers import pipeline

        _generator = pipeline("text2text-generation", model=GENERATION_MODEL)
    return _generator


def _build_prompt(question: str, context: list[dict]) -> str:
    blocks = []
    for c in context:
        blocks.append(f"[{c.get('source_document')} - {c.get('section')}]\n{c.get('text')}")
    context_text = "\n\n".join(blocks)
    return (
        "You are a knowledgeable Brasaland sales assistant. Answer the customer's "
        "question in a confident, helpful salesperson tone, using ONLY the "
        "information in the context below. Do not invent numbers, amounts, "
        "percentages or facts that are not in the context. For allergen "
        "questions never claim zero risk.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(question: str, context: list[dict]) -> str:
    """Generate the final answer from retrieved context using the generation LLM."""
    if not context:
        return NO_CONTEXT_ANSWER
    prompt = _build_prompt(question, context)
    output = _get_generator()(prompt, max_new_tokens=160, do_sample=False)
    return output[0]["generated_text"].strip()


def query(question: str) -> str:
    """Answer a question end-to-end: retrieve() -> generate_answer().

    The only entry point external consumers (the API) should call.
    """
    context = retrieve(question)
    return generate_answer(question, context)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How many points do I need for Gold tier?"
    print(f"Q: {q}")
    print(f"A: {query(q)}")
