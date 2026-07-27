"""Milestone 7 RAG — Phase 5: unit tests for retrieve() and query().

No live Qdrant or models: the Qdrant client is a stub, `embed()` is monkeypatched,
and generation is mocked. Verifies the two behaviours the brief calls out:
  - retrieve() drops hits below min_score and can return fewer than k.
  - query() returns the *generated* answer, not raw chunk text.
"""

import pytest

from data.pipelines import rag as pipe
from data.process import rag as kb


class _FakePoint:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class _FakeResponse:
    def __init__(self, points):
        self.points = points


class _FakeClient:
    """Minimal stand-in for QdrantClient.query_points()."""

    def __init__(self, points):
        self._points = points

    def query_points(self, collection_name, query, limit, with_payload):
        return _FakeResponse(self._points[:limit])


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    # Avoid loading the real embeddings model in unit tests.
    monkeypatch.setattr(kb, "embed", lambda text: [0.0] * kb.VECTOR_DIM)


def test_retrieve_excludes_hits_below_min_score():
    points = [
        _FakePoint(0.90, {"source_document": "loyalty-program", "section": "tiers", "text": "Gold 50+"}),
        _FakePoint(0.55, {"source_document": "loyalty-program", "section": "redeem", "text": "redeem 15"}),
        _FakePoint(0.20, {"source_document": "waste-protocol", "section": "target", "text": "4%"}),
    ]
    results = pipe.retrieve("q", k=5, min_score=0.40, client=_FakeClient(points))

    assert len(results) == 2  # the 0.20 hit is filtered out
    assert all(r["score"] >= 0.40 for r in results)
    assert results[0]["source_document"] == "loyalty-program"


def test_retrieve_can_return_fewer_than_k():
    points = [_FakePoint(0.9, {"source_document": "x", "section": "s", "text": "t"})]
    results = pipe.retrieve("q", k=5, min_score=0.40, client=_FakeClient(points))
    assert len(results) == 1  # only one point available, k not forced


def test_retrieve_returns_empty_when_nothing_clears_threshold():
    points = [_FakePoint(0.10, {"source_document": "x", "section": "s", "text": "t"})]
    results = pipe.retrieve("q", k=5, min_score=0.40, client=_FakeClient(points))
    assert results == []


def test_query_returns_generated_answer_not_raw_chunk(monkeypatch):
    chunks = [{"source_document": "loyalty-program", "section": "tiers",
               "text": "RAW CHUNK: Gold tier 50+ points", "score": 0.8}]
    captured = {}

    def fake_generate(question, context):
        captured["context"] = context
        return "You need 50 or more points to reach Gold."

    monkeypatch.setattr(pipe, "retrieve", lambda q: chunks)
    monkeypatch.setattr(pipe, "generate_answer", fake_generate)

    answer = pipe.query("How many points for Gold tier?")

    assert answer == "You need 50 or more points to reach Gold."
    assert "RAW CHUNK" not in answer  # not the raw retrieval result
    assert captured["context"] == chunks  # generation received the retrieved context


def test_query_honest_fallback_when_no_context(monkeypatch):
    monkeypatch.setattr(pipe, "retrieve", lambda q: [])
    # generate_answer with empty context must not invent facts
    assert pipe.query("something off-topic") == pipe.NO_CONTEXT_ANSWER
