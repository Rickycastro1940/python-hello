"""Milestone 7 RAG — Phase 3: FastAPI query endpoint (+ serves the UI).

Thin HTTP layer: it imports `query()` from data/pipelines and returns ONLY the
model-generated answer string. No retrieval or generation logic lives here, and
raw Qdrant results / scores are never returned to the client.

Run: uv run uvicorn services.api.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.pipelines.rag import query

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UI_FILE = PROJECT_ROOT / "uis" / "index.html"

app = FastAPI(title="Brasaland Knowledge Base", version="1.0.0")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.post("/knowledge/query", response_model=QueryResponse)
def knowledge_query(request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question from the Brasaland knowledge base."""
    return QueryResponse(answer=query(request.question))


@app.get("/")
def home() -> FileResponse:
    """Serve the minimal query UI."""
    return FileResponse(UI_FILE)
