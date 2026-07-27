"""Milestone 7 RAG — Phase 1: data preparation and indexing.

Responsibilities (kept separate from retrieval/generation in data/pipelines):
  - `setup()`   : read the corpus, chunk it, embed each chunk, upsert to Qdrant.
  - `embed()`   : turn one text into a vector with a DEDICATED embeddings model
                  (never the generation model).
  - `chunk_document()` : split a source document into self-contained semantic
                         chunks.

Company-specific values come from CONTEXT-brasaland (Milestone 7):
collection name, payload field names, and the four source documents.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KB_DIR = PROJECT_ROOT / "docs" / "company-knowledge-base"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_storage"

# --- Company-specific config (from CONTEXT-brasaland.en.md) ---
COMPANY = "brasaland"
LANGUAGE = "en"
COLLECTION = "brasaland_knowledge_base"

# Dedicated embeddings model (NOT the generation model used in query()).
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension

_embedder = None  # lazy singleton so importing the module stays cheap
_client = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed(text: str) -> list[float]:
    """Embed a single text with the dedicated embeddings model (cosine-ready).

    Used identically for chunks at index time and for the question at query
    time. Vectors are L2-normalized so cosine similarity is well-behaved.
    """
    vector = _get_embedder().encode(
        text, normalize_embeddings=True, show_progress_bar=False
    )
    return vector.tolist()


def _get_client():
    """Return a lazily-opened Qdrant client.

    Uses the server at ``QDRANT_URL`` (the docker-compose Qdrant service) when
    set; otherwise falls back to embedded on-disk mode at ``data/qdrant_storage``
    so the pipeline still runs with no server.
    """
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        url = os.environ.get("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url)
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def _source_document_key(filename: str) -> str:
    """Map a corpus filename to its CONTEXT `source_document` key.

    e.g. 'brasaland-loyalty-program.en.md' -> 'loyalty-program'.
    """
    stem = filename.replace("brasaland-", "").replace(".en.md", "")
    return stem


def chunk_document(text: str, source_document: str) -> list[dict]:
    """Split one Markdown document into self-contained semantic chunks.

    Strategy: chunk by blank-line-separated blocks (paragraphs and whole list
    or numbered-procedure blocks). Lists/procedures stay intact so no rule or
    condition is cut in half. The document title is captured as context and the
    first line of each block becomes the `section` label.
    """
    lines = text.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else source_document

    # Split the body (everything after the title) into blank-line blocks.
    body = "\n".join(lines[1:]).strip()
    raw_blocks = [b.strip() for b in body.split("\n\n") if b.strip()]

    chunks: list[dict] = []
    for block in raw_blocks:
        first_line = block.splitlines()[0].strip()
        # A trailing ':' marks a list/procedure header -> use it as the section.
        section = first_line.rstrip(":") if first_line.endswith(":") else title
        chunks.append(
            {
                "source_document": source_document,
                "section": section,
                "chunk_index": len(chunks),
                # store the clean block; prepend the title for embedding context
                "text": block,
                "embed_input": f"{title}\n{block}",
            }
        )
    return chunks


def load_corpus(kb_dir: Path = KB_DIR) -> list[dict]:
    """Read and chunk every source document in the knowledge-base folder."""
    all_chunks: list[dict] = []
    for path in sorted(kb_dir.glob("*.en.md")):
        source_document = _source_document_key(path.name)
        all_chunks.extend(
            chunk_document(path.read_text(encoding="utf-8"), source_document)
        )
    return all_chunks


def _point_id(source_document: str, chunk_index: int) -> str:
    """Deterministic UUID so re-running setup() upserts in place (idempotent)."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{COMPANY}:{source_document}:{chunk_index}")
    )


def setup(kb_dir: Path = KB_DIR) -> dict:
    """Build the Brasaland knowledge base: chunk, embed and upsert into Qdrant.

    Idempotent: the collection is recreated and points use deterministic IDs,
    so repeated runs never duplicate content.
    """
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = _get_client()
    chunks = load_corpus(kb_dir)

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    points = []
    for chunk in chunks:
        payload = {
            "company": COMPANY,
            "source_document": chunk["source_document"],
            "section": chunk["section"],
            "language": LANGUAGE,
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        }
        points.append(
            PointStruct(
                id=_point_id(chunk["source_document"], chunk["chunk_index"]),
                vector=embed(chunk["embed_input"]),
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)

    per_doc: dict[str, int] = {}
    for chunk in chunks:
        per_doc[chunk["source_document"]] = per_doc.get(chunk["source_document"], 0) + 1

    return {"collection": COLLECTION, "total_chunks": len(points), "per_document": per_doc}


if __name__ == "__main__":
    stats = setup()
    print(f"Indexed {stats['total_chunks']} chunks into '{stats['collection']}':")
    for doc, n in sorted(stats["per_document"].items()):
        print(f"  {doc}: {n} chunks")
