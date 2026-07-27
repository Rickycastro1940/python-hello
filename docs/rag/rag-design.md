# Brasaland RAG Knowledge Base — Design (Milestone 7)

Reference: `content/contexts/07-trainning-rag/brasaland/CONTEXT-brasaland.en.md`
(mirrored corpus in `docs/company-knowledge-base/`).

Reproduce: `uv run python -m data.process.rag` (build index) then
`uv run python -m data.eval.evaluate_rag` (Recall@3), or serve the API with
`uv run uvicorn services.api.main:app`.

## 1. RAG process (end-to-end flow)

```mermaid
flowchart LR
    Docs["docs/company-knowledge-base/*.en.md"] --> Setup["setup(): parse + chunk"]
    Setup --> Embed1["embed(chunk_text)"]
    Embed1 --> Qdrant[("Qdrant collection<br/>brasaland_knowledge_base")]
    Question["user question"] --> Embed2["embed(question)"]
    Embed2 --> Retrieve["retrieve(k, min_score)"]
    Qdrant --> Retrieve
    Retrieve --> Gen["generate_answer(question, context)"]
    Gen --> Answer["answer string"]
    Query["query()"] -.orchestrates.-> Retrieve
    Query -.orchestrates.-> Gen
```

Modules (one responsibility each, per the ticket):
- `data/process/rag.py` -> `setup()`, `embed()` (chunking + indexing).
- `data/pipelines/rag.py` -> `retrieve()`, `generate_answer()`, `query()`.
- `services/api/` -> FastAPI `POST /knowledge/query`.
- `uis/` -> minimal query UI.
- `tests/pipelines/test_rag.py` -> mocked unit tests for `retrieve()` + `query()`.

## 2. Chunking strategy

The four source documents are short, well-structured Markdown with clear
semantic units (a tier list, a numbered procedure, a per-dish allergen block).
We chunk **by semantic section**, not fixed character count, so no rule or
condition is cut in half:
- Split on Markdown headings and blank-line-separated blocks (tier list, FAQ
  block, numbered procedure, per-category list).
- Keep each list/procedure together as one chunk when small; otherwise group by
  subsection so a chunk is self-contained.
- Result: **18 chunks total**, all documents >= 3 (per CONTEXT section 5):
  loyalty-program 5, waste-protocol 5, menu-allergens 4, supplier-ordering 4.
  Each chunk carries its `section` title in the payload.

## 3. Payload / metadata (from CONTEXT section 3)

Each Qdrant point payload contains at minimum:
`company="brasaland"`, `source_document` (`loyalty-program | waste-protocol | menu-allergens | supplier-ordering`),
`section`, `language="en"`, `chunk_index`, and `text` (chunk body for prompt assembly).

Idempotency: deterministic point IDs (UUIDv5 of `source_document + chunk_index`)
so re-running `setup()` upserts in place instead of duplicating.

## 4. Embedding & generation practices

Two different models (never the same for both jobs):
- **Embeddings** (`embed()`): `sentence-transformers` `all-MiniLM-L6-v2`
  (384-dim, cosine distance) — local, no API key. Used identically at index
  time and query time.
- **Generation** (`generate_answer()`): a local instruction model
  (e.g. `google/flan-t5-base`) — answers from a salesperson's perspective using
  only retrieved context.

> Note: the milestone suggests the 4Geeks-provided hosted models. No 4Geeks
> credentials are available in this environment, so we default to local models.
> `embed()` and `generate_answer()` are isolated so swapping to the 4Geeks
> hosted embedding/generation endpoints is a one-function change each.

- **Distance metric:** cosine. **Vector dim:** 384 (MiniLM). Vectors are
  L2-normalized in `embed()`.
- **`min_score` threshold: 0.40.** Tuned against `data/eval/test-queries.json`:
  correct matches score 0.56-0.80 while off-topic questions score <= 0.29, so
  0.40 separates them with margin. If no chunk clears it, `query()` returns an
  honest "not enough information" answer — it never invents company facts.
- **Idempotency:** `setup()` recreates the collection and uses deterministic
  UUIDv5 point IDs (`company:source_document:chunk_index`), so re-running never
  duplicates points.
- **Qdrant deployment:** runs as a service in `docker-compose.yml`
  (`qdrant/qdrant`, REST 6333 / gRPC 6334). The client reads `QDRANT_URL`
  (server mode) with an embedded on-disk fallback (`data/qdrant_storage/`) when
  unset. Verified: `docker compose up -d` -> collection `status: green`,
  `points_count: 18`, `dim: 384`, `distance: Cosine`; Recall@3 = 100% against
  the server. Model IDs and `QDRANT_URL` are configurable via `.env`
  (`.env.example`).

## 5. Guardrails (CONTEXT section 4 & 6)

- Faithfulness: the answer must not contain numbers (%, amounts, kg) absent
  from the retrieved chunks.
- Allergens: never answer "zero risk"; follow `brasaland-menu-allergens.en.md`
  wording literally.
- Currency: keep COP/USD amounts exactly as written; no auto-conversion.
- Recall@3 target: >= 80% on the test queries. **Achieved: 100% (10/10)** — the
  expected source document appears in the top 3 for every test question.

## 6. Example (live)

`POST /knowledge/query {"question": "How many points do I need for the Gold tier?"}`
-> `{"answer": "50+ points."}` (grounded in `loyalty-program`; off-topic
questions like "What is the wifi password?" return the honest fallback).
