#!/usr/bin/env python3
"""Seed Athena with public-domain demo documents so the UI is immediately usable.

Usage:
    python -m scripts.seed_demo [--host http://localhost:8000] [--api-key KEY]

Documents seeded:
  - A short primer on Retrieval-Augmented Generation (generated)
  - An overview of vector databases and HNSW indexing
  - A note on chunking strategies and their trade-offs

These are plain-text summaries — no PDF parsing required.  The seeder is
idempotent: it checks the /api/documents list first and skips files that are
already present.
"""

import argparse
import sys

import httpx

DEMO_DOCS: list[tuple[str, str]] = [
    (
        "rag_primer.txt",
        """\
Retrieval-Augmented Generation (RAG) Overview
==============================================

Retrieval-Augmented Generation (RAG) is a hybrid natural-language processing
architecture that combines a retrieval system with a generative language model.
Instead of relying solely on knowledge encoded in model weights, RAG fetches
relevant passages from an external corpus at inference time and conditions the
LLM on those passages when generating a response.

Key components:
1. Document ingestion — raw documents (PDF, DOCX, HTML, plain text) are parsed,
   split into overlapping chunks, and embedded into dense vectors.
2. Vector store — embeddings are stored alongside their source text in a vector
   database (e.g. pgvector with an HNSW index) that supports approximate-nearest-
   neighbour (ANN) search.
3. Retrieval — given a user question, the question is embedded and the ANN index
   returns the top-k most similar chunks (dense retrieval). A BM25 full-text
   index provides complementary sparse retrieval; the two lists are fused via
   Reciprocal Rank Fusion (RRF).
4. Re-ranking — a cross-encoder model scores each candidate chunk against the
   question to produce a refined shortlist.
5. Generation — the LLM is prompted with the question and the re-ranked chunks as
   context; it synthesises a grounded, cited answer.

RAG dramatically reduces hallucination rates because the model can reference
authoritative passages rather than rely on potentially stale or incorrect
parametric knowledge.
""",
    ),
    (
        "vector_databases.txt",
        """\
Vector Databases and HNSW Indexing
====================================

A vector database stores high-dimensional embedding vectors and supports
efficient similarity search via Approximate Nearest Neighbour (ANN) algorithms.

Why ANN instead of exact search?
Exact k-NN requires computing the distance between the query vector and every
stored vector — O(n) per query.  For millions of vectors at 1024 dimensions this
is prohibitively slow.  ANN trades a small, controllable accuracy loss for
orders-of-magnitude faster queries.

Hierarchical Navigable Small World (HNSW):
HNSW builds a multi-layer graph where each node is an embedding vector.  Upper
layers are sparse "highway" graphs for coarse navigation; the bottom layer is a
dense proximity graph.  Search starts at a random entry point in the top layer,
greedily moves toward the query vector, then descends layer by layer to refine
the neighbourhood.

Key hyperparameters:
  - M (max connections per node) — higher M → better recall, more memory.
  - ef_construction (beam width during build) — higher → slower build, better index quality.
  - ef_search (beam width during query) — tunable at query time for recall/latency trade-off.

pgvector (the PostgreSQL extension used in this project) implements HNSW natively,
allowing vector similarity queries alongside standard SQL — no separate
infrastructure required.
""",
    ),
    (
        "chunking_strategies.txt",
        """\
Document Chunking Strategies
=============================

Before embedding, long documents must be split into chunks small enough for the
embedding model's context window and the LLM's prompt budget.  The chunking
strategy significantly affects retrieval quality.

Fixed-size chunking
  Split at every N tokens with an overlap of K tokens.  Simple and fast, but
  often cleaves sentences mid-thought, degrading embedding quality.

Recursive character splitting
  Tries a hierarchy of separators (double newline → newline → period → space) and
  only falls back to the next separator if a chunk would exceed the size limit.
  Tends to produce semantically cleaner chunks than fixed-size splitting.
  This is the default strategy in Athena based on RAGAS benchmark results
  (faithfulness 0.84, answer relevance 0.87).

Semantic chunking
  Groups consecutive sentences whose embeddings are similar; splits where
  cosine-similarity drops sharply.  Produces the most coherent chunks but is
  slower (requires embedding every sentence).

Overlap
  Every strategy benefits from overlapping adjacent chunks by 10–20 % of the
  chunk size.  This ensures that information at chunk boundaries appears in at
  least one chunk's full context.

Rule of thumb: start with recursive splitting at 512 tokens / 100-token overlap
and benchmark on your corpus before switching to semantic chunking.
""",
    ),
]


def already_uploaded(base_url: str, headers: dict[str, str], filename: str) -> bool:
    resp = httpx.get(f"{base_url}/api/documents", headers=headers, timeout=10)
    resp.raise_for_status()
    return any(d["filename"] == filename for d in resp.json())


def upload(base_url: str, headers: dict[str, str], filename: str, content: str) -> None:
    resp = httpx.post(
        f"{base_url}/api/documents/upload",
        headers=headers,
        files={"file": (filename, content.encode(), "text/plain")},
        data={"chunking_strategy": "recursive"},
        timeout=60,
    )
    resp.raise_for_status()
    doc = resp.json()
    print(f"  uploaded {filename!r}  →  {doc['chunk_count']} chunks  (id: {doc['id']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Athena with demo documents.")
    parser.add_argument("--host", default="http://localhost:8000", help="Athena API base URL")
    parser.add_argument("--api-key", default="", help="X-API-Key value if auth is enabled")
    args = parser.parse_args()

    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    print(f"Seeding demo documents → {args.host}")
    try:
        httpx.get(f"{args.host}/api/health", timeout=5).raise_for_status()
    except Exception as exc:
        print(f"ERROR: cannot reach {args.host} — {exc}", file=sys.stderr)
        sys.exit(1)

    for filename, content in DEMO_DOCS:
        if already_uploaded(args.host, headers, filename):
            print(f"  skip {filename!r}  (already present)")
        else:
            upload(args.host, headers, filename, content)

    print("\nDone! Open http://localhost:8501 to start querying.")


if __name__ == "__main__":
    main()
