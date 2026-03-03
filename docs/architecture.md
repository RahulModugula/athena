# Athena RAG — Architecture

## System Overview

Athena is a Retrieval-Augmented Generation (RAG) system designed for high-quality question answering over private document collections. The system combines dense vector search, BM25 sparse retrieval, reciprocal rank fusion, cross-encoder reranking, and a large language model to produce grounded, faithful answers.

The full pipeline has two phases: **ingestion** and **query**.

During ingestion, raw documents are loaded from disk or HTTP, split into chunks using one of three configurable strategies, embedded with a bilingual dense encoder, and stored in a PostgreSQL table backed by the pgvector extension. BM25 statistics are maintained in a parallel table so that both retrieval modes operate over the same corpus.

During query time, the user's question is embedded with the same encoder, a dual retrieval pass fetches candidates from both the dense ANN index and the BM25 inverted index, the two ranked lists are fused with Reciprocal Rank Fusion, the top candidates are reranked by a cross-encoder, and the final context window is assembled and sent to GLM-4 for answer generation.

---

## Component Descriptions

### FastAPI Backend (`backend/app/`)

The backend is an async Python service built on FastAPI. It exposes REST endpoints for document ingestion, querying, health checks, and evaluation. All database access is performed through SQLAlchemy's async engine with connection pooling. Background tasks handle chunking and embedding so that upload requests return immediately.

### pgvector

PostgreSQL with the pgvector extension stores both the chunk text and the 1024-dimensional float vectors produced by bge-m3. An HNSW index on the vector column enables approximate nearest-neighbour search in sub-linear time. Storing vectors in Postgres means retrieval, metadata filtering, and transactional integrity all live in a single system with no additional infrastructure.

### Embeddings — bge-m3

`BAAI/bge-m3` is loaded via the `sentence-transformers` library and run locally. It produces 1024-dimensional embeddings, supports texts up to 8192 tokens, and achieves state-of-the-art scores on multilingual retrieval benchmarks. The same model instance is used for both document encoding at ingestion time and query encoding at retrieval time, ensuring the vector space is consistent.

### Generation — GLM-4

Answer generation is handled by ZhipuAI's GLM-4 model, accessed through the official Python SDK. A structured prompt template separates the retrieved context passages from the user question. Temperature is kept low (0.1) to maximise factual grounding. The model is instructed to cite only information present in the provided context.

### Streamlit Frontend (`streamlit_app/`)

A lightweight Streamlit application provides a browser UI for uploading documents and submitting queries. It communicates with the FastAPI backend over HTTP. The UI displays the generated answer alongside the source chunks and their relevance scores, making retrieval behaviour inspectable.

---

## Ingestion Pipeline

```
Raw Document (PDF / DOCX / TXT / HTML)
        │
        ▼
┌───────────────┐
│  Document     │  Loaders per file type (PyMuPDF, python-docx,
│  Loader       │  BeautifulSoup). Outputs plain text + metadata.
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Chunker     │  Three strategies (Fixed / Recursive / Semantic).
│               │  Produces List[Chunk] with text + position metadata.
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Embedder    │  bge-m3 encodes each chunk to a 1024-dim vector.
│  (bge-m3)     │  Batched inference; GPU-optional.
└───────┬───────┘
        │
        ▼
┌───────────────────────────┐
│  pgvector (PostgreSQL)    │  Chunks + vectors stored in `chunks`
│                           │  table. HNSW index built on vectors.
│  + BM25 statistics table  │  Term frequencies materialised for
│                           │  sparse retrieval.
└───────────────────────────┘
```

---

## Query Pipeline

```
User Query (natural language)
        │
        ▼
┌───────────────┐
│   Embedder    │  Same bge-m3 model encodes the query to a vector.
│  (bge-m3)     │
└───────┬───────┘
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
┌───────────────┐               ┌──────────────────┐
│  Dense Search │               │   BM25 Search    │
│  (pgvector    │               │  (term-frequency │
│   ANN / HNSW) │               │   statistics)    │
│  top-k=50     │               │  top-k=50        │
└───────┬───────┘               └────────┬─────────┘
        │                                │
        └──────────────┬─────────────────┘
                       ▼
              ┌─────────────────┐
              │   RRF Fusion    │  score = Σ 1/(k + rank_i), k=60
              │                 │  Merges both ranked lists.
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Cross-Encoder  │  Reranks top-20 candidates with
              │   Reranker      │  a fine-tuned reranker model.
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Context Window │  Top-5 chunks assembled into prompt.
              │  Assembly       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     GLM-4       │  Generates a grounded answer.
              │  (ZhipuAI API)  │
              └────────┬────────┘
                       │
                       ▼
                Final Answer + Source Citations
```

---

## Design Decisions

### Why pgvector?

pgvector was chosen over dedicated vector databases (Chroma, Qdrant, Weaviate) for three reasons. First, PostgreSQL provides full ACID guarantees, so document ingestion and vector storage are handled in a single transaction — partial uploads cannot leave the index in an inconsistent state. Second, metadata filtering and vector similarity can be combined in a single SQL query using standard `WHERE` clauses and SQL joins, which is awkward or impossible to express natively in most dedicated stores. Third, the HNSW index implementation in pgvector achieves query latency comparable to standalone ANN libraries while eliminating an additional service from the deployment topology.

### Why Hybrid Search with RRF?

Dense retrieval excels at semantic similarity but can miss exact keyword matches, especially for proper nouns, product codes, and technical terms with no paraphrases in the training corpus. BM25 sparse retrieval handles exact matches reliably but fails on semantically equivalent phrasing. Combining both with Reciprocal Rank Fusion (RRF) consistently outperforms either method alone across document types. The RRF formula `score = 1/(k + rank)` with `k=60` is robust to score-scale differences between the two systems and requires no tuned interpolation weight.

### Why Cross-Encoder Reranking?

Bi-encoder retrieval (embedding model + ANN index) is fast but scores query–document pairs independently, missing fine-grained interaction signals. A cross-encoder reads the query and each candidate passage jointly, producing a much more accurate relevance score at the cost of higher latency. The two-stage design (fast ANN retrieval of 50 candidates, then cross-encoder reranking to top 5) balances precision and throughput: the expensive cross-encoder only runs on a small candidate set.

### Why bge-m3?

`BAAI/bge-m3` achieves top scores on the MTEB multilingual retrieval benchmark. Its 1024-dimensional output provides a good capacity–speed tradeoff. The 8192-token context length means longer chunks can be embedded without truncation. Being an open-weight model, it runs locally without API dependency or per-token cost, which matters for high-volume ingestion.

### Why GLM-4?

GLM-4 provides strong instruction-following quality in both English and Chinese, covers the primary use case for Athena, and is accessible via a straightforward Python SDK. Its context window accommodates the assembled retrieval context plus a detailed system prompt without truncation pressure.

### Why Three Chunking Strategies?

Different document types benefit from different segmentation approaches:

- **Fixed-size chunking** splits on token count with overlap. It is predictable and fast, and performs well on structured, uniform documents (legal clauses, data sheets).
- **Recursive character splitting** respects natural text boundaries (paragraphs, sentences) before falling back to character splits. It handles prose documents well and is the baseline recommendation.
- **Semantic chunking** groups sentences by embedding similarity, keeping topically coherent content together. It reduces context fragmentation for long-form documents but has higher ingestion cost.

Benchmarking all three strategies on the target corpus and selecting per document type is built into the evaluation framework.

---

## Evaluation Framework

Athena ships with an evaluation runner (`backend/eval/runner.py`) that measures retrieval and generation quality using the RAGAS framework.

### RAGAS Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Fraction of answer claims that are entailed by the retrieved context. Detects hallucination. |
| **Answer Relevance** | Semantic similarity between the generated answer and the original question. Measures response focus. |
| **Context Precision** | Fraction of retrieved chunks that are actually relevant to the question. Measures retrieval precision. |
| **Context Recall** | Fraction of ground-truth answer content that is covered by the retrieved chunks. Measures retrieval recall. |

The benchmark dataset is a set of (question, ground-truth answer, source document) triples stored in `eval/data/benchmark.jsonl`. The runner ingests the source documents, runs each question through the full pipeline, collects the generated answer and retrieved context, then calls RAGAS to score each sample. Aggregate scores are written to `eval/results/`.

To run the full benchmark:

```bash
python -m eval.runner --benchmark
```

To evaluate a single question interactively:

```bash
python -m eval.runner --query "What is the amortisation period for goodwill?"
```
