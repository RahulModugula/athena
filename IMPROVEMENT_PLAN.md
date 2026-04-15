# Athena Improvement & Publishing Plan

*Created: 2026-04-16. Comprehensive plan covering technical improvements, product roadmap, competitive positioning, and go-to-market strategy.*

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Competitive Landscape](#competitive-landscape)
3. [Critical Fixes (Pre-Launch Blockers)](#critical-fixes)
4. [Product Improvements](#product-improvements)
5. [Technical Roadmap](#technical-roadmap)
6. [Go-to-Market & GitHub Stars Strategy](#go-to-market-strategy)
7. [Publishing Plan](#publishing-plan)
8. [Success Metrics](#success-metrics)
9. [Reference Links](#reference-links)

---

## Executive Summary

Athena (`athena-verify`) is positioned in a **genuinely open niche**: the only open-source, RAG-specific, runtime hallucination verification layer with sentence-level granularity.

**The problem is real.** Every AI engineer with a RAG in production has experienced hallucinations. Ragas, DeepEval, TruLens, and LangSmith all do offline batch evaluation. Patronus and Galileo do runtime detection but are closed-source and paid. Lynx and Vectara HHEM ship weights, not a runtime layer. Nobody owns the open-source inline RAG verification guardrail.

**This plan covers four pillars:**

1. **Critical Fixes** — Blocking liabilities that damage credibility
2. **Product Improvements** — Features users actually need, prioritized by impact
3. **Go-to-Market Strategy** — How to get GitHub stars and community adoption
4. **Publishing Plan** — Detailed launch sequence with channels, timing, and messaging

---

## Competitive Landscape

### Top RAG Frameworks & Libraries (by GitHub Stars, April 2026)

| Project | Stars | Language | Category | URL |
|---------|-------|----------|----------|-----|
| **Dify** | 138k | TypeScript/Python | Agentic workflow platform | https://github.com/langgenius/dify |
| **LangChain** | 134k | Python | Agent engineering platform | https://github.com/langchain-ai/langchain |
| **Open-WebUI** | 132k | Python | AI interface / chat UI | https://github.com/open-webui/open-webui |
| **RAGFlow** | 78.3k | Python | Dedicated RAG engine | https://github.com/infiniflow/ragflow |
| **Pathway** | 60k | Python/Jupyter | Real-time RAG pipelines | https://github.com/pathwaycom/pathway |
| **Anything-LLM** | 58.4k | JavaScript | All-in-one AI productivity | https://github.com/Mintplex-Labs/anything-llm |
| **Mem0** | 53.2k | Python | AI memory layer | https://github.com/mem0ai/mem0 |
| **Flowise** | 52k | TypeScript | Visual AI agent builder | https://github.com/FlowiseAI/Flowise |
| **LlamaIndex** | 48.6k | Python | Document agent / OCR platform | https://github.com/run-llama/llama_index |
| **Milvus** | 43.8k | Go | Vector database | https://github.com/milvus-io/milvus |
| **Quivr** | 39.1k | Python | Opinionated RAG framework | https://github.com/QuivrHQ/quivr |
| **Khoj** | 34.1k | Python | Self-hosted AI second brain | https://github.com/khoj-ai/khoj |
| **Microsoft GraphRAG** | 32.3k | Python | Graph-based RAG | https://github.com/microsoft/graphrag |
| **Haystack** | 24.9k | Python | AI orchestration framework | https://github.com/deepset-ai/haystack |

**Key insight:** The top-starred repos aren't pure RAG frameworks — they're **application platforms** (Dify, Open-WebUI, Anything-LLM) or **broad agent frameworks** (LangChain). Dedicated RAG tools like RAGFlow (78.3k) and LlamaIndex (48.6k) are further down. The market values end-to-end solutions over infrastructure primitives.

### Direct Competitors (Verification / Evaluation Space)

| Tool | Type | Open Source? | RAG-Specific? | Runtime? | URL | Gap Athena Fills |
|---|---|---|---|---|---|---|
| **Ragas** | Offline eval framework | Yes | Yes | No (batch) | https://github.com/explodinggradients/ragas | Athena runs inline, in real-time |
| **DeepEval** | Offline eval framework | Yes | Yes | No | https://github.com/confident-ai/deepeval | Same |
| **TruLens** | Offline eval framework | Yes | Yes | No | https://github.com/truera/trulens | Same |
| **Phoenix (Arize)** | Observability + eval | Yes | Partial | No | https://github.com/Arize-ai/phoenix | Same |
| **LangSmith** | Eval + tracing | No (freemium) | Yes | No | https://github.com/langsmith | Same |
| **Patronus** | Runtime detection | No (paid API) | Yes | Yes | https://www.patronus.ai | Athena is open-source |
| **Galileo** | Runtime detection | No (paid API) | Yes | Yes | https://www.rungalileo.io | Same |
| **Lynx-8B** | Weights only | Yes | Yes | No runtime | https://github.com/PatronusAI/Lynx | Athena wraps this into a usable layer |
| **Vectara HHEM** | Weights + metric | Yes | Yes | No runtime | https://github.com/vectara/hallucination_evaluation_model | Same |
| **Guardrails AI** | Runtime guardrail framework | Yes | No (schema/safety focus) | Yes | https://github.com/guardrails-ai/guardrails | Not RAG-hallucination specific |

**Athena's unique position:** The only open-source, RAG-specific, runtime verification layer with sentence-level granularity. This is a real, defensible niche.

### What Users Complain About Most (Across All RAG Tools)

These are opportunities for Athena to address:

1. **"My RAG gives bad answers"** — The #1 complaint across r/LocalLLaMA, r/LangChain, r/MachineLearning. Users have no systematic way to know *which* sentences are wrong.
2. **"Over-abstraction & complexity"** — LangChain's most common criticism. Users want something simple that does one thing well.
3. **"Poor documentation & API churn"** — All major frameworks suffer from this. Clean, stable API is a differentiator.
4. **"No good RAG evaluation"** — Users struggle with chunking strategy, embedding model, retrieval tuning. No framework guides them.
5. **"Production readiness gaps"** — Streaming is inconsistent, error handling is fragile, memory management is poor.
6. **"Vendor lock-in"** — LangChain↔LangSmith, LlamaIndex↔LlamaParse. Users want independence.
7. **"Cost"** — GraphRAG is expensive. No framework helps minimize LLM API costs.

### Emerging RAG Trends (2025-2026)

| Trend | Description | Athena Opportunity |
|---|---|---|
| **Agentic RAG** | Pipelines that autonomously decide when/what to retrieve | Verify agent decisions in real-time |
| **GraphRAG** | Knowledge graph extraction for holistic understanding | Verify cross-document claims |
| **Multimodal RAG** | Images, tables, charts, video, audio | Future: verify multimodal claims |
| **Context Engineering** | Systematic context assembly beyond prompt engineering | Athena IS context verification |
| **Real-time / Streaming RAG** | Always-up-to-date RAG from live data | Must support streaming output |
| **RAG Evaluation & Observability** | RAGAS, TruLens, LangSmith gaining traction | Complementary, not competing |
| **MCP Integration** | Model Context Protocol for universal connectors | Add MCP tool for verification |

### Gaps in the Current RAG Landscape

| Gap | Description | Athena Opportunity |
|---|---|---|
| **No batteries-included evaluation** | Every RAG project reinvents evaluation | Integrated quality metrics in every call |
| **Terrible developer onboarding** | Getting a "good" RAG takes weeks | 30-second path to verified answers |
| **No good structured data RAG** | Tables, databases, APIs are second-class | Future: verify structured data claims |
| **Hybrid search is still hard** | BM25 + semantic + graph is complex | Not our problem — we verify whatever comes out |
| **Privacy/security-first RAG** | Few tools support local/offline models | Local NLI model = no cloud API required |
| **Cost optimization** | No framework minimizes LLM costs | Verify locally, only call LLM for borderline |
| **Lightweight / embeddable** | Most frameworks are heavy (LangChain = 100+ deps) | Athena is ~1200 LOC, minimal deps |

---

## Critical Fixes

These are **blocking liabilities** that must be fixed before any new work or launch.

### 1. XSS in Widget (HIGH)

- **File:** `widget/src/widget.ts:184, 198`
- **Issue:** `renderAnswer` injects server-provided data (`data.answer`, `src.title`, `src.snippet`) via `innerHTML`. An attacker-controlled API response could inject arbitrary JavaScript.
- **Fix:** Use `textContent` for dynamic content or use DOMPurify sanitizer.
- **Note:** If widget is archived (per STRATEGY.md), this becomes moot.

### 2. Blocking Sync NLI on Event Loop (HIGH)

- **File:** `backend/app/verification/verifier.py:165`
- **Issue:** `batch_compute_entailment_async` is documented as "for now, just call the sync version" and calls synchronous `batch_compute_entailment`, which runs CrossEncoder model on the event loop and blocks all async operations during inference.
- **Fix:** Use `asyncio.to_thread()` or `loop.run_in_executor()` for the sync NLI call.
- **Status:** Fixed in `athena_verify/core.py` with `verify_async()` using thread pool.

### 3. Missing Tenant Isolation (HIGH)

- **File:** `backend/app/api/routes.py:316-343` (`/search`), `routes.py:418-439` (`/eval/results`), `routes.py:346-415` (`/eval/run`)
- **Issue:** These endpoints do not extract or apply `tenant_id`, meaning any authenticated tenant can access all documents and eval results across tenants.
- **Note:** If backend is archived, this becomes moot.

### 4. LLM Client Re-Instantiation (MEDIUM)

- **File:** `athena_verify/llm_judge.py:56, 81`
- **Issue:** `OpenAIJudge.complete()` and `AnthropicJudge.complete()` create a **new client instance on every call** (`OpenAI(api_key=...)`, `anthropic.Anthropic(api_key=...)`). This creates unnecessary overhead and connection churn.
- **Fix:** Cache client instances on the class or module level.

### 5. No LLM Timeout (MEDIUM)

- **Files:** `backend/app/services/document_service.py:59`, `backend/app/agents/llm.py:35`, `backend/app/generation/chain.py:42`
- **Issue:** `timeout=None` on Anthropic LLM clients means a stalled API call will hang indefinitely.
- **Fix:** Set reasonable default timeout (e.g., 60 seconds) for all LLM client instantiations.

### 6. Regex Injection in Neo4j Queries (MEDIUM)

- **File:** `backend/app/graph/store.py:92`
- **Issue:** `pattern=f"(?i).*{query}.*"` — user-provided `query` is interpolated directly into a Cypher regex. A malicious regex could cause ReDoS.
- **Fix:** Escape special regex characters in `query` before interpolation.
- **Note:** If backend is archived, this becomes moot.

### 7. Redis KEYS Command (MEDIUM)

- **File:** `backend/app/services/cache.py:55`
- **Issue:** `await self._client.keys(pattern)` — `KEYS` is O(N) over all keys and blocks Redis.
- **Fix:** Use `SCAN` iterator instead.
- **Note:** If backend is archived, this becomes moot.

### Additional Code Quality Issues

| Issue | Count | Files Affected |
|---|---|---|
| Broad `except Exception` with silent fallback | 35+ occurrences | `graph/store.py`, `services/cache.py`, `services/document_service.py`, `graph/extractor.py`, `api/routes.py`, `agents/writer.py`, `llm_judge.py` |
| Missing return type annotations | 56 functions | Tests, examples, benchmarks |
| Bare `list` without type parameters | 3 occurrences | `bm25_search.py`, `vector_search.py`, `hybrid.py` |
| `Any` used where specific types exist | 4 occurrences | `document_service.py`, `agents/state.py`, `graph/store.py` |
| Hardcoded `localhost:8000` URLs | 10 occurrences | All Streamlit pages, eval runners, seed scripts |
| Hardcoded database credentials | 3 occurrences | `config.py`, `docker-compose.yml` |
| Hardcoded magic numbers | 8 occurrences | `routes.py` (50MB upload, 200 char truncation), `document_service.py` (batch 32, 3 chunks, 8000 chars), `hybrid.py` (5x over-fetch), `verifier_node.py` (0.7/0.3 thresholds) |
| Duplicated code patterns | 6 major patterns | LLM factories, tenant getters, document batch-loading, state initialization, localhost URLs, search function signatures |
| Commented-out code | 15 lines | `examples/` files, `langchain.py` integration |
| Missing database indexes | 5 tables | `chunks(tenant_id, document_id)`, `documents(tenant_id)`, `documents(content_hash, tenant_id)`, `queries(tenant_id, created_at)`, `eval_runs(created_at)` |
| N+1 Neo4j queries | 2 methods | `upsert_entities`, `upsert_relationships` in `graph/store.py` |
| Sequential sub-query retrieval | 1 method | `researcher.py:59-68` — could use `asyncio.gather()` |
| Sync embedding/reranking on event loop | 3 methods | `embedder.py`, `reranker.py`, `verifier.py` |

---

## Product Improvements

### A. Must-Have (Before Launch — Week 1-2)

#### 1. Streaming Support

**Why:** Production RAG systems stream tokens. If Athena can't verify streaming output, it's dead on arrival for real production use.

**What to build:**

```python
from athena_verify import verify_stream

async for result in verify_stream(
    question="What is the cap?",
    answer_stream=llm_stream,  # async iterator of tokens
    context=retrieved_chunks,
):
    # result.trust_score updates as each sentence completes
    # result.sentences grows incrementally
    pass
```

**Implementation:**
- Buffer incoming tokens until a sentence boundary (`.`, `!`, `?`, `\n`)
- Run verification on each complete sentence
- Yield incremental results
- Final result includes full verification

**Impact:** Critical — enables production adoption.

---

#### 2. Lightweight NLI Fallback

**Why:** The 1.2GB DeBERTa model (`cross-encoder/nli-deberta-v3-base`) is a dealbreaker for quick evaluation and constrained environments.

**What to build:**

```python
result = verify(
    question="What is the cap?",
    answer="The cap is $1M.",
    context=chunks,
    nli_model="lightweight",  # uses smaller model or API
)
```

**Options for lightweight NLI:**

| Option | Size | Accuracy Tradeoff | URL |
|---|---|---|---|
| `vectara/hallucination_evaluation_model` | ~300MB | Slightly lower F1 | https://huggingface.co/vectara/hallucination_evaluation_model |
| `cross-encoder/nli-MiniLM2-L6-H768` | ~80MB | Lower F1, faster | https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768 |
| API-based NLI (Vectara HHEM API) | 0MB | Network dependency | https://huggingface.co/vectara |
| `MoritzLaworr/NLI-deberta-base` | ~450MB | Similar accuracy | https://huggingface.co/MoritzLaworr/NLI-deberta-base |

**Impact:** High — removes the biggest friction point for first-time users.

---

#### 3. Suggested Revision

**Why:** When a sentence is unsupported, telling the user "this is wrong" is only half the solution. Suggesting the correction is the "wow" moment.

**What to build:**

```python
result = verify(
    question="What is the cap?",
    answer="The cap is $5M per incident.",
    context=chunks,
    suggest_revisions=True,  # enables LLM-powered correction
)

for sentence in result.unsupported:
    print(sentence.text)           # "The cap is $5M per incident."
    print(sentence.suggested_fix)  # "The cap is $1M per incident."
```

**Implementation:**
- For each unsupported sentence, call LLM with the sentence + context
- Prompt: "This sentence contradicts or is unsupported by the context. Rewrite it using ONLY information from the context."
- Requires `use_llm_judge=True` or separate `suggest_revisions=True` flag

**Impact:** High — this is the demo moment that makes people share the GIF.

---

#### 4. Batch Verification API

**Why:** Production users need to verify many answers at once (logging, auditing, batch processing).

**What to build:**

```python
from athena_verify import verify_batch

results = verify_batch(
    questions=[q1, q2, q3],
    answers=[a1, a2, a3],
    contexts=[ctx1, ctx2, ctx3],
    batch_size=32,
)

# Or for a single question with many context chunks:
results = verify_batch(
    question="What is the cap?",
    answer="The cap is $1M.",
    context=large_chunk_list,
)
```

**Implementation:**
- Process sentences in batches through NLI model (already supported by CrossEncoder)
- Return list of `VerificationResult`
- Support async variant: `verify_batch_async()`

**Impact:** High — required for production observability and auditing use cases.

---

#### 5. Structured Logging / Observability Output

**Why:** Users want to pipe verification results into their observability stack (LangSmith, Datadog, Langfuse).

**What to build:**

```python
import json

result = verify(...)

# JSON output for logging pipelines
log_entry = result.to_json()
# {"trust_score": 0.82, "sentences": [...], "unsupported": [...], "metadata": {...}}

# OpenTelemetry-compatible span
result.to_otel_span()

# Langfuse-compatible format
result.to_langfuse_trace()
```

**Impact:** Medium — important for production users but not a launch blocker.

---

### B. Differentiators (Post-Launch — Month 2)

#### 6. Citation Extraction with Span Grounding

**Why:** Not just "is this supported?" but "which specific context span supports it?" Return character-level `(start, end)` spans. Nobody does this well in open source.

**What to build:**

```python
result = verify(
    question="What is the cap?",
    answer="The cap is $1M per incident.",
    context=chunks,
    extract_citations=True,
)

for sentence in result.sentences:
    for citation in sentence.citations:
        print(citation.chunk_index)     # which context chunk
        print(citation.start_char)      # character offset in chunk
        print(citation.end_char)        # end offset
        print(citation.text)            # the exact supporting text
```

**Implementation:**
- After NLI scoring, run alignment between sentence tokens and chunk tokens
- Use dynamic programming (similar to squad-style span selection) to find best matching span
- Return `CitationSpan` objects with chunk index + character offsets

**Impact:** Very High — this is unique and extremely useful for production RAG systems.

---

#### 7. Confidence Calibration Display (Rich Output)

**Why:** Visual output showing green/yellow/red per sentence makes the demo GIF 10x more compelling and helps users quickly identify problems.

**What to build:**

```python
from athena_verify import verify, format_result

result = verify(...)

# Terminal output with colors
print(format_result(result, format="terminal"))
# ✅ "The contract was signed on January 15." (trust: 0.95)
# ⚠️ "It covers liability up to $5M." (trust: 0.62)
# ❌ "The vendor is based in Delaware." (trust: 0.21)

# HTML output
html = format_result(result, format="html")
# Returns HTML with color-coded sentences, hover tooltips

# Markdown output
md = format_result(result, format="markdown")
```

**Impact:** High — directly drives the demo GIF quality.

---

#### 8. Multilingual Support

**Why:** Current sentence splitter is English-only (regex-based). RAGFlow's Chinese support drove massive adoption (3-5x audience expansion).

**What to build:**
- Replace regex sentence splitter with `stanza` or `spaCy` multilingual sentence segmentation
- Use multilingual NLI model: `joeddav/xlm-roberta-large-xnli` (supports 100+ languages)
- Or use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**Impact:** High — expands addressable audience by 3-5x.

---

#### 9. Guardrails AI Integration

**Why:** Complementary, not competing. Guardrails AI does schema validation and content safety. Athena does RAG hallucination detection. Together they form a complete guardrail stack.

**What to build:**
- `GuardrailsAIValidator` that wraps Athena's `verify()` as a Guardrails validator
- Blog post: "Using Guardrails AI + Athena for Complete RAG Safety"

**Impact:** Medium — cross-promotion with another OSS project.

---

#### 10. Observability Integrations (Langfuse / Phoenix / LangSmith)

**Why:** Production users already have observability tools. Auto-exporting verification results reduces integration friction.

**What to build:**

```python
from athena_verify import verify
from athena_verify.integrations.langfuse import LangfuseCallback

result = verify(
    ...,
    callbacks=[LangfuseCallback(public_key="...", secret_key="...")],
)
```

**Supported targets:**
- Langfuse: https://github.com/langfuse/langfuse
- Phoenix (Arize): https://github.com/Arize-ai/phoenix
- LangSmith: https://github.com/langsmith (API-based)
- OpenTelemetry: Generic span export

**Impact:** Medium — important for enterprise adoption.

---

### C. Moonshots (Month 3+)

#### 11. Self-Healing RAG Loop

**Why:** When verification fails, auto-re-retrieve and regenerate. Your backend already has this in the LangGraph agent pipeline — extract it as a library feature.

**What to build:**

```python
from athena_verify import verified_rag

result = await verified_rag(
    question="What is the cap?",
    retriever=my_retriever,
    llm=my_llm,
    max_retries=3,
    trust_threshold=0.7,
)
# Automatically re-retrieves and regenerates if verification fails
```

**Impact:** Very High — this is the "holy grail" of RAG reliability.

---

#### 12. Agentic Verification

**Why:** Let an LLM agent decide *how* to verify (which signals, which thresholds) based on the domain and question type.

**What to build:**
- Domain detection (legal, medical, financial, general)
- Adaptive signal weighting based on domain
- Adaptive threshold selection
- Optional multi-pass verification for high-stakes domains

**Impact:** High — pushes the state of the art.

---

#### 13. Public Hallucination Leaderboard

**Why:** A public, community-contributed benchmark comparing RAG systems drives SEO, recurring visits, and positions Athena as the authority in the space.

**What to build:**
- Static site (Next.js or MkDocs) at `athena-leaderboard.dev` or GitHub Pages
- Community-submitted benchmark results
- Automated CI pipeline to re-run benchmarks
- Comparison tables: Athena vs Ragas vs Lynx vs HHEM vs GPT-4-as-judge

**Impact:** High — ongoing traffic and authority building.

---

## Technical Roadmap

### Week 0 — Immediate Fixes (This Week)

| Task | Priority | File(s) | Effort |
|---|---|---|---|
| Archive legacy code to `legacy/full-stack` branch | Critical | All backend, streamlit, widget, k8s | 1 hour |
| Fix LLM client re-instantiation | High | `athena_verify/llm_judge.py` | 30 min |
| Add timeouts to all LLM client calls | High | `llm_judge.py`, `core.py` | 30 min |
| Clean up commented-out code | Low | `examples/`, `integrations/` | 30 min |
| Verify all existing tests pass | Critical | `tests/` | 15 min |

### Week 1 — Core Improvements

| Task | Priority | Effort |
|---|---|---|
| Build `suggested_revision` feature | High | 1 day |
| Add lightweight NLI fallback option | High | 1 day |
| Implement batch verification API | High | 4 hours |
| Add JSON logging output option | Medium | 2 hours |
| Build rich output formatter (terminal + HTML) | Medium | 4 hours |
| Create demo GIF from rich output | High | 2 hours |
| Update README with demo GIF at top | High | 1 hour |

### Week 2 — Real Benchmarks

| Task | Priority | Effort |
|---|---|---|
| Run RAGTruth QA subset benchmark | Critical | 1 day |
| Run HaluEval QA subset benchmark | Critical | 1 day |
| Run FActScore benchmark | High | 1 day |
| Compare against Ragas faithfulness baseline | Critical | 4 hours |
| Compare against GPT-4-as-judge baseline | High | 4 hours |
| Publish reproducible results to `benchmarks/RESULTS.md` | Critical | 2 hours |
| Update README with real numbers | Critical | 1 hour |

**Evaluation protocol:**

1. Load benchmark dataset and run Athena's `verify()` on each `(question, answer, context)` triple
2. Compare Athena's `unsupported` classification against gold-standard hallucination labels
3. Compute precision, recall, F1 on hallucination detection
4. Compute calibration (ECE — Expected Calibration Error)
5. Measure latency (p50/p95) and cost per 1K sentences
6. Run the same evaluation for each baseline using their standard APIs
7. All scripts must be deterministic given the same model weights (set seeds, document GPU/CPU)

**Success gate:** Athena beats Ragas on F1 by at least 5 points on 2 of 3 benchmarks. If it doesn't, fix the verifier before launching.

**Benchmark datasets:**

| Dataset | Size | Focus | URL |
|---|---|---|---|
| RAGTruth | 18K annotated hallucinations | Summarization, QA, data-to-text | https://github.com/FLAME-SCAI/RAGTruth |
| HaluEval | 35K hallucination examples | QA, dialogue, summarization | https://github.com/microsoft/HaluEval |
| FActScore | Long-form factuality | Summarization, biography | https://github.com/shmsw25/FActScore |

**Baselines to compare against:**

| Baseline | Type | URL |
|---|---|---|
| Ragas faithfulness | Offline eval metric | https://github.com/explodinggradients/ragas |
| Lynx-8B | Weights only | https://github.com/PatronusAI/Lynx |
| Vectara HHEM | Weights + metric | https://huggingface.co/vectara/hallucination_evaluation_model |
| GPT-4-as-judge | LLM-as-judge | OpenAI API |
| DeepEval faithfulness | Offline eval metric | https://github.com/confident-ai/deepeval |

### Week 3 — Demo & Documentation

| Task | Priority | Effort |
|---|---|---|
| Build Colab notebook (3 cells, zero-config) | Critical | 4 hours |
| Build HF Spaces hosted demo | High | 1 day |
| Create MkDocs documentation site | High | 1 day |
| Build citation extraction feature | High | 1 day |
| Pre-launch checklist verification | Critical | 2 hours |
| Final README polish | Critical | 2 hours |

**Colab notebook structure:**

```python
# Cell 1: Install
!pip install athena-verify

# Cell 2: Paste API key (optional, only for LLM-judge)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# Cell 3: Run verification
from athena_verify import verify

result = verify(
    question="What is the indemnification cap?",
    answer="The cap is $5M per incident.",
    context=[
        "The indemnification cap is set at $1,000,000 per incident.",
        "The agreement was signed on January 15, 2024.",
    ],
)

print(f"Trust Score: {result.trust_score}")
print(f"Passed: {result.verification_passed}")
for s in result.sentences:
    print(f"  {s.support_status.value}: {s.text} (trust: {s.trust_score:.2f})")
```

**MkDocs documentation structure:**

```
docs/
├── index.md           # Landing page
├── quickstart.md      # 5-minute guide
├── api.md             # Full API reference
├── integrations.md    # LangChain, LlamaIndex, SDK guides
├── benchmarks.md      # Real benchmark results
├── how-it-works.md    # Technical deep dive
├── contributing.md    # Contribution guide
└── changelog.md       # Version history
```

### Week 4 — Launch

See [Publishing Plan](#publishing-plan) below for the detailed launch sequence.

### Month 2 — Growth Features

| Task | Priority |
|---|---|
| Add streaming support (`verify_stream`) | Critical |
| Add multilingual NLI support | High |
| Build citation extraction with span grounding | High |
| Add Langfuse/Phoenix observability integration | Medium |
| Build Guardrails AI integration | Medium |
| Create YouTube tutorials (3 videos) | High |

### Month 3 — Advanced Features

| Task | Priority |
|---|---|
| Build self-healing RAG loop (`verified_rag`) | High |
| Build agentic verification (adaptive thresholds) | Medium |
| Launch public hallucination leaderboard | Medium |
| Build comparison blog posts (vs Ragas, vs TruLens) | High |

---

## Go-to-Market Strategy

### Phase 1: Pre-Launch (Weeks 1-3)

#### Content Strategy (Do in Parallel with Coding)

**1. Build in Public on X/Twitter**

Post daily progress updates. Example threads:

- "Day 1: Building an open-source RAG hallucination detector. The problem: your RAG confidently makes stuff up and you can't ship it to customers. Here's my approach..."
- "Day 5: Just ran my first real benchmark. Athena catches [X]% of hallucinations on RAGTruth. Here's what I learned..."
- "Day 12: The demo is ready. 3 lines of code to verify any RAG answer sentence-by-sentence. [GIF]"

**2. Write 3 Blog Posts** (publish on dev.to + personal blog + Medium)

| Blog Post | Angle | Why |
|---|---|---|
| "Why RAG Hallucinates: A Technical Deep Dive" | Educational | SEO magnet — people search "why does RAG hallucinate" |
| "Athena vs Ragas vs TruLens: Runtime vs Offline RAG Evaluation" | Comparison | People search "ragas vs trulens" and discover you |
| "How Sentence-Level NLI Catches RAG Hallucinations" | Technical | Establishes credibility and explains the approach |

**3. Create the Killer Demo GIF**

This is the single most important asset. Structure:

```
Frame 1: RAG answer with a subtle hallucination
         "The indemnification cap is $5M per incident."
                              ^^^
Frame 2: Athena highlights it in red
         ❌ "The indemnification cap is $5M per incident." (trust: 0.21)
Frame 3: Athena suggests the correction in green
         ✅ "The indemnification cap is $1M per incident." (trust: 0.95)
Frame 4: Show the code (3 lines)
         from athena_verify import verify
         result = verify(question=q, answer=a, context=ctx)
```

**GIF specs:**
- Under 5 MB
- Loops cleanly
- Under 30 seconds
- Shows at the very top of README (before any text)

**4. README as Landing Page**

Current README is good. Enhance with:

- **Demo GIF** at the very top (before "Install")
- **Badges:** PyPI version, downloads, license, CI status, Python versions
- **1-command quickstart** that actually works
- **Real benchmark numbers** (even if just on one dataset)
- **Comparison table** vs Ragas, TruLens, Patronus
- **"Why This Exists"** section (already have — good)
- Keep under 200 lines — scannable on mobile

**5. GitHub Topics & SEO**

Add these GitHub topics to the repository:
`rag`, `hallucination`, `hallucination-detection`, `nli`, `guardrail`, `llm`, `langchain`, `llamaindex`, `verification`, `ai-safety`, `fact-checking`, `retrieval-augmented-generation`, `natural-language-inference`, `sentence-transformers`

**6. Seed Early Community**

- Create Discord server with channels: `#general`, `#help`, `#showcase`, `#benchmarks`, `#contributors`
- Invite 20-30 AI engineers from your network as early testers
- Ask them to run the Colab notebook and give feedback
- Fix any issues they find before launch

### Phase 2: Launch (Week 4)

#### Launch Day Sequence (Tuesday, 8 AM PT)

| Time (PT) | Action | Channel | Details |
|---|---|---|---|
| **8:00 AM** | Show HN post | https://news.ycombinator.com/showhn.html | Title: "Show HN: Athena – open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence" |
| **9:00 AM** | Reddit post | https://reddit.com/r/LocalLLaMA | Title: "I got tired of RAG lying. Built a sentence-level hallucination detector — [X]% F1 on RAGTruth" |
| **10:00 AM** | X/Twitter thread | Your handle | Thread: problem → solution → GIF → real numbers → Colab link → GitHub link |
| **11:00 AM** | LinkedIn post | Your profile | Professional angle: "Excited to open-source Athena..." |
| **12:00 PM** | Dev.to cross-post | https://dev.to | "Why your RAG is hallucinating and how sentence-level NLI catches it" |
| **All day** | Respond to every comment | All channels | First 24 hours — respond within 30 minutes |

**HN Show Post Template:**

```
Show HN: Athena – Open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence

Every RAG system hallucinates. Ragas and TruLens tell you about it after the fact. 
Patronus and Galileo charge you for it. Lynx and HHEM give you weights, not a runtime layer.

Athena wraps any LLM answer with per-sentence NLI entailment, lexical overlap, 
and optional LLM-ensemble scoring. Three lines of code:

    from athena_verify import verify
    result = verify(question=q, answer=a, context=ctx)
    print(result.trust_score, result.unsupported)

Results on RAGTruth: [X]% F1 (vs Ragas [Y]%). 
Try it now: [Colab link]

Built this because my RAG kept confidently making stuff up and I couldn't find 
an open-source tool to catch it at runtime.
```

**Reddit Post Template (r/LocalLLaMA):**

```
Title: I got tired of RAG lying. Built an open-source sentence-level hallucination detector — [X]% F1 on RAGTruth.

[GIF at top showing red/green highlighting]

My RAG kept confidently making stuff up. Every evaluation tool I found was 
either offline (Ragas, TruLens) or paid (Patronus, Galileo). So I built Athena.

It runs inline, in real-time, on any RAG pipeline:
- Per-sentence NLI entailment scoring (DeBERTa cross-encoder)
- Token-level lexical overlap
- Optional LLM-as-judge for borderline cases
- Calibrated trust scores (0.0-1.0)
- Auto-suggested corrections for unsupported sentences

pip install athena-verify

Benchmarks: [X]% F1 on RAGTruth QA subset.
Colab: [link] | GitHub: [link]

Happy to answer questions about the approach.
```

#### Day 2 — Extended Launch

| Time | Channel | Angle |
|---|---|---|
| Morning | r/LangChain | "Drop-in verification for your LangChain RAG" |
| Morning | r/MachineLearning | "Project: Athena — open-source runtime RAG hallucination detection" |
| Afternoon | r/ChatGPTCoding | "Verify your RAG answers in 3 lines of code" |
| Evening | r/artificial | Different angle: AI safety / reliability |

#### Day 3 — Technical Content

| Action | Channel | Details |
|---|---|---|
| Technical blog post | dev.to + personal blog | "Why your RAG is hallucinating and how sentence-level NLI catches it" |
| Code walkthrough | X/Twitter thread | Thread explaining the NLI → overlap → calibration pipeline |

#### Day 5 — Direct Outreach

| Action | Channel | Details |
|---|---|---|
| DM 20 AI engineers | X/Twitter + LinkedIn | Personal messages to people who've publicly complained about RAG hallucinations. Not spam — reference their specific complaint, include Colab link. |

**Target people to DM:**
- People who've tweeted about RAG hallucination problems
- People who've asked about Ragas alternatives on Reddit
- People who've written about RAG evaluation challenges
- AI engineers at companies building RAG products

### Phase 3: Sustained Growth (Month 2-3)

#### Content Calendar

| Week | Content | Channel |
|---|---|---|
| Week 5 | "Athena vs Ragas: Runtime vs Offline RAG Evaluation" blog post | dev.to, Medium |
| Week 5 | "How to Add Hallucination Detection to Your LangChain RAG" tutorial | YouTube |
| Week 6 | "Athena + LlamaIndex: Verified RAG in 5 Minutes" tutorial | YouTube |
| Week 6 | "Building a Production RAG with Verification" guide | dev.to |
| Week 7 | "Understanding NLI for Hallucination Detection" deep dive | dev.to |
| Week 7 | Guest on AI podcast (target: Latent Space, Practical AI) | Podcast |
| Week 8 | "Athena vs TruLens vs DeepEval" comparison blog post | dev.to |
| Week 8 | "How [Company] Uses Athena for RAG Reliability" case study | dev.to |

#### Community Building

| Action | Details |
|---|---|
| Discord community | Channels: #general, #help, #showcase, #benchmarks, #contributors |
| "Good first issue" labels | Attract new contributors with easy issues |
| Monthly community call | 30-minute Zoom/Discord call: demos, Q&A, roadmap |
| Contributor recognition | README section thanking contributors |
| Integration gallery | Showcase projects built with Athena |

#### Cross-Promotion Opportunities

| Partner | How | Benefit |
|---|---|---|
| LangChain | Get listed in their integration gallery | Discoverability |
| LlamaIndex | Get listed in their integration gallery | Discoverability |
| Guardrails AI | Co-authored blog post | Shared audience |
| RAGAS | Benchmark comparison (friendly) | Technical credibility |
| Haystack | Integration blog post | Different audience |
| Ollama | Support local model verification | Large community |

---

## Publishing Plan

### Pre-Launch Checklist

Before Week 4 launch, verify EVERY item:

- [ ] `pip install athena-verify` works on a clean Python 3.12 venv
- [ ] `from athena_verify import verify` runs without errors
- [ ] The Colab notebook executes end-to-end without modification (user only pastes API key)
- [ ] `benchmarks/run_ragtruth.py` produces reproducible results
- [ ] README contains **zero fabricated or projected numbers**
- [ ] README loads in under 3 seconds on mobile
- [ ] The GIF is under 5 MB and loops cleanly
- [ ] `legacy/full-stack` branch exists and all deleted code is recoverable
- [ ] LICENSE file is present (MIT)
- [ ] CONTRIBUTING.md is updated for the new scope
- [ ] `pyproject.toml` metadata is correct (description, classifiers, URLs)
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Lint passes: `ruff check athena_verify/ tests/`
- [ ] Type check passes: `mypy athena_verify/`
- [ ] GitHub Actions CI is green
- [ ] Discord server is created and seeded with early testers
- [ ] Blog posts are written (at least draft)
- [ ] Demo GIF is recorded and optimized

### PyPI Publication Steps

```bash
# 1. Install build tools
pip install build twine

# 2. Build the package
python -m build

# 3. Check the package
twine check dist/*

# 4. Upload to TestPyPI first
twine upload --repository testpypi dist/*

# 5. Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ athena-verify

# 6. Upload to PyPI
twine upload dist/*

# 7. Verify
pip install athena-verify
python -c "from athena_verify import verify; print('OK')"
```

### Documentation Site (MkDocs Material)

**Setup:**

```bash
pip install mkdocs-material mkdocstrings-python
mkdocs new docs
```

**Structure:**

```
docs/
├── index.md           # Landing page with GIF and install
├── quickstart.md      # 5-minute getting started
├── api/
│   ├── verify.md      # verify() reference
│   ├── models.md      # VerificationResult, SentenceScore, etc.
│   └── config.md      # Configuration options
├── integrations/
│   ├── langchain.md   # LangChain guide
│   ├── llamaindex.md  # LlamaIndex guide
│   └── sdk.md         # OpenAI/Anthropic SDK guide
├── benchmarks.md      # Real benchmark results with charts
├── how-it-works.md    # Technical deep dive
├── contributing.md    # Contribution guide
└── changelog.md       # Version history
```

**Deploy to GitHub Pages:**

```bash
mkdocs gh-deploy
```

**Custom domain** (optional): `athena-verify.dev` or `athena.modugula.dev`

### HF Spaces Hosted Demo

**Setup:**

1. Create a Gradio app that accepts:
   - Question (text input)
   - Answer (text area)
   - Context (text area, one chunk per line)
2. Run `verify()` and display:
   - Overall trust score
   - Per-sentence color-coded output
   - Unsupported sentences highlighted
3. Include example inputs pre-loaded

**File: `app.py`**

```python
import gradio as gr
from athena_verify import verify

def verify_answer(question, answer, context_text):
    context = [c.strip() for c in context_text.split("\n") if c.strip()]
    result = verify(question=question, answer=answer, context=context)
    # Format output...
    return formatted_output

demo = gr.Interface(
    fn=verify_answer,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Textbox(label="Answer to Verify"),
        gr.Textbox(label="Context (one chunk per line)", lines=5),
    ],
    outputs=gr.HTML(label="Verification Result"),
    examples=[
        ["What is the indemnification cap?",
         "The cap is $5M per incident.",
         "The indemnification cap is set at $1,000,000 per incident.\nThe agreement was signed on January 15, 2024."],
    ],
    title="Athena — RAG Hallucination Detector",
    description="Verify any RAG answer against retrieved context. Get sentence-level trust scores.",
)

demo.launch()
```

### Colab Notebook

**File: `notebooks/athena_quickstart.ipynb`**

Structure (3 cells):

```python
# Cell 1: Install (auto-runs)
!pip install athena-verify

# Cell 2: Optional — set API key for LLM-judge (skip for NLI-only)
import os
os.environ["OPENAI_API_KEY"] = ""  # Paste your key here (optional)

# Cell 3: Run verification
from athena_verify import verify

result = verify(
    question="What is the indemnification cap?",
    answer="The cap is $5M per incident and applies to all claims.",
    context=[
        "The indemnification cap is set at $1,000,000 per incident.",
        "The cap applies only to third-party claims.",
        "The agreement was signed on January 15, 2024.",
    ],
)

print(f"Overall Trust Score: {result.trust_score:.2f}")
print(f"Verification Passed: {result.verification_passed}")
print()
for s in result.sentences:
    icon = "✅" if s.trust_score >= 0.7 else "❌"
    print(f"{icon} {s.text} (trust: {s.trust_score:.2f})")
if result.unsupported:
    print(f"\n⚠️  {len(result.unsupported)} unsupported sentence(s) detected!")
```

---

## Success Metrics

### 4-Week Targets (Launch)

| Metric | Target |
|---|---|
| GitHub stars | 500 |
| PyPI weekly downloads | 200 |
| HN front page | Yes |
| r/LocalLLaMA top-10 of day | Yes |
| Real benchmark numbers in README | Yes |
| Inbound hiring DMs | 3 |
| Community contributors | 1 |
| Discord members | 50 |

### 12-Week Targets (Growth)

| Metric | Target |
|---|---|
| GitHub stars | 3,000 |
| PyPI weekly downloads | 2,000 |
| Community contributors | 5 |
| Discord members | 200 |
| Integration partners | 2 (LangChain + LlamaIndex galleries) |
| Blog posts published | 6 |
| YouTube tutorials | 3 |

### 6-Month Targets (Scale)

| Metric | Target |
|---|---|
| GitHub stars | 10,000 |
| PyPI weekly downloads | 10,000 |
| Community contributors | 15 |
| Integration partners | 5+ |
| Production users (known) | 10+ |
| Conference talk | 1 (AI Engineer Summit, PyData, etc.) |

**If you miss all Week 4 launch metrics → the positioning is wrong, not the execution. Re-evaluate before building more.**

---

## Reference Links

### Competitors

| Tool | URL |
|---|---|
| Ragas | https://github.com/explodinggradients/ragas |
| DeepEval | https://github.com/confident-ai/deepeval |
| TruLens | https://github.com/truera/trulens |
| Phoenix (Arize) | https://github.com/Arize-ai/phoenix |
| LangSmith | https://smith.langchain.com |
| Patronus | https://www.patronus.ai |
| Galileo | https://www.rungalileo.io |
| Lynx | https://github.com/PatronusAI/Lynx |
| Vectara HHEM | https://huggingface.co/vectara/hallucination_evaluation_model |
| Guardrails AI | https://github.com/guardrails-ai/guardrails |
| LangChain | https://github.com/langchain-ai/langchain |
| LlamaIndex | https://github.com/run-llama/llama_index |
| Haystack | https://github.com/deepset-ai/haystack |
| RAGFlow | https://github.com/infiniflow/ragflow |
| Dify | https://github.com/langgenius/dify |
| Open-WebUI | https://github.com/open-webui/open-webui |
| Microsoft GraphRAG | https://github.com/microsoft/graphrag |

### Benchmarks

| Dataset | URL |
|---|---|
| RAGTruth | https://github.com/FLAME-SCAI/RAGTruth |
| HaluEval | https://github.com/microsoft/HaluEval |
| FActScore | https://github.com/shmsw25/FActScore |
| LegalBench-RAG | https://github.com/actuallynotalegalbench/legalbench-rag |

### NLI Models

| Model | Size | URL |
|---|---|---|
| cross-encoder/nli-deberta-v3-base (default) | ~1.2 GB | https://huggingface.co/cross-encoder/nli-deberta-v3-base |
| vectara/hallucination_evaluation_model (lightweight) | ~300 MB | https://huggingface.co/vectara/hallucination_evaluation_model |
| cross-encoder/nli-MiniLM2-L6-H768 (tiny) | ~80 MB | https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768 |
| joeddav/xlm-roberta-large-xnli (multilingual) | ~1.1 GB | https://huggingface.co/joeddav/xlm-roberta-large-xnli |

### Launch Channels

| Channel | URL |
|---|---|
| Hacker News Show HN | https://news.ycombinator.com/submit |
| r/LocalLLaMA | https://reddit.com/r/LocalLLaMA |
| r/LangChain | https://reddit.com/r/LangChain |
| r/MachineLearning | https://reddit.com/r/MachineLearning |
| r/ChatGPTCoding | https://reddit.com/r/ChatGPTCoding |
| r/artificial | https://reddit.com/r/artificial |
| dev.to | https://dev.to |
| Medium | https://medium.com |
| Product Hunt | https://producthunt.com |
| AlternativeTo | https://alternativeto.net |

### Tools & Infrastructure

| Tool | Purpose | URL |
|---|---|---|
| MkDocs Material | Documentation site | https://squidfunk.github.io/mkdocs-material/ |
| Gradio | HF Spaces demo | https://gradio.app |
| Google Colab | Quick-start notebook | https://colab.research.google.com |
| Shields.io | README badges | https://shields.io |
| SimpleAnalytics | README analytics | https://simpleanalytics.com |
| Discord | Community | https://discord.com |

### Learning Resources

| Resource | Topic | URL |
|---|---|---|
| "How Dify Got 138k Stars" | Go-to-market | https://github.com/langgenius/dify |
| "RAGTruth: Benchmark" | Hallucination detection | https://arxiv.org/abs/2401.00396 |
| "FActScore" | Factuality evaluation | https://arxiv.org/abs/2305.14251 |
| "HaluEval" | Hallucination evaluation | https://arxiv.org/abs/2305.11747 |
| "A Survey on Hallucination in LLMs" | Background | https://arxiv.org/abs/2311.05232 |

---

*This plan supersedes `WORLDCLASS_PLAN.md` and supplements `plans/STRATEGY.md`. Execute in order: Critical Fixes → Product Improvements → Real Benchmarks → Demo → Launch → Growth.*
