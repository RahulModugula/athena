# Athena Strategy: Narrow to the Verification Layer

*Supersedes `WORLDCLASS_PLAN.md` and `plans/roadmap.md`. Written 2026-04-13. Revised 2026-04-16.*

---

## One-sentence positioning

> **Athena is the open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence before they reach users — drop it on top of any LangChain, LlamaIndex, or raw-LLM pipeline in three lines of code.**

Not a RAG system. Not an eval harness. A **runtime verification layer.** That is the entire product.

---

## Why this positioning (and not the others)

Three directions were on the table:

| Option | Wedge | Best for | Verdict |
|---|---|---|---|
| **A. Verification layer** | Drop-in runtime guardrail on any RAG | Reddit stars + AI infra hiring (Anthropic, LangChain, Arize, Cursor) | ✅ **Commit** |
| B. Public hallucination leaderboard | Benchmark everyone else runs against | Research hiring | Fold into A as supporting asset |
| C. Legal contract QA vertical | Narrow domain, real buyers | Co-founder / YC | Park for later |

**Why A wins the stated goal** (subreddit traction + senior AI engineer hiring):

1. **The gap is real and open.** Ragas, DeepEval, TruLens, Phoenix, LangSmith all do *offline batch eval*. Patronus and Galileo do *runtime* detection but are closed-source and paid. Lynx and Vectara HHEM ship weights, not a runtime layer. Guardrails AI ships a general-purpose guardrail framework but focuses on output schema validation and content safety — not RAG-specific hallucination detection with citation grounding. Nobody owns the open-source inline RAG verification guardrail.
2. **Every AI engineer with a RAG in prod has this pain.** Distribution is natural — r/LocalLLaMA, r/LangChain, r/MachineLearning, HN, AI Engineer Slack.
3. **It's demoable in 30 seconds.** A Colab cell with a wrong answer → highlighted red → fixed. That's the GIF.
4. **It's the narrow, deep thing a senior hire owns.** "I built the open verification layer for RAG" is a complete sentence at a phone screen.

---

## What's actually wrong with the repo today

Honest diagnosis — fix before any new work:

### 1. Scope is a swamp
The repo is simultaneously: a full RAG system, a LangGraph agent pipeline, an MCP server, a Neo4j knowledge graph, a Streamlit UI, a Kubernetes deployment, an embeddable widget, a multi-tenant API, **and** a verification layer. A reader has no idea what it is in 30 seconds → bounces.

### 2. Benchmark numbers are not real
The README table claiming *"Hallucination Rate: 23% → 5%, 78% reduction"* and *"Precision@1: 2.65% → 18.2%, 6.9× better"* is **not measured on this codebase**. The 2.65% figure is a LegalBench-RAG baseline for a specific poorly-configured vanilla system, not a fair head-to-head with this code. The 5%/23% numbers are projected, not observed. The `WORLDCLASS_PLAN.md` leaderboard table with fabricated numbers for Medical, Research, Financial, and Technical domains is even worse — those benchmarks were never run.

**This is the single biggest liability.** A hiring manager who runs `python -m eval.runner --benchmark` and sees different numbers immediately distrusts the whole project. Every fabricated number in the README must be deleted or re-measured this week.

### 3. No 30-second path
Current onboarding: clone → docker compose → seed scripts → Streamlit → upload PDF → query. That is minutes, not seconds, and requires a local Postgres + embedding model download. Nobody will try it from a Reddit post.

### 4. No identifiable target user
"Enterprise RAG" is not a user. "AI engineer with a LangChain RAG in prod who just had a hallucination embarrass them" is.

---

## Technical audit: what the verification code actually does

Before planning the migration, here is an honest assessment of the existing verification module at `backend/app/verification/`.

### What exists (4 files, ~330 lines)

| File | Purpose | Status |
|---|---|---|
| `models.py` | Pydantic models: `VerifiedSentence`, `VerifiedAnswer`, `CitationSpan` | ✅ Clean, reusable |
| `nli.py` | Cross-encoder NLI (`cross-encoder/nli-deberta-v3-base`) for entailment scoring | ✅ Works, but sync-only |
| `parser.py` | Parse structured JSON from writer agent OR fallback sentence splitting | ⚠️ Coupled to agent output format |
| `verifier.py` | Orchestrate: parse → NLI → lexical overlap → trust score | 🔴 Has bugs, needs rewrite |

### Critical bugs found

**Bug 1 — Lexical overlap is computed against chunk IDs, not chunk content.**

In `verifier.py` line 107:
```python
lexical = _lexical_overlap(sent_text, " ".join(c.get("chunk_id", "") for c in citations))
```

This computes overlap between the sentence and the *UUIDs of cited chunks*, not the actual text. The lexical overlap score is meaningless as implemented. It should compare against the cited text spans recovered from chunks (which the code already extracts into `cited_texts` but then ignores for the overlap computation).

**Bug 2 — No LLM-ensemble scoring exists.**

The README and strategy claim three verification signals (NLI entailment, lexical overlap, LLM-as-judge). Only NLI actually works. Lexical overlap is broken (see above). LLM-as-judge is not implemented anywhere in the codebase. This must be built or the claim must be dropped.

**Bug 3 — Parser expects agent-structured JSON.**

`parse_answer()` tries JSON first (expecting `{"sentences": [{"text": ..., "citations": [...]}]}`), then falls back to regex sentence splitting with no citation recovery. For the standalone library, the parser must accept plain text answers and plain text context — no structured citation spans required.

### What needs to be built for the standalone package

| Component | Current state | What's needed |
|---|---|---|
| Core `verify()` function | Tightly coupled to agent pipeline | Clean function: `(question, answer, context) → VerificationResult` |
| Lexical overlap | Broken (computes against IDs) | Fix to compare sentence vs. context chunks |
| LLM-as-judge scoring | Not implemented | Optional LLM call for borderline cases |
| Trust score calibration | Naive average of 2 scores | Proper weighted ensemble with configurable thresholds |
| `suggested_revision` | Not implemented | LLM-powered correction for unsupported sentences |
| Async support | `batch_compute_entailment_async` is a sync wrapper | True async with `asyncio.to_thread` or thread pool |
| Plain-text context handling | Requires structured chunks with offsets | Accept `list[str]` or `list[Chunk]` |
| Sentence splitting | Regex-based, English-only | Use `nltk` or `spaCy` for robust splitting |

### Dependency analysis

The standalone package should be lightweight:

| Dependency | Weight | Decision |
|---|---|---|
| `sentence-transformers` | ~2 GB (pulls PyTorch) | **Core** — needed for NLI model. Accept this; it runs locally. |
| `pydantic` | Light | **Core** — models |
| `structlog` | Light | **Core** — logging |
| `openai` / `anthropic` | Light | **Optional** — only if LLM-as-judge is enabled |

The NLI model download is a one-time cost (~1.2 GB for `nli-deberta-v3-base`). Document this clearly in the README. Consider offering a lighter fallback (e.g., `vectara/hallucination_evaluation_model` at ~300 MB) as an option.

---

## Target user

| Axis | Value |
|---|---|
| Role | AI engineer / ML engineer / applied AI founder |
| Current stack | LangChain, LlamaIndex, or raw `openai`/`anthropic` SDK with a vector DB |
| Daily pain | "My RAG confidently makes stuff up and I can't ship it to customers" |
| Where they live | r/LocalLLaMA, r/LangChain, r/MachineLearning, Hacker News, AI Engineer Slack, X |
| What they pay in | Attention and GitHub stars first. Pilot deals later. |
| Decision loop | Read README → run `pip install` → see a verified answer in Colab → star repo |

**Every decision below is scored against: "Does it make that user star the repo in the first 60 seconds?"**

---

## The product: what Athena becomes

### Package naming

`athena` is taken on PyPI (by an unrelated project). Options:

| Name | PyPI available? | Import | Verdict |
|---|---|---|---|
| `athena-verify` | Likely yes | `from athena_verify import verify` | ✅ **Recommended** — clear, literal, SEO-friendly |
| `ragverify` | Likely yes | `from ragverify import verify` | Good but less brand connection |
| `truthlens` | Unknown | `from truthlens import verify` | Too abstract |

**Decision: `athena-verify` on PyPI, import as `athena_verify`.** Update the strategy to use this throughout.

### Public API (the whole product)

```python
from athena_verify import verify

result = verify(
    question="What is the indemnification cap?",
    answer="The cap is $1M per incident.",
    context=retrieved_chunks,           # list[str] or list[Chunk]
)

result.trust_score          # 0.0 – 1.0, calibrated
result.sentences            # per-sentence scores + supporting spans
result.unsupported          # sentences that failed verification
result.suggested_revision   # optional: corrected answer (if LLM enabled)
```

**That is it.** No document ingestion. No chunking. No agents. No K8s. No widget. No Streamlit.

### Integrations (thin wrappers, not products)

```python
# LangChain
from athena_verify.integrations.langchain import VerifyingLLM
chain = RetrievalQA.from_llm(VerifyingLLM(llm), retriever=r)

# LlamaIndex
from athena_verify.integrations.llamaindex import VerifyingPostprocessor
engine = index.as_query_engine(response_postprocessors=[VerifyingPostprocessor()])

# OpenAI / Anthropic SDK
from athena_verify import verified_completion
resp = verified_completion(model="gpt-4o", question=q, context=ctx)
```

### What gets deleted from the current repo

| Current component | Fate |
|---|---|
| FastAPI backend, multi-tenant API, widget, Streamlit UI | **Archive to `legacy/` branch, remove from main** |
| LangGraph agents (supervisor/researcher/analyst/writer) | **Archive** |
| Neo4j / graph extraction | **Archive** |
| MCP server | **Archive** |
| K8s manifests | **Delete** |
| Docker Compose (keep a tiny one for Postgres-in-tests only) | **Shrink** |
| Document ingestion / chunking / embedding | **Archive** — verification takes chunks as input, not PDFs |
| Verification package (`backend/app/verification/`) | **Becomes the product — promote to top-level `athena_verify/`** |

The repo shrinks by ~70%. That is the point. A reader reads less and remembers more.

### Target repo structure

```
athena/
├── athena_verify/              # The library
│   ├── __init__.py             # exports verify(), verified_completion()
│   ├── core.py                 # verify() implementation
│   ├── models.py               # VerificationResult, VerifiedSentence, Chunk
│   ├── nli.py                  # NLI entailment scoring
│   ├── overlap.py              # Lexical overlap (fixed)
│   ├── llm_judge.py            # Optional LLM-as-judge scoring
│   ├── calibration.py          # Trust score calibration
│   ├── parser.py               # Sentence splitting (robust)
│   └── integrations/
│       ├── __init__.py
│       ├── langchain.py        # VerifyingLLM
│       └── llamaindex.py       # VerifyingPostprocessor
├── tests/
│   ├── test_verify.py
│   ├── test_nli.py
│   ├── test_overlap.py
│   └── test_integrations/
├── benchmarks/
│   ├── RESULTS.md              # Real, reproducible results
│   ├── run_ragtruth.py
│   ├── run_halueval.py
│   └── run_factscore.py
├── examples/
│   ├── quickstart.py
│   ├── langchain_example.py
│   └── llamaindex_example.py
├── docs/
│   ├── index.md
│   ├── api.md
│   └── benchmarks.md
├── README.md
├── pyproject.toml
└── LICENSE
```

---

## The execution plan

### Week 0 — Immediate fixes (before the 4-week sprint)

These are blocking liabilities that must be fixed before any new work:

- [ ] **Delete fabricated numbers from README.** Remove the "Hallucination Mitigation" table (lines 216-224) and the `WORLDCLASS_PLAN.md` leaderboard (lines 61-81). Replace with: *"Benchmarks in progress — see `benchmarks/` for reproducible runs."*
- [ ] **Fix the lexical overlap bug** in `backend/app/verification/verifier.py` line 107. Change from comparing against chunk IDs to comparing against the recovered cited text.
- [ ] **Branch `legacy/full-stack`** from current `main`. Push and leave it. All current code is preserved.
- [ ] **Update `WORLDCLASS_PLAN.md`** header to mark it as superseded with a link to this document.

### Week 1 — Cut and consolidate

- [ ] On `main`: delete everything listed in the deletion table above. Repo root becomes `athena_verify/`, `tests/`, `examples/`, `benchmarks/`, `README.md`, `pyproject.toml`.
- [ ] Promote `backend/app/verification/` to `athena_verify/` as a clean, pip-installable package.
- [ ] **Rewrite `verify()` to accept `(question, answer, context)` where context is `list[str]`.** No structured citation spans required. The function splits the answer into sentences, pairs each with context chunks, runs NLI, computes fixed lexical overlap, and returns a `VerificationResult`.
- [ ] Fix `parser.py` to handle plain text answers robustly (not just agent-structured JSON).
- [ ] Make `sentence-transformers` the only heavy dependency. All others (`structlog`, `pydantic`) are light.
- [ ] Ship `pip install athena-verify` on TestPyPI.
- [ ] **Deliverable:** a 150-line README with the 5-line install → 3-line usage → 1 screenshot path.

### Week 2 — Real, honest benchmarks

Run *actual* measurements on public data. No projections.

- [ ] **RAGTruth** (Niu et al., 2024) — 18K annotated hallucinations across summarization, QA, data-to-text. Public, peer-reviewed, the canonical hallucination benchmark. Use the QA subset first (most relevant to RAG).
- [ ] **HaluEval** — 35K hallucination examples over QA, dialogue, summarization. Use the QA subset.
- [ ] **FActScore** — long-form factuality benchmark. Use for the summarization use case.
- [ ] Score: Athena vs. Ragas faithfulness vs. Lynx-8B vs. Vectara HHEM vs. GPT-4-as-judge.
- [ ] Report: precision, recall, F1 on hallucination detection, calibration (ECE), latency (p50/p95), cost per 1K sentences.
- [ ] Publish to `benchmarks/RESULTS.md` with reproduction scripts. CI re-runs on PRs.

**Evaluation protocol:**
1. For each benchmark, load the dataset and run Athena's `verify()` on each (question, answer, context) triple.
2. Compare Athena's `unsupported` classification against the gold-standard hallucination labels.
3. Compute precision (of sentences flagged as unsupported, how many are actually hallucinated), recall (of all hallucinated sentences, how many did we catch), and F1.
4. Run the same evaluation for each baseline using their standard APIs.
5. All scripts must be deterministic given the same model weights. Set seeds. Document GPU/CPU used.

**Success gate:** Athena beats Ragas on F1 by at least 5 points on 2 of 3 benchmarks. If it doesn't, fix the verifier before launching.

If it does, *those* are the numbers in the README. Real, citable, reproducible.

**Risk mitigation:** If Athena doesn't beat Ragas on F1, the most likely lever is adding the LLM-as-judge signal for borderline cases (NLI scores 0.3–0.7). Build this as a configurable option in Week 1 so it's available as a fallback.

### Week 3 — The 30-second demo

- [ ] Public Colab notebook: 3 cells. Install, paste your OpenAI key, run → highlighted sentence-level output.
- [ ] 45-second screen recording → GIF → pinned at top of README.
- [ ] Hosted demo on HF Spaces: paste an answer and context, get a verified output. No signup.
- [ ] One-page docs site (MkDocs Material, `athena-verify.dev` or GitHub Pages). Three pages: Quickstart, API, Benchmarks.

### Week 4 — Launch

- [ ] **Tuesday 8 AM PT** — Show HN post. Title: *"Athena — open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence."* Body: the problem, the API, the Colab, the benchmarks.
- [ ] **Same day** — r/LocalLLaMA post with GIF at top. Different angle: "I got tired of RAG lying, so I built a verifier — [X]% F1 vs Ragas's [Y]% on RAGTruth." (Use real numbers from Week 2.)
- [ ] **Day 2** — r/LangChain, r/MachineLearning (Project tag), r/ChatGPTCoding.
- [ ] **Day 3** — Technical writeup on personal blog + dev.to. Title: *"Why your RAG is hallucinating and how sentence-level NLI catches it."*
- [ ] **Day 5** — Targeted outreach: DM 20 AI engineers who've publicly complained about hallucinations on X. Not spam — personal, with the Colab link.

---

## Non-goals (explicitly)

- No ArXiv paper in v1. The 4-week plan beats the 8-week paper plan for stars + hires.
- No production case study yet. That's v2.
- No K8s, no multi-tenant, no widget, no UI.
- No new vector DB, no new chunker, no new embedder.
- No chat interface. This is a library, not an app.
- No fine-tuning or custom model training in v1. Use off-the-shelf NLI models.

---

## Success metrics

| Metric | 4 weeks | 12 weeks |
|---|---|---|
| GitHub stars | 500 | 3,000 |
| PyPI weekly downloads | 200 | 2,000 |
| HN front page | Yes | — |
| r/LocalLLaMA top-10 of day | Yes | — |
| Real benchmark numbers in README | Yes | Yes |
| Inbound hiring DMs | 3 | 15 |
| Community contributors | 1 | 5 |

Miss all of week-4's launch metrics → the positioning is wrong, not the execution. Re-evaluate before building more.

---

## Pre-launch checklist

Before the Week 4 launch, verify every item:

- [ ] `pip install athena-verify` works on a clean Python 3.12 venv
- [ ] `from athena_verify import verify` runs without errors
- [ ] The Colab notebook executes end-to-end without modification (user only pastes API key)
- [ ] `benchmarks/run_ragtruth.py` produces reproducible results
- [ ] README contains zero fabricated or projected numbers
- [ ] README loads in under 3 seconds on mobile
- [ ] The GIF is under 5 MB and loops cleanly
- [ ] `legacy/full-stack` branch exists and all deleted code is recoverable
- [ ] LICENSE file is present (MIT)
- [ ] CONTRIBUTING.md is updated for the new scope
- [ ] `pyproject.toml` metadata is correct (description, classifiers, URLs)

---

## Open questions (resolved)

1. **Package name:** `athena-verify` on PyPI, import as `athena_verify`. Keeps brand connection while being available and descriptive.
2. **License stays MIT.** Open-source credibility is the entire distribution strategy.
3. **Deleting 70% of the code is the point.** The `legacy/full-stack` branch preserves everything. The main branch becomes the product.

---

## Competitive landscape (updated)

| Tool | Type | Open source? | RAG-specific? | Runtime? | Gap |
|---|---|---|---|---|---|
| Ragas | Offline eval framework | Yes | Yes | No batch | Athena runs inline |
| DeepEval | Offline eval framework | Yes | Yes | No | Same |
| TruLens | Offline eval framework | Yes | Yes | No | Same |
| Phoenix (Arize) | Observability + eval | Yes | Partial | No | Same |
| LangSmith | Eval + tracing | No (freemium) | Yes | No | Same |
| Patronus | Runtime detection | No (paid API) | Yes | Yes | Athena is open-source |
| Galileo | Runtime detection | No (paid API) | Yes | Yes | Same |
| Lynx-8B | Weights only | Yes | Yes | No runtime | Athena wraps this |
| Vectara HHEM | Weights + metric | Yes | Yes | No runtime | Same |
| Guardrails AI | Runtime guardrail framework | Yes | No (schema/safety) | Yes | Not RAG-hallucination specific |

**Athena's position:** The only open-source, RAG-specific, runtime verification layer with sentence-level granularity. This is a real, defensible niche.

---

## What this buys the author

**Hiring narrative (phone screen):**

> "Ragas and TruLens are offline eval frameworks. Patronus is a paid API. Lynx is a weights release. Guardrails AI does schema validation, not hallucination detection. There was no open-source runtime verification layer for RAG, so I built one. It wraps any LLM answer with per-sentence NLI entailment, lexical overlap, and optional LLM-ensemble scoring, returns calibrated trust scores in under 400ms, and beats Ragas on RAGTruth F1 by [X] points. [N] GitHub stars, [M] PyPI downloads. Here's the Colab."

That is a complete, specific, verifiable answer. It is the difference between "impressive project" and "we should talk about a role."

---

## Appendix: Migration plan (file-level)

### Files to promote (rewrite into `athena_verify/`)

| Source | Destination | Action |
|---|---|---|
| `backend/app/verification/models.py` | `athena_verify/models.py` | Refactor: add `VerificationResult`, `Chunk`, remove `CitationSpan` dependency |
| `backend/app/verification/nli.py` | `athena_verify/nli.py` | Keep mostly as-is, add proper async with `asyncio.to_thread` |
| `backend/app/verification/parser.py` | `athena_verify/parser.py` | Rewrite: accept plain text, use `nltk.sent_tokenize` or similar |
| `backend/app/verification/verifier.py` | `athena_verify/core.py` | Rewrite: new `verify()` signature, fix lexical overlap bug, add calibration |

### Files to create new

| File | Purpose |
|---|---|
| `athena_verify/overlap.py` | Fixed lexical overlap computation |
| `athena_verify/calibration.py` | Trust score calibration (weighted ensemble, configurable thresholds) |
| `athena_verify/llm_judge.py` | Optional LLM-as-judge for borderline cases |
| `athena_verify/integrations/langchain.py` | `VerifyingLLM` wrapper |
| `athena_verify/integrations/llamaindex.py` | `VerifyingPostprocessor` wrapper |

### Files to archive (move to `legacy/full-stack` branch)

Everything in: `backend/app/agents/`, `backend/app/api/`, `backend/app/generation/`, `backend/app/graph/`, `backend/app/ingestion/`, `backend/app/mcp/`, `backend/app/models/`, `backend/app/observability/`, `backend/app/retrieval/`, `backend/app/services/`, `backend/app/main.py`, `backend/app/config.py`, `backend/app/database.py`, `streamlit_app/`, `widget/`, `k8s/`, `backend/alembic/`, `backend/eval/`, `backend/scripts/`, `datasets/`.
