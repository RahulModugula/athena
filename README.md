# athena-verify

**Open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence.**

Drop it on top of any LangChain, LlamaIndex, or raw-LLM pipeline in three lines of code.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

---

## Install

```bash
pip install athena-verify
```

The NLI model (`cross-encoder/nli-deberta-v3-base`, ~1.2 GB) downloads automatically on first use.

## Quick Start

```python
from athena_verify import verify

result = verify(
    question="What is the indemnification cap?",
    answer="The cap is $1M per incident.",
    context=retrieved_chunks,           # list[str] or list[Chunk]
)

result.trust_score          # 0.82 — overall calibrated score
result.sentences            # per-sentence scores + support status
result.unsupported          # sentences that failed verification
result.verification_passed  # True/False
```

That's it. No document ingestion. No chunking. No agents. No database.

## How It Works

Every sentence in the LLM answer is scored against the retrieved context using three signals:

| Signal | What it measures | Weight |
|---|---|---|
| **NLI entailment** | Does the context logically entail this sentence? | 55% |
| **Lexical overlap** | Token-level F1 between sentence and context | 25% |
| **LLM-as-judge** (optional) | Does an LLM say this sentence is supported? | 20% |

The signals are combined into a calibrated trust score (0.0–1.0) per sentence, then aggregated into an overall score.

## Integrations

### LangChain

```python
from athena_verify.integrations.langchain import VerifyingLLM

chain = RetrievalQA.from_llm(
    VerifyingLLM(llm, on_unsupported="flag"),
    retriever=retriever,
)
```

### LlamaIndex

```python
from athena_verify.integrations.llamaindex import VerifyingPostprocessor

engine = index.as_query_engine(
    response_postprocessors=[VerifyingPostprocessor()]
)
```

### OpenAI / Anthropic SDK

```python
from athena_verify import verified_completion

result = verified_completion(
    model="gpt-4o",
    question="What is the indemnification cap?",
    context=retrieved_chunks,
)
```

## API Reference

### `verify(question, answer, context, ...)`

| Parameter | Type | Description |
|---|---|---|
| `question` | `str` | The original question |
| `answer` | `str` | The LLM-generated answer to verify |
| `context` | `list[str] \| list[Chunk]` | Retrieved context chunks |
| `nli_model` | `str` | Cross-encoder model (default: `nli-deberta-v3-base`) |
| `use_llm_judge` | `bool` | Enable LLM-as-judge scoring (default: `False`) |
| `trust_threshold` | `float` | Minimum trust to pass (default: `0.70`) |

Returns a `VerificationResult` with:

| Field | Type | Description |
|---|---|---|
| `trust_score` | `float` | Overall calibrated trust score (0.0–1.0) |
| `sentences` | `list[SentenceScore]` | Per-sentence verification details |
| `unsupported` | `list[SentenceScore]` | Sentences that failed verification |
| `supported` | `list[SentenceScore]` | Sentences that passed verification |
| `verification_passed` | `bool` | Whether overall trust ≥ threshold |
| `metadata` | `dict` | Latency, model info, etc. |

## Benchmarks

Benchmarks in progress on RAGTruth, HaluEval, and FActScore. See [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for reproducible runs and reproduction scripts.

## Examples

| Example | Description |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | 5-minute getting started |
| [`examples/langchain_example.py`](examples/langchain_example.py) | LangChain RetrievalQA integration |
| [`examples/llamaindex_example.py`](examples/llamaindex_example.py) | LlamaIndex query engine integration |

## Architecture

```
Answer → Sentence Splitter → [Sentence₁, Sentence₂, ...]
                                    ↓
                        ┌───────────────────────────┐
                        │  For each sentence:        │
                        │  1. NLI entailment vs ctx  │
                        │  2. Lexical overlap vs ctx │
                        │  3. (optional) LLM judge   │
                        │  4. Calibrate → trust      │
                        └───────────────────────────┘
                                    ↓
                    VerificationResult (per-sentence + overall)
```

## Why This Exists

Ragas, DeepEval, TruLens, and LangSmith all do **offline batch evaluation**. Patronus and Galileo do **runtime detection** but are closed-source and paid. Lynx and Vectara HHEM ship **weights, not a runtime layer**.

There was no open-source, RAG-specific, runtime verification layer with sentence-level granularity. Now there is.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). PRs welcome — especially benchmark results, new integrations, and NLI model improvements.

## License

[MIT](LICENSE)
