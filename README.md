# athena-verify

**Open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence.**

Drop it on top of any LangChain, LlamaIndex, or raw-LLM pipeline in three lines:

```python
from athena_verify import verify

result = verify(
    question="What is the indemnification cap?",
    answer="The cap is $1M per incident.",
    context=retrieved_chunks,
)
result.unsupported  # → ["The cap is $1M per incident."]  ← caught!
```

That's it. No document ingestion. No chunking. No agents. No database.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

## How It Works

```
  LLM Answer
      │
      ▼
  Split into sentences
      │
      ▼
  ┌─────────────────────────────────────┐
  │  For each sentence:                 │
  │                                     │
  │  1. NLI entailment vs context   ──► │ score 0.0–1.0
  │  2. Lexical overlap vs context  ──► │ score 0.0–1.0
  │  3. [optional] LLM judge        ──► │ SUPPORTED / UNSUPPORTED
  │  4. Combine → trust score           │
  └──────────────┬──────────────────────┘
                 │
                 ▼
    VerificationResult
    ├─ trust_score: 0.0–1.0
    ├─ supported: [sentences that passed]
    ├─ unsupported: [sentences that failed]   ← flag or filter these
    └─ verification_passed: bool
```

Every sentence is scored independently. **Two modes:**
- **NLI-only** (default): ~20ms per sentence, catches fabricated claims and out-of-context info
- **NLI + LLM judge**: send every sentence to a local LLM for verification — catches number swaps and negation flips with near-perfect accuracy

## Install

```bash
pip install athena-verify
```

The NLI model (DeBERTa-v3, ~1.2 GB) downloads automatically on first use.

For LLM-judge support (optional, local models via LM Studio or API):

```bash
pip install "athena-verify[all]"
```

## Benchmarks

Tested on 100 synthetic cases across 6 hallucination categories (legal, medical, technical, general).

| Category | NLI-only F1 | + LLM-judge F1 |
|----------|-------------|-----------------|
| Fabricated claims | 87.9% | — |
| Out-of-context | 88.9% | — |
| Partial support | 48.3% | — |
| Number substitutions | 29.4% | **93.9%** |
| Subtle contradictions | 23.5% | **100.0%** |

NLI-only latency: **~20ms p50**. LLM-judge latency: **~7.4s/sentence** (local gemma-4-31b-it).

Full results: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)

## How We Compare

| | Athena | Ragas | TruLens | Patronus |
|---|---|---|---|---|
| **Runtime detection** | Yes | Offline eval | Offline eval | Yes |
| **Sentence-level** | Yes | Answer-level | Answer-level | Yes |
| **Open source** | Yes | Yes | Partial | No |
| **Free / local** | Yes | Yes | Yes | No |
| **No external API** | Yes | No (LLM calls) | No (LLM calls) | No |

Ragas and TruLens are great for **offline evaluation**. Athena is the **runtime guardrail** — it catches hallucinations before they reach users, in production, with zero API cost.

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

### OpenAI / Anthropic

```python
from athena_verify import verified_completion

result = verified_completion(
    model="gpt-4o",
    question="What is the indemnification cap?",
    context=retrieved_chunks,
)
```

## API

### `verify(question, answer, context, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | required | The original question |
| `answer` | `str` | required | The LLM-generated answer |
| `context` | `list[str]` | required | Retrieved context chunks |
| `nli_model` | `str` | `nli-deberta-v3-base` | Cross-encoder model |
| `use_llm_judge` | `bool` | `False` | Enable LLM judge for all sentences |
| `trust_threshold` | `float` | `0.70` | Minimum trust to pass |

Returns `VerificationResult` with `trust_score`, `sentences`, `supported`, `unsupported`, and `verification_passed`.

## Examples

| Example | Description |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | 5-minute getting started |
| [`examples/langchain_example.py`](examples/langchain_example.py) | LangChain RetrievalQA |
| [`examples/llamaindex_example.py`](examples/llamaindex_example.py) | LlamaIndex query engine |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). PRs welcome — especially benchmark results, new integrations, and NLI model improvements.

## License

[MIT](LICENSE)
