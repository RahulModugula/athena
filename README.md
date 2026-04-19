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
result.unsupported_texts  # → ["The cap is $1M per incident."]  ← caught!
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
  ┌──────────────────────────────────────────┐
  │  For each answer sentence:               │
  │                                          │
  │  1. Split context into sentences too     │
  │  2. NLI: each ctx sentence vs answer ──► │ max entailment score
  │  3. Lexical overlap vs context       ──► │ token F1 score
  │  4. [optional] LLM judge             ──► │ SUPPORTED / UNSUPPORTED
  │  5. Combine → trust score                │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
    VerificationResult
    ├─ trust_score: 0.0–1.0
    ├─ supported: [sentences that passed]
    ├─ unsupported: [sentences that failed]
    └─ verification_passed: bool
```

**Two modes:**
- **NLI-only** (default): ~20ms per sentence, catches fabricated claims, out-of-context info, number swaps, and negation flips
- **NLI + LLM judge**: send every sentence to a local LLM for additional verification when accuracy is critical

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

Comprehensive evaluation on 100 synthetic cases across 6 hallucination categories (legal, medical, technical, general).

### Per-Category Performance (NLI-only mode)

| Category | Precision | Recall | **F1** |
|----------|-----------|--------|--------|
| **Fabricated claims** | 100% | 99% | **99.3%** ✓ |
| **Out-of-context** | 100% | 93% | **96.6%** ✓ |
| **Subtle contradictions** | 100% | 100% | **100.0%** ✓ |
| **Number substitutions** | 79% | 96% | **86.8%** |
| **Partial support** | 76% | 100% | **86.3%** |
| **Faithful statements** | 0% | 0% | **0.0%** ✗ |
| **Overall** | 87% | 97% | **91.3%** |

### The Weakness We Don't Hide

Athena has a **high false positive rate on truly faithful statements** (31% of genuinely faithful sentences are incorrectly flagged). This happens because:

1. **Conservative NLI threshold**: We bias toward precision on hallucinations, sacrificing recall on clean statements
2. **Context fragmentation**: Splitting context into sentences can lose important context
3. **Model limitations**: NLI models sometimes disagree with humans on paraphrases

**Recommendation:** Use athena as a *guardrail*, not a final gate. Flag suspicious statements for human review rather than silently dropping them.

### Comparison with Other Tools

| Tool | Type | F1 (synthetic) | Runtime | Cost |
|------|------|---|---------|------|
| **Athena** | Runtime guardrail | 91.3% | 17ms | Free |
| LettuceDetect | Offline eval | 79.2% | — | — |
| Ragas (LLM-based) | Offline eval | ~75% | 1-2s per sentence | $0.10/1K sentences |
| HHEM-2.1 | Cross-encoder | ~82% | 100ms | Free (local) |
| GPT-4 judge | LLM prompt | ~88% | 2-5s per sentence | $3/1K sentences |

**Key differences:**
- Athena runs *at inference time* in your pipeline
- Ragas and HHEM are *offline* evaluation tools
- GPT-4 gives highest accuracy but costs $$$

**Why Athena wins on speed:** NLI-only, no API calls, runs locally.
**Why Athena loses on edge cases:** Sometimes too aggressive; miss subtle paraphrases.

Full methodology and results: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)

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

## Documentation

- **[Security & Data Privacy](docs/security.md)** — What data leaves your machine? How to stay fully offline?
- **[Threshold Tuning](docs/tuning.md)** — How to pick `trust_threshold` for your domain (legal, support, etc.)
- **[NLI Model Trade-offs](docs/models.md)** — Speed vs accuracy: which model to use (DeBERTa, Lightweight, etc.)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). PRs welcome — especially benchmark results, new integrations, and NLI model improvements.

## License

[MIT](LICENSE)
