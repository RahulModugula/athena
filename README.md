# athena-verify

**Open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence — with per-claim source spans that show exactly which chunk supported (or didn't support) each sentence.**

```python
from athena_verify import verify

result = verify(
    question="What is the indemnification cap?",
    answer="The cap is $1M per incident, with a $5M annual aggregate.",
    context=retrieved_chunks,
)

for s in result.sentences:
    status = "✓" if s.supported else "✗"
    print(f"{status} {s.text}")
    for span in s.supporting_spans:
        print(f"  ← chunk[{span.chunk_idx}] {span.start}–{span.end}: {span.text!r}")
```

No document ingestion. No chunking. No agents. No database. Works identically on GPT, Claude, Gemini, Llama, Qwen, or any other model — provider-neutral by design.

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
- **NLI + LLM judge**: escalate borderline cases to a local LLM for additional verification when accuracy is critical

**Hard latency budget** — the first open-source verifier with an explicit `latency_budget_ms` knob:

```python
verify(..., latency_budget_ms=50)   # pure NLI+lexical only — voice AI / agent fast-path
verify(..., latency_budget_ms=500)  # escalate borderline cases if budget allows
verify(..., latency_budget_ms=None) # always escalate (default)
```

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

Evaluated on 100 synthetic cases across 6 hallucination categories (legal, medical, technical, general). Real-world benchmarks against RAGTruth and HaluEval are in progress — download instructions are in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

### Per-Category Performance (NLI-only, synthetic, nli-deberta-v3-base)

| Category | Precision | Recall | **F1** |
|----------|-----------|--------|--------|
| **Fabricated claims** | 100% | 97% | **98.6%** ✓ |
| **Out-of-context** | 100% | 97% | **98.3%** ✓ |
| **Subtle contradictions** | 100% | 97% | **98.3%** ✓ |
| **Number substitutions** | 79% | 96% | **86.8%** |
| **Partial support** | 78% | 95% | **85.7%** |
| **Faithful statements** | 0% | 0% | **0.0%** ✗ |
| **Overall** | 87% | 97% | **91.3%** (synthetic) |

### Where We Lose

Athena has a **high false positive rate on truly faithful statements** (31% of genuinely faithful sentences are incorrectly flagged). This is a known NLI-model limitation — conservative thresholds bias toward catching hallucinations at the cost of flagging clean sentences.

**LettuceDetect beats athena on span-level F1** on real-world benchmarks (LettuceDetect 79.2% F1 on annotated spans vs. athena's unvalidated real-world score). Athena wins on latency bounds, provider-neutrality, offline execution, and the spans-in-library integration story — not raw F1.

**Recommendation:** Use athena as a *guardrail*, not a final gate. Flag suspicious statements for human review rather than silently dropping them.

### Comparison

| Tool | Runs locally | Provider-neutral | Latency budget knob | Per-claim spans | F1 (real) |
|------|---|---|---|---|---|
| **Athena** | Yes | Yes | Yes | Yes | TBD¹ |
| LettuceDetect | Yes | Yes | No | No | **79.2%** |
| HHEM-2.1 | Yes | Yes | No | No | ~82% |
| Ragas | Yes | No (LLM calls) | No | No | ~75% |
| Azure Groundedness | No (cloud only) | No (GPT-4o only) | No | No | ~90% |
| Vertex Grounding | No (cloud only) | No (Gemini only) | No | No | ~88% |
| Anthropic Citations | No (cloud only) | No (Claude only) | No | No | — |

¹ RAGTruth and HaluEval benchmarks pending; see `benchmarks/RESULTS.md` for download instructions.

Full methodology: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)

## How We Compare

| | Athena | Ragas | Azure Groundedness | LettuceDetect |
|---|---|---|---|---|
| **Runtime detection** | Yes | Offline eval | Yes | Offline eval |
| **Sentence-level + spans** | Yes | Answer-level | No | Span-level |
| **Works offline / local** | Yes | Yes | No (cloud) | Yes |
| **Provider-neutral** | Yes | Partial | No (GPT-4o only) | Yes |
| **Latency budget knob** | Yes | No | No | No |
| **Open source** | Yes | Yes | No | Yes |
| **No external API required** | Yes | No (LLM calls) | No | Yes |

Ragas and TruLens are great for **offline evaluation**. Azure/Vertex/Anthropic detectors work only in their own cloud. Athena is the **runtime guardrail for everywhere else** — any model, any stack, fully offline.

## Integrations

### LangChain

```python
from athena_verify.integrations.langchain import VerifyingLLM

chain = RetrievalQA.from_llm(
    VerifyingLLM(llm, retriever=retriever, on_unsupported="re-retrieve", max_retries=2),
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

### LangGraph (agent circuit-breaker)

```python
from athena_verify.integrations.langgraph import VerifyStepNode

graph.add_node("verify", VerifyStepNode(threshold=0.8))
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
| `latency_budget_ms` | `int \| None` | `None` | Hard latency cap; `≤100` skips LLM judge entirely |

Returns `VerificationResult` with `trust_score`, `sentences` (each with `supporting_spans`), `supported`, `unsupported`, and `verification_passed`.

### `verify_step(claim, evidence, threshold)`

Circuit-breaker primitive for agent pipelines:

```python
from athena_verify import verify_step

step = verify_step(
    claim="The contract was signed on 2024-01-15",
    evidence=retrieved_chunks,
    threshold=0.8,
)
# step.passed: bool, step.trust_score: float, step.action: "continue" | "halt"
if step.action == "halt":
    raise ValueError(f"Ungrounded claim blocked (trust={step.trust_score:.2f})")
```

See [`examples/agent_circuit_breaker.py`](examples/agent_circuit_breaker.py) for a full LangGraph example.

## Examples

| Example | Description |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | 5-minute getting started |
| [`examples/langchain_example.py`](examples/langchain_example.py) | LangChain RetrievalQA with self-healing re-retrieve |
| [`examples/llamaindex_example.py`](examples/llamaindex_example.py) | LlamaIndex query engine |
| [`examples/agent_circuit_breaker.py`](examples/agent_circuit_breaker.py) | LangGraph agent with `verify_step` halt |

## Documentation

- **[Security & Data Privacy](docs/security.md)** — What data leaves your machine? How to stay fully offline?
- **[Threshold Tuning](docs/tuning.md)** — How to pick `trust_threshold` for your domain (legal, support, etc.)
- **[NLI Model Trade-offs](docs/models.md)** — Speed vs accuracy: which model to use (DeBERTa, Lightweight, etc.)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). PRs welcome — especially benchmark results, new integrations, and NLI model improvements.

## License

[MIT](LICENSE)
