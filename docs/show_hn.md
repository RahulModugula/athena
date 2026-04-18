Title: Athena — Open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence (91% F1, 17ms, free)

Every RAG system hallucinates. The question is whether you catch it before users see it.

Athena is an open-source (MIT) Python library that verifies LLM answers against retrieved context in real time. It splits the answer into sentences, splits the context into sentences, and scores each answer sentence against each context sentence using NLI entailment (DeBERTa-v3). No API calls required — runs entirely on a MacBook.

**How it works:** Pass in your question, the LLM answer, and the retrieved context. Get back a trust score, per-sentence verification, and a list of unsupported claims. Three lines of code:

```python
from athena_verify import verify

result = verify(question="What is the cap?", answer="...", context=chunks)
result.unsupported_texts  # sentences that failed verification
```

**Benchmarks (100 test cases, 6 hallucination categories, Apple M1 Max):**

| Category | F1 |
|----------|-----|
| Fabricated claims | 99.3% |
| Out-of-context | 96.6% |
| Number substitutions | 86.8% |
| Subtle contradictions | 100.0% |
| Partial support | 86.3% |
| **Overall** | **91.3%** |

At ~17ms p50 latency, $0 cost, running locally on a MacBook.

**Key insight:** Splitting context into individual sentences before NLI scoring makes a huge difference. When context is fed as one long string, the NLI model says "neutral" for everything because the premise contains information beyond the hypothesis. Per-sentence scoring lets the model focus on the relevant context and correctly catch number swaps ($2M → $1M) and negation flips ("shall not" → "is permitted").

**Why this exists:** Ragas, TruLens, and DeepEval do offline batch evaluation. Patronus and Galileo do runtime detection but are closed-source and paid. There was no open-source, sentence-level, runtime verification layer for RAG. Now there is.

**Integrations:** LangChain, LlamaIndex, and raw OpenAI/Anthropic SDK. Works with any RAG pipeline.

Repo: https://github.com/RahulModugula/athena

Install: `pip install athena-verify`

Happy to answer questions about the approach, the benchmarks, or the per-sentence NLI trick.
