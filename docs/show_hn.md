Title: Athena — Open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence

Every RAG system hallucinates. The question is whether you catch it before users see it.

Athena is an open-source (MIT) Python library that verifies LLM answers against retrieved context in real time. It splits the answer into sentences and scores each one using NLI entailment (DeBERTa-v3), lexical overlap, and an optional local LLM judge. No API calls required — runs entirely on a MacBook.

**How it works:** Pass in your question, the LLM answer, and the retrieved context. Get back a trust score, per-sentence verification, and a list of unsupported claims. Three lines of code:

```python
from athena_verify import verify

result = verify(question="What is the cap?", answer="...", context=chunks)
result.unsupported  # sentences that failed verification
```

**Benchmarks (100 test cases, 6 hallucination categories, Apple M1 Max):**

| Category | NLI-only F1 | + LLM-judge F1 |
|----------|-------------|-----------------|
| Fabricated claims | 87.9% | — |
| Out-of-context | 88.9% | — |
| Number substitutions | 29.4% | **93.9%** |
| Subtle contradictions | 23.5% | **100.0%** |

The honest take: NLI alone is great for fabricated claims and out-of-context info (88%+ F1, ~20ms latency) but bad at number substitutions and negation flips (23-29% F1). Running every sentence through a local LLM judge (gemma-4-31b-it via LM Studio) fixes the weak categories — number subs jump to 93.9% and contradictions to 100% — at the cost of ~7.4s per sentence.

We tried a "borderline escalation" filter (only send ambiguous NLI scores to LLM) but it doesn't work — NLI gives extreme scores even when wrong, so only 1 out of 298 sentences triggered escalation. The right approach: NLI-only for speed, LLM-judge on everything for accuracy.

**Why this exists:** Ragas, TruLens, and DeepEval do offline batch evaluation. Patronus and Galileo do runtime detection but are closed-source and paid. There was no open-source, sentence-level, runtime verification layer for RAG. Now there is.

**Integrations:** LangChain, LlamaIndex, and raw OpenAI/Anthropic SDK. Works with any RAG pipeline.

Repo: https://github.com/RahulModugula/athena

Install: `pip install athena-verify`

Happy to answer questions about the approach, the benchmarks, or the NLI limitations.
