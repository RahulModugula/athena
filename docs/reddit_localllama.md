Title: I built a local-only RAG hallucination detector — runs entirely on your M-series Mac, no API needed

Hey r/LocalLLaMA — I built [Athena](https://github.com/RahulModugula/athena), an open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence, running 100% locally on Apple Silicon.

**The problem:** Every RAG system hallucinates. Existing tools like Ragas and TruLens are offline evaluators — they tell you your hallucination rate after the fact, not during production. Patronus and Galileo do runtime detection but are closed-source and paid.

**The approach:** Split the LLM answer into sentences. Score each one against the retrieved context using:

1. NLI entailment (DeBERTa-v3-base, ~1.2GB, runs on Metal GPU) — ~20ms per sentence
2. Lexical overlap (token F1) — negligible cost
3. Optional LLM judge for borderline cases — I tested with gemma-4-31b-it via LM Studio

**The benchmarks (100 test cases, Apple M1 Max, 64GB RAM):**

NLI alone catches fabricated claims (F1: 87.9%) and out-of-context info (F1: 88.9%) in ~20ms. But it struggles with number substitutions ("$2M" → "$1M": F1 29.4%) and negation flips ("shall not" → "is permitted": F1 23.5%).

Running every sentence through a local LLM judge (gemma-4-31b-it via LM Studio) fixes both: number substitution F1 jumps to **93.9%** and contradiction F1 to **100%**. The tradeoff is latency: ~7.4s per sentence vs ~20ms for NLI-only.

We tried filtering by NLI confidence (only send borderline scores to the LLM) but it doesn't work — NLI gives extreme scores even when wrong, so almost no sentences got escalated. The right approach is simple: NLI-only for speed, LLM-judge on everything for accuracy.

**The API:**

```python
from athena_verify import verify

result = verify(question="...", answer="...", context=retrieved_chunks)
result.unsupported  # sentences that failed verification
result.trust_score  # 0.0-1.0 overall score
```

Works with LangChain, LlamaIndex, or raw LLM calls. MIT licensed. Zero API cost.

The whole thing — NLI model + LM Studio with a local model — fits comfortably on an M1/M2/M3 Mac. No GPU cloud, no API keys, no data leaving your machine.

Repo: https://github.com/RahulModugula/athena
Install: `pip install athena-verify`

Would love feedback from folks running local RAG pipelines. Especially interested in how it performs with other local models as the LLM judge (I only tested gemma-4-31b-it).
