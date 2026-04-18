Title: I built a local-only RAG hallucination detector — 91% F1 at 17ms, runs entirely on your M-series Mac

Hey r/LocalLLaMA — I built [Athena](https://github.com/RahulModugula/athena), an open-source runtime guardrail that catches RAG hallucinations sentence-by-sentence, running 100% locally on Apple Silicon.

**The problem:** Every RAG system hallucinates. Existing tools like Ragas and TruLens are offline evaluators — they tell you your hallucination rate after the fact, not during production. Patronus and Galileo do runtime detection but are closed-source and paid.

**The approach:** Split the LLM answer into sentences. Split the context into sentences. Score each answer sentence against each context sentence using NLI entailment (DeBERTa-v3-base, ~1.2GB, runs on Metal GPU).

**The key trick:** Feed context to NLI one sentence at a time, not as one big chunk. When context is one long string, the NLI model classifies everything as "neutral" because the premise has information beyond the hypothesis. Per-sentence scoring lets the model focus and correctly catches number swaps and negation flips.

**The benchmarks (100 test cases, Apple M1 Max, 64GB RAM):**

| Category | F1 |
|----------|-----|
| Fabricated claims | 99.3% |
| Out-of-context | 96.6% |
| Number substitutions | 86.8% |
| Subtle contradictions | 100.0% |
| **Overall** | **91.3%** |

~17ms p50 latency, $0 cost, no API calls.

**The API:**

```python
from athena_verify import verify

result = verify(question="...", answer="...", context=retrieved_chunks)
result.unsupported_texts  # sentences that failed verification
result.trust_score         # 0.0-1.0 overall score
```

Works with LangChain, LlamaIndex, or raw LLM calls. MIT licensed. Zero API cost.

The whole thing — NLI model + optional LM Studio with a local model — fits comfortably on an M1/M2/M3 Mac. No GPU cloud, no API keys, no data leaving your machine.

Repo: https://github.com/RahulModugula/athena
Install: `pip install athena-verify`

Would love feedback from folks running local RAG pipelines.
