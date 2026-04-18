# Why Your RAG is Hallucinating and How Sentence-Level NLI Catches It

Every RAG system hallucinates. Not sometimes — always. The question isn't whether your retrieval-augmented generation pipeline produces fabricated facts. It's whether you know when it does.

I spent the last few months building [Athena](https://github.com/RahulModugula/athena), an open-source runtime guardrail that catches RAG hallucinations at the sentence level, in production, with zero API cost. Here's what I learned about why RAG hallucinates and how to catch it.

## The Inevitability of RAG Hallucinations

RAG systems don't hallucinate because they're broken. They hallucinate because of fundamental tensions in how they work:

**Retrieval is lossy.** You chunk documents, embed them, and retrieve the top-k by vector similarity. But similarity isn't sufficiency. The retrieved chunks may not contain the specific fact the LLM needs, and the LLM will fill the gap with its training data rather than admitting ignorance.

**Generation is confabulatory.** LLMs are completion engines. Given a prompt that says "the indemnification cap is" they will complete it, even if the context says nothing about indemnification. The more fluent the model, the more convincing the hallucination.

**Context windows create ambiguity.** Even when the right information is retrieved, the LLM must synthesize across multiple chunks. Synthesis introduces errors — conflating numbers from different sources, attributing properties to the wrong entity, or contradicting a clause while paraphrasing it.

In production, we've seen hallucination rates of 5-15% even on well-tuned RAG pipelines. The existing tools for detecting this — Ragas, TruLens, DeepEval — are offline batch evaluators. They tell you your system's hallucination rate *after the fact*. They don't catch hallucinations *before they reach the user*.

That's the gap Athena fills.

## The Approach: Three Signals, One Score

Athena splits the LLM's answer into sentences and independently verifies each one against the retrieved context using three signals:

### 1. NLI Entailment (weight: 55%)

Natural Language Inference models are trained to determine whether a premise *entails* a hypothesis. We frame context as the premise and each answer sentence as the hypothesis. If the NLI model says "entailed" with high confidence, the sentence is supported. If it says "contradicted," the sentence is a hallucination.

We use DeBERTa-v3-base as our cross-encoder NLI model. It's small (~1.2 GB), fast (~20ms per sentence on Apple Silicon), and surprisingly effective at detecting fabricated claims and out-of-context information.

### 2. Lexical Overlap (weight: 25%)

Token-level F1 overlap between the sentence and the context chunks. This catches cases where the LLM uses vocabulary that doesn't appear anywhere in the retrieved context — a strong signal of fabrication.

### 3. LLM-as-Judge (weight: 20%, optional)

For high-stakes use cases, you can enable the LLM judge on every sentence. We tested with gemma-4-31b-it via LM Studio. The LLM judge is asked a simple question: "Is this claim fully supported by the context? Answer SUPPORTED or UNSUPPORTED."

The LLM judge dramatically catches what NLI misses — number substitutions jump from 29.4% F1 to 93.9%, and subtle contradictions from 23.5% to 100%. The tradeoff is latency: ~7.4s per sentence on a local M1 Max.

## What the Benchmarks Show (The Honest Version)

We built a synthetic benchmark with 100 test cases across six hallucination categories: fabricated claims, out-of-context information, partial support, number substitutions, subtle contradictions, and faithful answers. Each case has gold-standard sentence-level labels. Here are the real numbers:

### NLI-only (nli-deberta-v3-large)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Fabricated claims | 100.0% | 78.4% | **87.9%** |
| Out-of-context | 100.0% | 80.0% | **88.9%** |
| Partial support | 38.9% | 63.6% | 48.3% |
| Number substitutions | 50.0% | 20.8% | 29.4% |
| Subtle contradictions | 100.0% | 13.3% | 23.5% |

Latency: p50 ~20ms, p95 ~45ms. Zero cost.

### What NLI Gets Wrong

Two categories tank the overall score: **number substitutions** and **subtle contradictions**.

**Number substitutions** are when the LLM changes "$2M" to "$1M" or "90 days" to "30 days". NLI models are trained on entailment datasets that test logical reasoning, not numerical precision. When two sentences differ only in a number, the cross-encoder sees nearly identical tokens and classifies them as entailed. The F1 score of 29.4% means the model struggles to distinguish "$2 million" from "$1 million" because the surrounding structure is identical.

**Subtle contradictions** are negation flips: "shall not disclose" becomes "is permitted to disclose". The NLI model sees two sentences about disclosure with very similar structure and misses the negation. This is a well-documented weakness of NLI models on the NegNLI dataset.

### The LLM-Judge Fix

Sending every sentence to a local LLM judge (instead of trying to filter borderline NLI scores) dramatically improves both weak categories:

| Category | NLI-only F1 | + LLM-judge F1 |
|----------|-------------|-----------------|
| Number substitutions | 29.4% | **93.9%** |
| Subtle contradictions | 23.5% | **100.0%** |

We initially tried a "borderline escalation" approach — only sending sentences with NLI scores between 0.3 and 0.7 to the LLM. This doesn't work because NLI gives extreme scores (near 0 or near 1) even when wrong. Out of 298 sentences, only 1 triggered escalation. The right approach is simpler: use NLI-only when speed matters, use LLM-judge on everything when accuracy matters.

The latency tradeoff is clear: NLI-only gives you ~20ms per sentence. LLM-judge adds ~7.4s per sentence (local gemma-4-31b-it on M1 Max). Pick the mode that fits your use case.

## Why Not Just Use an LLM for Everything?

You could. GPT-4 as a judge gets ~85-90% F1 on these benchmarks. But:

1. **Cost**: GPT-4 is ~$3 per 1K sentences. At production scale (millions of queries), that's real money.
2. **Latency**: Each GPT-4 call is 1-3 seconds. Users won't wait.
3. **Dependency**: Your verification system now depends on OpenAI's uptime.
4. **Privacy**: You're sending user queries and retrieved context to a third party.

The NLI-only mode gives you 88%+ F1 on fabricated claims and out-of-context info at ~20ms per sentence, zero cost, running entirely on a MacBook. When you need accuracy on numbers and negations, enable the LLM judge — it's still local, still free, but trades latency for precision.

## The Three-Line API

```python
from athena_verify import verify

result = verify(
    question="What is the indemnification cap?",
    answer="The cap is $1M per incident.",
    context=retrieved_chunks,
)
result.unsupported  # → ["The cap is $1M per incident."]
```

No document ingestion. No chunking. No agents. No database. You pass in the question, the answer, and the context your RAG system already retrieved. You get back a trust score, per-sentence verification, and a list of unsupported claims.

Athena is open source (MIT), runs locally, and works with any RAG framework — LangChain, LlamaIndex, or raw LLM calls. Check it out at [github.com/RahulModugula/athena](https://github.com/RahulModugula/athena).

## What's Next

The NLI-only path handles fabricated claims and out-of-context information well. The LLM-judge path adds number and negation detection. Remaining areas to improve:

- **Multi-hop reasoning**: When a sentence requires combining facts from multiple context chunks, both NLI and LLM judges struggle.
- **Implicit knowledge**: Sentences that are true but not directly stated in the context (e.g., "Paris is the capital of France" when the context mentions France but not its capital).
- **Smaller LLM judges**: Can we get the same accuracy with a 3B model instead of 31B?

Contributions and benchmark results welcome. The synthetic benchmark is in the repo, reproducible with one command.

---

*Tested on Apple M1 Max, 64GB RAM. All numbers are from real runs with deterministic seeds. See [benchmarks/RESULTS.md](https://github.com/RahulModugula/athena/blob/main/benchmarks/RESULTS.md) for full details and reproduction instructions.*
