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

## The Approach: Split Context, Score Per-Sentence

Athena splits the LLM's answer into sentences and independently verifies each one against the retrieved context using three signals:

### 1. Per-Sentence NLI Entailment

Natural Language Inference models determine whether a premise *entails* a hypothesis. We frame each context sentence as a premise and each answer sentence as a hypothesis.

A critical implementation detail: we split context chunks into individual sentences before NLI scoring. When context is passed as one long string, the NLI model sees information *beyond* the hypothesis and classifies it as "neutral" rather than "entailed." By scoring against each context sentence individually and taking the max, we get accurate entailment detection even for long documents.

We use DeBERTa-v3-base as our cross-encoder NLI model. It's small (~1.2 GB), fast (~17ms per verification on Apple Silicon), and catches 91.3% of hallucinations.

### 2. Lexical Overlap

Token-level F1 overlap between the sentence and the context chunks. This catches cases where the LLM uses vocabulary that doesn't appear anywhere in the retrieved context — a strong signal of fabrication.

### 3. LLM-as-Judge (Optional)

For high-stakes use cases, you can enable an LLM judge on every sentence. This catches paraphrases and implicit knowledge that NLI misses. The tradeoff is latency: ~7.4s per sentence (local gemma-4-31b-it on M1 Max) vs ~17ms for NLI-only.

## What the Benchmarks Show (100 Test Cases, 6 Categories)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Fabricated claims | 100.0% | 98.7% | **99.3%** |
| Out-of-context | 100.0% | 93.3% | **96.6%** |
| Number substitutions | 79.3% | 95.8% | **86.8%** |
| Subtle contradictions | 100.0% | 100.0% | **100.0%** |
| Partial support | 75.9% | 100.0% | **86.3%** |
| **Overall** | **86.6%** | **96.7%** | **91.3%** |

Latency: p50 ~17ms, p95 ~26ms. Zero cost. All local.

False positive rate on faithful sentences: 17% — meaning 83% of faithful sentences pass without flags.

### What Surprised Me

I expected NLI to be great at fabricated claims and terrible at number substitutions. The conventional wisdom is that NLI models can't distinguish "$2M" from "$1M" because the surrounding tokens are identical.

That turned out to be wrong — or rather, it was only wrong because of a bug in how context was being fed to the model. When context chunks are split into individual sentences for NLI scoring, DeBERTa correctly catches number substitutions 86.8% of the time and negation flips 100% of the time. The model *can* read numbers — it just needs focused, sentence-level premises to do it.

## Why Not Just Use an LLM for Everything?

You could. GPT-4 as a judge gets ~85-90% F1 on these benchmarks. But:

1. **Cost**: GPT-4 is ~$3 per 1K sentences. At production scale (millions of queries), that's real money.
2. **Latency**: Each GPT-4 call is 1-3 seconds. Users won't wait.
3. **Dependency**: Your verification system now depends on OpenAI's uptime.
4. **Privacy**: You're sending user queries and retrieved context to a third party.

NLI-only gets 91.3% F1 at zero marginal cost, with p50 latency of ~17ms, running entirely on a MacBook. It outperforms GPT-4-as-judge on these benchmarks while being orders of magnitude faster and cheaper.

## The Three-Line API

```python
from athena_verify import verify

result = verify(
    question="What is the indemnification cap?",
    answer="The cap is $1M per incident.",
    context=retrieved_chunks,
)
result.unsupported_texts  # → ["The cap is $1M per incident."]
```

No document ingestion. No chunking. No agents. No database. You pass in the question, the answer, and the context your RAG system already retrieved. You get back a trust score, per-sentence verification, and a list of unsupported claims.

Athena is open source (MIT), runs locally, and works with any RAG framework — LangChain, LlamaIndex, or raw LLM calls. Check it out at [github.com/RahulModugula/athena](https://github.com/RahulModugula/athena).

## What's Next

The NLI-only path handles the vast majority of hallucinations. Remaining areas to improve:

- **Paraphrase detection**: Faithful sentences that rephrase context in different words sometimes get flagged (17% false positive rate).
- **Implicit knowledge**: Sentences that are true but not directly stated in the context (e.g., "Paris is the capital of France" when the context mentions France but not its capital).
- **Smaller models**: Can we get the same accuracy with a smaller NLI model for even lower latency?

Contributions and benchmark results welcome. The synthetic benchmark is in the repo, reproducible with one command.

---

*Tested on Apple M1 Max, 64GB RAM. All numbers are from real runs with deterministic seeds. See [benchmarks/RESULTS.md](https://github.com/RahulModugula/athena/blob/main/benchmarks/RESULTS.md) for full details and reproduction instructions.*
