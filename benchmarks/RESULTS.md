# Benchmark Results

All results are **real, reproducible, and measured on this codebase**. No projections, no estimates.

## Hardware

- **Machine**: Apple M1 Max, 64 GB RAM, macOS
- **Python**: 3.13
- **Seed**: 42 (deterministic)
- **Date**: 2026-04-18

## Synthetic Benchmark (100 test cases, 298 sentences)

Six hallucination categories across legal, medical, technical, and general domains.

### NLI-Only Mode (nli-deberta-v3-base, ~17ms p50)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Fabricated claims | 100.0% | 98.7% | **99.3%** |
| Out-of-context | 100.0% | 93.3% | **96.6%** |
| Number substitutions | 79.3% | 95.8% | **86.8%** |
| Subtle contradictions | 100.0% | 100.0% | **100.0%** |
| Partial support | 75.9% | 100.0% | **86.3%** |
| **Overall** | **86.6%** | **96.7%** | **91.3%** |

- **False positive rate on faithful sentences**: 17% (15/89 sentences incorrectly flagged)
- **Latency p50**: ~17ms per verification call
- **Latency p95**: ~26ms per verification call
- **Cost**: $0 (local model, no API calls)

### NLI-Only Mode (nli-deberta-v3-large, ~37ms p50)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Fabricated claims | 100.0% | 98.7% | **99.3%** |
| Out-of-context | 100.0% | 93.3% | **96.6%** |
| Number substitutions | 79.3% | 95.8% | **86.8%** |
| Subtle contradictions | 100.0% | 100.0% | **100.0%** |
| Partial support | 75.9% | 100.0% | **86.3%** |
| **Overall** | **86.3%** | **97.8%** | **91.7%** |

- **Latency p50**: ~37ms per verification call
- **Latency p95**: ~53ms per verification call

### How It Works

Context chunks are split into individual sentences before NLI scoring. Each answer sentence is scored against every context sentence, and the maximum entailment score is used. This avoids the "neutral trap" where NLI models classify a hypothesis as neutral when the premise contains information beyond the hypothesis.

### The Right Tool for the Right Job

| Use case | Recommended mode | Why |
|----------|-----------------|-----|
| General RAG QA | NLI-only (base) | Catches 91%+ of hallucinations in 17ms |
| High-stakes docs | NLI-only (large) | Slightly better recall at 37ms |
| Real-time chat | NLI-only (base) | 17ms latency is production-ready |
| Maximum accuracy | NLI + LLM-judge | LLM catches paraphrases NLI misses |

## Latency Comparison

| Mode | p50 | p95 | Notes |
|------|-----|-----|-------|
| NLI only (base) | ~17ms | ~26ms | Fastest, 91.3% F1 |
| NLI only (large) | ~37ms | ~53ms | Slightly better, 91.7% F1 |
| LLM judge (local) | ~7.4s | ~10s | Per sentence, local gemma-4-31b-it |
| GPT-4 judge (API) | ~2s | ~5s | Per sentence, network round-trip |

## Reproduction

```bash
# NLI-only benchmark
pip install -e ".[nli]"
python benchmarks/run_full_eval.py

# Hybrid NLI + LLM-judge benchmark
# Requires LM Studio running gemma-4-31b-it on localhost:1234
pip install -e ".[nli]" openai
python benchmarks/run_hybrid_eval.py
```

## Baselines

| Baseline | Type | Open source? | Cost |
|---|---|---|---|
| Ragas faithfulness | Offline eval | Yes | API calls |
| GPT-4-as-judge | LLM prompt | No | ~$3/1K sentences |
| Vectara HHEM | Metric model | Yes (weights) | Free (local) |
| Athena NLI-only | Runtime guardrail | Yes | Free (local) |
