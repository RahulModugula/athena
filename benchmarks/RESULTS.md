# Benchmark Results

All results are **real, reproducible, and measured on this codebase**. No projections, no estimates.

## Hardware

- **Machine**: Apple M1 Max, 64 GB RAM, macOS
- **Python**: 3.13
- **Seed**: 42 (deterministic)
- **Date**: 2026-06-27

## Real Dataset Acquisition

### RAGTruth (18K examples)

Download from: https://github.com/fluencelab/RAGTruth

```bash
# Clone the repository
git clone https://github.com/fluencelab/RAGTruth.git
cp -r RAGTruth/data/* ./data/ragtruth/

# Run benchmark
python benchmarks/run_ragtruth.py \
    --data-dir ./data/ragtruth \
    --subset qa \
    --output benchmarks/results/ragtruth.json
```

### HaluEval (35K examples)

Download from: https://github.com/RUCAIBox/HaluEval

```bash
# Clone and prepare data
git clone https://github.com/RUCAIBox/HaluEval.git
cp -r HaluEval/data/* ./data/halueval/

# Run benchmark
python benchmarks/run_halueval.py \
    --data-dir ./data/halueval \
    --subset qa \
    --output benchmarks/results/halueval.json
```

### FaithBench (multi-domain faithfulness)

Download from: https://huggingface.co/datasets/CogComp/FaithBench

```bash
# Using HF datasets library
python -c "from datasets import load_dataset; \
ds = load_dataset('CogComp/FaithBench'); \
import json; \
[print(json.dumps(x)) for x in ds['test'][:1000]]" > ./data/faithbench/faithbench.jsonl

# Run benchmark
python benchmarks/run_faithbench.py \
    --data-dir ./data/faithbench \
    --output benchmarks/results/faithbench.json
```

If datasets are unavailable, benchmarks can run on synthetic data:

```bash
python benchmarks/run_faithbench.py --synthetic \
    --num-synthetic 100 \
    --output benchmarks/results/faithbench_synthetic.json
```

## Comprehensive Synthetic Benchmark (100 test cases, 298 sentences)

Six hallucination categories across legal, medical, technical, and general domains.

### NLI-Only Mode (nli-deberta-v3-base, ~24ms p50)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Fabricated claims | 100.0% | 96.0% | **97.9%** |
| Out-of-context | 100.0% | 96.7% | **98.3%** |
| Number substitutions | 82.1% | 95.8% | **88.5%** |
| Subtle contradictions | 100.0% | 100.0% | **100.0%** |
| Partial support | 87.5% | 95.5% | **91.3%** |
| **Overall** | **90.6%** | **96.7%** | **93.6%** |

- **False positive rate on faithful sentences**: 11.5% (10/87 sentences incorrectly flagged)
- **Latency p50**: ~23.5ms per verification call
- **Latency p95**: ~36.2ms per verification call
- **Cost**: $0 (local model, no API calls)

### NLI-Only Mode (nli-deberta-v3-large, ~53ms p50)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Fabricated claims | 100.0% | 98.7% | **99.3%** |
| Out-of-context | 100.0% | 93.3% | **96.5%** |
| Number substitutions | 82.1% | 95.8% | **88.5%** |
| Subtle contradictions | 100.0% | 100.0% | **100.0%** |
| Partial support | 80.8% | 95.5% | **87.5%** |
| **Overall** | **90.7%** | **97.2%** | **93.8%** |

- **False positive rate on faithful sentences**: 9.2% (8/87 sentences incorrectly flagged)
- **Latency p50**: ~52.5ms per verification call
- **Latency p95**: ~83.2ms per verification call

### How It Works

Context chunks are split into individual sentences before NLI scoring. Each answer sentence is scored against every context sentence **and the full chunk**, and the maximum entailment score is used. This avoids the "neutral trap" where NLI models classify a hypothesis as neutral when the premise contains information beyond the hypothesis, while still catching facts spread across several context sentences. Answer sentences that open with an anaphor ("This cap…", "It also…") are joined with the previous sentence before scoring so the referent is preserved — the single largest source of false positives on faithful text.

### The Right Tool for the Right Job

| Use case | Recommended mode | Why |
|----------|-----------------|-----|
| General RAG QA | NLI-only (base) | 93.6% F1 in ~24ms |
| High-stakes docs | NLI-only (large) | Lower false-positive rate at ~53ms |
| Real-time chat | NLI-only (base) | ~24ms latency is production-ready |
| Maximum accuracy | NLI + LLM-judge | LLM catches paraphrases NLI misses |

## Latency Comparison

| Mode | p50 | p95 | Notes |
|------|-----|-----|-------|
| NLI only (base) | ~24ms | ~36ms | Fastest, 93.6% F1, 11.5% FP on faithful |
| NLI only (large) | ~53ms | ~83ms | Lower FP (9.2%), 93.8% F1 |
| LLM judge (local) | ~7.4s | ~10s | Per sentence, local gemma-4-31b-it |
| GPT-4 judge (API) | ~2s | ~5s | Per sentence, network round-trip |

## FaithBench Synthetic Benchmark (100 synthetic examples)

Simple faithful vs. hallucinated classification. Synthetic data is by design clear-cut:
all hallucinated sentences are obviously false, all faithful sentences are obviously true.

| Metric | Value |
|--------|-------|
| **Precision** | 100.0% |
| **Recall** | 100.0% |
| **F1** | **100.0%** |
| **ECE** | 0.144 |
| **Latency p50** | 22ms |
| **Latency p95** | 86ms |
| **Cost** | Free (local model) |

**Note:** Perfect scores on synthetic data don't guarantee real-world performance. Real hallucinations are subtler (paraphrases, number approximations, missing context).

## Reproduction

```bash
# NLI-only benchmark (comprehensive)
pip install -e ".[nli]"
python benchmarks/run_full_eval.py

# FaithBench synthetic benchmark
python benchmarks/run_faithbench.py --synthetic

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
