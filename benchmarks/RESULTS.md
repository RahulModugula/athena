# Benchmark Results

Benchmarks in progress. Results will be published after running on:

- **RAGTruth** (Niu et al., 2024) — 18K annotated hallucinations
- **HaluEval** — 35K hallucination examples
- **FActScore** — long-form factuality benchmark

## Reproduction

```bash
# RAGTruth
python benchmarks/run_ragtruth.py --data-dir ./data/ragtruth --output benchmarks/results/ragtruth.json

# HaluEval
python benchmarks/run_halueval.py --data-dir ./data/halueval --output benchmarks/results/halueval.json

# FActScore
python benchmarks/run_factscore.py --data-dir ./data/factscore --output benchmarks/results/factscore.json
```

## Baselines

Athena-verify will be compared against:

| Baseline | Type | Notes |
|---|---|---|
| Ragas faithfulness | Offline eval | Industry standard |
| Lynx-8B | Weights only | Patra et al., 2024 |
| Vectara HHEM | Metric only | Commercial |
| GPT-4-as-judge | LLM prompt | Expensive but strong |

## Metrics

- **Precision**: Of sentences flagged as unsupported, how many are actually hallucinated
- **Recall**: Of all hallucinated sentences, how many did we catch
- **F1**: Harmonic mean of precision and recall
- **ECE**: Expected Calibration Error (how well trust scores match actual accuracy)
- **Latency**: p50 and p95 per sentence
- **Cost**: Per 1K sentences (API costs for LLM-based methods)
