# Benchmark Results

All results are **real, reproducible, and measured on this codebase**. No projections, no estimates.

## Evaluation Protocol

1. For each benchmark, load the dataset and run `verify()` on each (question, answer, context) triple.
2. Compare Athena's `unsupported` classification against gold-standard hallucination labels.
3. Compute:
   - **Precision**: Of sentences flagged as unsupported, how many are actually hallucinated.
   - **Recall**: Of all hallucinated sentences, how many did we catch.
   - **F1**: Harmonic mean of precision and recall.
   - **ECE**: Expected Calibration Error — how well trust scores match actual accuracy (lower is better).
   - **Latency**: p50 and p95 per verification call.
   - **Cost**: Per 1K sentences (local model = $0; API-based methods vary).
4. Run the same evaluation for each baseline using their standard APIs.
5. All scripts are deterministic given the same model weights. Seeds are set. Hardware is documented.

## Success Gate

Athena must beat Ragas faithfulness on F1 by at least 5 points on 2 of 3 benchmarks. If it doesn't, the verifier must be improved before launch.

## Results

**Status: Awaiting dataset downloads and benchmark runs.**

To reproduce:

```bash
# Install with NLI support
pip install -e ".[nli]"

# RAGTruth (download from https://github.com/fluencelab/RAGTruth)
python benchmarks/run_ragtruth.py --data-dir ./data/ragtruth --output benchmarks/results/ragtruth.json

# HaluEval (download from https://github.com/RUCAIBox/HaluEval)
python benchmarks/run_halueval.py --data-dir ./data/halueval --output benchmarks/results/halueval.json

# FActScore (download from https://github.com/shmsw25/FActScore)
python benchmarks/run_factscore.py --data-dir ./data/factscore --output benchmarks/results/factscore.json

# Baselines (requires API keys)
python benchmarks/run_baselines.py --method ragas --data-dir ./data/ragtruth --output benchmarks/results/baseline_ragas.json
python benchmarks/run_baselines.py --method gpt4_judge --data-dir ./data/ragtruth --output benchmarks/results/baseline_gpt4.json
python benchmarks/run_baselines.py --method vectara_hhem --data-dir ./data/ragtruth --output benchmarks/results/baseline_hhem.json
```

## Baselines

| Baseline | Type | Open source? | Notes |
|---|---|---|---|
| Ragas faithfulness | Offline eval | Yes | Industry standard for RAG eval |
| GPT-4-as-judge | LLM prompt | No (API) | Expensive but strong |
| Vectara HHEM | Metric model | Yes (weights) | Sentence-level hallucination detection |
| Lynx-8B | LLM weights | Yes | Patr et al., 2024 |

## Metrics

| Metric | Description | Target |
|---|---|---|
| Precision | Flagged as unsupported, actually hallucinated | > 0.80 |
| Recall | Hallucinated sentences correctly caught | > 0.70 |
| F1 | Harmonic mean of precision and recall | Beat Ragas by 5+ pts |
| ECE | Expected Calibration Error (lower = better) | < 0.10 |
| Latency p50 | Median time per verification call | < 500ms |
| Latency p95 | 95th percentile time | < 2s |
| Cost per 1K sentences | API or compute cost | $0 (local model) |

## Hardware Requirements

- **NLI model**: `cross-encoder/nli-deberta-v3-base` (~1.2 GB, one-time download)
- **RAM**: 4 GB minimum, 8 GB recommended
- **GPU**: Optional (CPU works, GPU is faster)
- **Disk**: ~2 GB for model weights

## Output Format

Each benchmark runner outputs a JSON file with the following structure:

```json
{
  "num_examples": 1000,
  "total_sentences": 3500,
  "confusion_matrix": {
    "true_positives": 450,
    "false_positives": 80,
    "true_negatives": 2800,
    "false_negatives": 170
  },
  "precision": 0.8491,
  "recall": 0.7258,
  "f1": 0.7826,
  "ece": 0.0632,
  "latency_mean_s": 0.312,
  "latency_p50_s": 0.285,
  "latency_p95_s": 0.892,
  "latency_p99_s": 1.453,
  "cost_per_1k_sentences": "local_model (no API cost)",
  "nli_model": "cross-encoder/nli-deberta-v3-base",
  "trust_threshold": 0.70,
  "environment": {
    "python": "3.12.0",
    "platform": "macOS-14.0-arm64",
    "seed": 42
  }
}
```
