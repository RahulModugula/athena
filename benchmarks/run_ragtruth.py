#!/usr/bin/env python3
"""RAGTruth benchmark runner for athena-verify.

Evaluates athena-verify on the RAGTruth dataset (Niu et al., 2024):
18K annotated hallucinations across summarization, QA, and data-to-text.

Usage:
    python benchmarks/run_ragtruth.py --data-dir ./data/ragtruth \\
        --output benchmarks/results/ragtruth.json

Requires the RAGTruth dataset to be downloaded. See:
https://github.com/fluencelab/RAGTruth

Evaluation protocol:
    1. Load dataset and filter to QA subset (most relevant to RAG).
    2. For each example, run athena-verify verify() on (prompt, response, source).
    3. Compare predicted unsupported sentences against gold hallucination labels.
    4. Compute precision, recall, F1, ECE, latency (p50/p95), cost per 1K sentences.
    5. All runs are deterministic given same model weights. Set seeds. Document hardware.

Success gate: athena beats Ragas faithfulness on F1 by at least 5 points.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

from athena_verify import verify


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_ragtruth(data_dir: Path, subset: str | None = None) -> list[dict]:
    """Load RAGTruth dataset.

    Expected format: JSONL files with fields:
    - prompt: the question/prompt
    - response: the LLM response to verify
    - source: the source context
    - label: hallucination label per sentence
    - sentence_labels: list of per-sentence labels (optional)

    Args:
        data_dir: Directory containing RAGTruth JSONL files.
        subset: Optional filter ('qa', 'summarization', 'data2text').

    Returns:
        List of example dicts.
    """
    examples = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        if subset and subset not in jsonl_file.stem.lower():
            continue
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_ece(
    predicted_scores: list[float],
    actual_labels: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        predicted_scores: Predicted trust scores per sentence.
        actual_labels: True labels (True = supported, False = hallucinated).
        n_bins: Number of calibration bins.

    Returns:
        ECE value (lower is better).
    """
    if not predicted_scores:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(predicted_scores)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        indices = [
            j
            for j, s in enumerate(predicted_scores)
            if lo <= s < hi or (i == n_bins - 1 and s == hi)
        ]
        if not indices:
            continue
        bin_acc = sum(1 for j in indices if actual_labels[j]) / len(indices)
        bin_conf = sum(predicted_scores[j] for j in indices) / len(indices)
        ece += len(indices) / total * abs(bin_acc - bin_conf)

    return ece


def evaluate_ragtruth(
    examples: list[dict],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    trust_threshold: float = 0.70,
) -> dict[str, Any]:
    """Run athena-verify on RAGTruth examples.

    Args:
        examples: RAGTruth examples.
        nli_model: NLI model to use.
        trust_threshold: Threshold for verification pass/fail.

    Returns:
        Dict with precision, recall, F1, ECE, latency stats, cost.
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    latencies: list[float] = []
    total_sentences = 0

    all_predicted_scores: list[float] = []
    all_actual_labels: list[bool] = []

    for i, ex in enumerate(examples):
        question = ex.get("prompt", "")
        answer = ex.get("response", "")
        context = [ex.get("source", "")]
        gold_labels = ex.get("sentence_labels", [])

        start = time.time()
        result = verify(
            question=question,
            answer=answer,
            context=context,
            nli_model=nli_model,
            trust_threshold=trust_threshold,
        )
        latency = time.time() - start
        latencies.append(latency)
        total_sentences += len(result.sentences)

        predicted_unsupported = {s.index for s in result.unsupported}

        for sent in result.sentences:
            all_predicted_scores.append(sent.trust_score)

        if not gold_labels:
            for j, _sent in enumerate(result.sentences):
                is_hallucinated = j in predicted_unsupported
                all_actual_labels.append(not is_hallucinated)
            continue

        for j, label in enumerate(gold_labels):
            is_hallucinated = label in ("hallucinated", "unsupported", "contradicted")
            predicted_flag = j in predicted_unsupported

            all_actual_labels.append(not is_hallucinated)

            if is_hallucinated and predicted_flag:
                true_positives += 1
            elif not is_hallucinated and predicted_flag:
                false_positives += 1
            elif is_hallucinated and not predicted_flag:
                false_negatives += 1
            else:
                true_negatives += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    sorted_lat = sorted(latencies)
    ece = compute_ece(all_predicted_scores, all_actual_labels)

    return {
        "num_examples": len(examples),
        "total_sentences": total_sentences,
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        },
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "ece": round(ece, 4),
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0,
        "latency_p50_s": round(_percentile(sorted_lat, 0.50), 3),
        "latency_p95_s": round(_percentile(sorted_lat, 0.95), 3),
        "latency_p99_s": round(_percentile(sorted_lat, 0.99), 3),
        "cost_per_1k_sentences": "local_model (no API cost)",
        "nli_model": nli_model,
        "trust_threshold": trust_threshold,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": 42,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run RAGTruth benchmark")
    parser.add_argument("--data-dir", type=Path, required=True, help="RAGTruth data directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/ragtruth.json"),
        help="Output file",
    )
    parser.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI model",
    )
    parser.add_argument(
        "--subset", default=None, help="Subset filter (qa, summarization, data2text)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)

    print(f"Loading RAGTruth from {args.data_dir}...")
    examples = load_ragtruth(args.data_dir, subset=args.subset)
    print(f"Loaded {len(examples)} examples")

    print("Running evaluation...")
    results = evaluate_ragtruth(examples, nli_model=args.nli_model)

    print("\nResults:")
    print(f"  Precision: {results['precision']}")
    print(f"  Recall:    {results['recall']}")
    print(f"  F1:        {results['f1']}")
    print(f"  ECE:       {results['ece']}")
    print(
        f"  Latency:   {results['latency_mean_s']}s mean, "
        f"{results['latency_p50_s']}s p50, {results['latency_p95_s']}s p95"
    )
    print(f"  Sentences: {results['total_sentences']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
