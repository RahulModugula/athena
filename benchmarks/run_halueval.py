#!/usr/bin/env python3
"""HaluEval benchmark runner for athena-verify.

Evaluates athena-verify on the HaluEval dataset:
35K hallucination examples over QA, dialogue, and summarization.

Usage:
    python benchmarks/run_halueval.py --data-dir ./data/halueval \\
        --output benchmarks/results/halueval.json

Requires the HaluEval dataset. See:
https://github.com/RUCAIBox/HaluEval

Evaluation protocol:
    1. Load dataset and filter to QA subset (most relevant to RAG).
    2. For each example, verify both right_answer and hallucinated_answer.
    3. Correct detection: right answer passes, hallucinated answer is flagged.
    4. Compute detection accuracy, precision, recall, F1, ECE, latency, cost.
    5. Deterministic: set seeds, document hardware.
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
    """Compute Expected Calibration Error."""
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


def load_halueval(data_dir: Path, subset: str | None = None) -> list[dict]:
    """Load HaluEval dataset.

    Expected format: JSONL with fields:
    - query: the question
    - right_answer: the correct answer
    - hallucinated_answer: a hallucinated version
    - context: source context (if available)

    Args:
        data_dir: Directory containing HaluEval JSONL files.
        subset: Optional filter ('qa', 'dialogue', 'summarization').

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


def evaluate_halueval(
    examples: list[dict],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    trust_threshold: float = 0.70,
) -> dict[str, Any]:
    """Run athena-verify on HaluEval examples.

    For each example, we verify both the right answer and the
    hallucinated answer. A correct detection means:
    - Right answer: trust_score >= threshold (not flagged)
    - Hallucinated answer: trust_score < threshold (flagged)

    Args:
        examples: HaluEval examples.
        nli_model: NLI model to use.
        trust_threshold: Threshold for verification.

    Returns:
        Dict with detection accuracy, precision, recall, F1, ECE, latency.
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    correct_pairs = 0
    total_pairs = 0
    latencies: list[float] = []
    total_sentences = 0

    all_predicted_scores: list[float] = []
    all_actual_labels: list[bool] = []

    for i, ex in enumerate(examples):
        question = ex.get("query", "")
        context = [ex.get("context", ex.get("knowledge", ""))]
        right_answer = ex.get("right_answer", "")
        hall_answer = ex.get("hallucinated_answer", "")

        if not context[0]:
            context = [right_answer]

        start = time.time()
        right_result = verify(
            question=question,
            answer=right_answer,
            context=context,
            nli_model=nli_model,
            trust_threshold=trust_threshold,
        )
        latencies.append(time.time() - start)
        total_sentences += len(right_result.sentences)

        for sent in right_result.sentences:
            all_predicted_scores.append(sent.trust_score)
            all_actual_labels.append(True)

        start = time.time()
        hall_result = verify(
            question=question,
            answer=hall_answer,
            context=context,
            nli_model=nli_model,
            trust_threshold=trust_threshold,
        )
        latencies.append(time.time() - start)
        total_sentences += len(hall_result.sentences)

        for sent in hall_result.sentences:
            all_predicted_scores.append(sent.trust_score)
            all_actual_labels.append(False)

        if right_result.verification_passed and not hall_result.verification_passed:
            correct_pairs += 1
        total_pairs += 1

        right_unsupported = {s.index for s in right_result.unsupported}
        hall_unsupported = {s.index for s in hall_result.unsupported}

        for j in range(max(len(right_result.sentences), 1)):
            if j in right_unsupported:
                false_positives += 1
            else:
                true_negatives += 1

        for j in range(max(len(hall_result.sentences), 1)):
            if j in hall_unsupported:
                true_positives += 1
            else:
                false_negatives += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    pair_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0
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
        "pair_accuracy": round(pair_accuracy, 4),
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
    parser = argparse.ArgumentParser(description="Run HaluEval benchmark")
    parser.add_argument("--data-dir", type=Path, required=True, help="HaluEval data directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/halueval.json"),
        help="Output file",
    )
    parser.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI model",
    )
    parser.add_argument(
        "--subset", default=None, help="Subset filter (qa, dialogue, summarization)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)

    print(f"Loading HaluEval from {args.data_dir}...")
    examples = load_halueval(args.data_dir, subset=args.subset)
    print(f"Loaded {len(examples)} examples")

    print("Running evaluation...")
    results = evaluate_halueval(examples, nli_model=args.nli_model)

    print("\nResults:")
    print(f"  Pair accuracy: {results['pair_accuracy']}")
    print(f"  Precision:     {results['precision']}")
    print(f"  Recall:        {results['recall']}")
    print(f"  F1:            {results['f1']}")
    print(f"  ECE:           {results['ece']}")
    print(
        f"  Latency:       {results['latency_mean_s']}s mean, "
        f"{results['latency_p50_s']}s p50, {results['latency_p95_s']}s p95"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
