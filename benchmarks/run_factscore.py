#!/usr/bin/env python3
"""FActScore benchmark runner for athena-verify.

Evaluates athena-verify on the FActScore long-form factuality benchmark.

Usage:
    python benchmarks/run_factscore.py --data-dir ./data/factscore \\
        --output benchmarks/results/factscore.json

Requires the FActScore dataset. See:
https://github.com/shmsw25/FActScore

Evaluation protocol:
    1. Load dataset with Wikipedia-sourced biographies.
    2. For each biography, verify against source knowledge.
    3. Compare sentence-level predictions against gold fact-level labels.
    4. Compute precision, recall, F1, ECE, latency, cost.
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


def load_factscore(data_dir: Path) -> list[dict]:
    """Load FActScore dataset.

    Expected format: JSONL with fields:
    - input: the topic/prompt
    - output: the generated biography/text
    - facts: list of individual facts extracted from the output
    - labels: list of labels (SUPPORTED/NOT_SUPPORTED) for each fact
    - source or knowledge: Wikipedia source paragraphs

    Args:
        data_dir: Directory containing FActScore JSONL files.

    Returns:
        List of example dicts.
    """
    examples = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def _best_matching_sentence_index(fact: str, sentences: list[str]) -> int:
    """Find the sentence most similar to a fact by token overlap."""
    fact_tokens = set(fact.lower().split())
    best_idx = 0
    best_overlap = 0.0
    for idx, sent in enumerate(sentences):
        sent_tokens = set(sent.lower().split())
        overlap = len(fact_tokens & sent_tokens) / max(len(fact_tokens | sent_tokens), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    return best_idx


def evaluate_factscore(
    examples: list[dict],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    trust_threshold: float = 0.70,
) -> dict[str, Any]:
    """Run athena-verify on FActScore examples.

    For each example, we verify the generated text against available
    knowledge/source and compare against gold fact-level labels.

    Args:
        examples: FActScore examples.
        nli_model: NLI model to use.
        trust_threshold: Threshold for verification.

    Returns:
        Dict with fact-level precision, recall, F1, ECE, latency.
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    latencies: list[float] = []
    total_sentences = 0
    total_facts = 0

    all_predicted_scores: list[float] = []
    all_actual_labels: list[bool] = []

    for i, ex in enumerate(examples):
        topic = ex.get("input", "")
        biography = ex.get("output", "")
        source = ex.get("source", ex.get("knowledge", ""))
        context = [source] if source else [biography]

        start = time.time()
        result = verify(
            question=f"Tell me about {topic}",
            answer=biography,
            context=context,
            nli_model=nli_model,
            trust_threshold=trust_threshold,
        )
        latencies.append(time.time() - start)
        total_sentences += len(result.sentences)

        facts = ex.get("facts", [])
        labels = ex.get("labels", [])

        predicted_unsupported_indices = {s.index for s in result.unsupported}
        predicted_status = {s.index: s.trust_score for s in result.sentences}

        if not facts or not labels:
            for sent in result.sentences:
                all_predicted_scores.append(sent.trust_score)
                all_actual_labels.append(True)
            continue

        total_facts += len(facts)

        for fact, label in zip(facts, labels, strict=False):
            is_not_supported = label in ("NOT_SUPPORTED", "false", "incorrect")
            matched_idx = _best_matching_sentence_index(fact, [s.text for s in result.sentences])
            predicted_flagged = matched_idx in predicted_unsupported_indices

            trust = predicted_status.get(matched_idx, 0.5)
            all_predicted_scores.append(trust)
            all_actual_labels.append(not is_not_supported)

            if is_not_supported and predicted_flagged:
                true_positives += 1
            elif not is_not_supported and predicted_flagged:
                false_positives += 1
            elif is_not_supported and not predicted_flagged:
                false_negatives += 1
            else:
                true_negatives += 1

        if (i + 1) % 50 == 0:
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
        "total_facts": total_facts,
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
    parser = argparse.ArgumentParser(description="Run FActScore benchmark")
    parser.add_argument("--data-dir", type=Path, required=True, help="FActScore data directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/factscore.json"),
        help="Output file",
    )
    parser.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)

    print(f"Loading FActScore from {args.data_dir}...")
    examples = load_factscore(args.data_dir)
    print(f"Loaded {len(examples)} examples")

    print("Running evaluation...")
    results = evaluate_factscore(examples, nli_model=args.nli_model)

    print("\nResults:")
    print(f"  Precision: {results['precision']}")
    print(f"  Recall:    {results['recall']}")
    print(f"  F1:        {results['f1']}")
    print(f"  ECE:       {results['ece']}")
    print(f"  Facts:     {results['total_facts']}")
    print(
        f"  Latency:   {results['latency_mean_s']}s mean, "
        f"{results['latency_p50_s']}s p50, {results['latency_p95_s']}s p95"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
