#!/usr/bin/env python3
"""RAGTruth benchmark runner for athena-verify.

Evaluates athena-verify on the RAGTruth dataset (Niu et al., 2024):
18K annotated hallucinations across summarization, QA, and data-to-text.

Usage:
    python benchmarks/run_ragtruth.py --data-dir ./data/ragtruth --output benchmarks/results/ragtruth.json

Requires the RAGTruth dataset to be downloaded. See:
https://github.com/fluencelab/RAGTruth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from athena_verify import verify


def load_ragtruth(data_dir: Path) -> list[dict]:
    """Load RAGTruth dataset.

    Expected format: JSONL files with fields:
    - prompt: the question/prompt
    - response: the LLM response to verify
    - source: the source context
    - label: hallucination label per sentence

    Args:
        data_dir: Directory containing RAGTruth JSONL files.

    Returns:
        List of example dicts.
    """
    examples = []
    for jsonl_file in data_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def evaluate_ragtruth(
    examples: list[dict],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
) -> dict:
    """Run athena-verify on RAGTruth examples.

    Args:
        examples: RAGTruth examples.
        nli_model: NLI model to use.

    Returns:
        Dict with precision, recall, F1, latency stats.
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    latencies = []

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
        )
        latency = time.time() - start
        latencies.append(latency)

        # Compare predicted unsupported vs gold hallucination labels
        predicted_unsupported = {s.index for s in result.unsupported}

        if not gold_labels:
            # No gold labels — skip evaluation for this example
            continue

        for j, label in enumerate(gold_labels):
            is_hallucinated = label in ("hallucinated", "unsupported", "contradicted")
            predicted_unsupported_flag = j in predicted_unsupported

            if is_hallucinated and predicted_unsupported_flag:
                true_positives += 1
            elif not is_hallucinated and predicted_unsupported_flag:
                false_positives += 1
            elif is_hallucinated and not predicted_unsupported_flag:
                false_negatives += 1
            else:
                true_negatives += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "num_examples": len(examples),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "latency_p50_s": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0,
        "nli_model": nli_model,
    }


def main():
    parser = argparse.ArgumentParser(description="Run RAGTruth benchmark")
    parser.add_argument("--data-dir", type=Path, required=True, help="RAGTruth data directory")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/ragtruth.json"), help="Output file")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base", help="NLI model")
    args = parser.parse_args()

    print(f"Loading RAGTruth from {args.data_dir}...")
    examples = load_ragtruth(args.data_dir)
    print(f"Loaded {len(examples)} examples")

    print("Running evaluation...")
    results = evaluate_ragtruth(examples, nli_model=args.nli_model)

    print(f"\nResults:")
    print(f"  Precision: {results['precision']}")
    print(f"  Recall:    {results['recall']}")
    print(f"  F1:        {results['f1']}")
    print(f"  Latency:   {results['latency_mean_s']}s mean, {results['latency_p50_s']}s p50")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
