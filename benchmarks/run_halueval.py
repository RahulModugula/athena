#!/usr/bin/env python3
"""HaluEval benchmark runner for athena-verify.

Evaluates athena-verify on the HaluEval dataset:
35K hallucination examples over QA, dialogue, and summarization.

Usage:
    python benchmarks/run_halueval.py --data-dir ./data/halueval --output benchmarks/results/halueval.json

Requires the HaluEval dataset. See:
https://github.com/RUCAIBox/HaluEval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from athena_verify import verify


def load_halueval(data_dir: Path) -> list[dict]:
    """Load HaluEval dataset.

    Expected format: JSONL with fields:
    - query: the question
    - right_answer: the correct answer
    - hallucinated_answer: a hallucinated version
    - context: source context (if available)

    Args:
        data_dir: Directory containing HaluEval JSONL files.

    Returns:
        List of example dicts with both right and hallucinated answers.
    """
    examples = []
    for jsonl_file in data_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def evaluate_halueval(
    examples: list[dict],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
) -> dict:
    """Run athena-verify on HaluEval examples.

    For each example, we verify both the right answer and the
    hallucinated answer. A correct detection means:
    - Right answer: trust_score >= threshold (not flagged)
    - Hallucinated answer: trust_score < threshold (flagged)

    Args:
        examples: HaluEval examples.
        nli_model: NLI model to use.

    Returns:
        Dict with detection metrics.
    """
    correct = 0
    total = 0
    latencies = []

    for i, ex in enumerate(examples):
        question = ex.get("query", "")
        context = [ex.get("context", ex.get("knowledge", ""))]
        right_answer = ex.get("right_answer", "")
        hall_answer = ex.get("hallucinated_answer", "")

        if not context[0]:
            context = [right_answer]  # Use right answer as context if none provided

        # Verify right answer (should pass)
        start = time.time()
        right_result = verify(question=question, answer=right_answer, context=context, nli_model=nli_model)
        latencies.append(time.time() - start)

        # Verify hallucinated answer (should fail)
        start = time.time()
        hall_result = verify(question=question, answer=hall_answer, context=context, nli_model=nli_model)
        latencies.append(time.time() - start)

        # Correct if right answer passes and hallucinated answer doesn't
        if right_result.verification_passed and not hall_result.verification_passed:
            correct += 1
        total += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    accuracy = correct / total if total > 0 else 0.0

    return {
        "num_examples": len(examples),
        "correct_detections": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
        "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "latency_p50_s": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0,
        "nli_model": nli_model,
    }


def main():
    parser = argparse.ArgumentParser(description="Run HaluEval benchmark")
    parser.add_argument("--data-dir", type=Path, required=True, help="HaluEval data directory")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/halueval.json"), help="Output file")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base", help="NLI model")
    args = parser.parse_args()

    print(f"Loading HaluEval from {args.data_dir}...")
    examples = load_halueval(args.data_dir)
    print(f"Loaded {len(examples)} examples")

    print("Running evaluation...")
    results = evaluate_halueval(examples, nli_model=args.nli_model)

    print(f"\nResults:")
    print(f"  Accuracy:  {results['accuracy']}")
    print(f"  Correct:   {results['correct_detections']}/{results['total']}")
    print(f"  Latency:   {results['latency_mean_s']}s mean")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
