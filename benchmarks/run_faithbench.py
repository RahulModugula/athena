#!/usr/bin/env python3
"""FaithBench benchmark runner for athena-verify.

Evaluates athena-verify on the FaithBench dataset: comprehensive faithfulness
evaluation across multiple domains and hallucination types.

FaithBench is available at: https://huggingface.co/datasets/CogComp/FaithBench

Usage:
    # With FaithBench dataset downloaded
    python benchmarks/run_faithbench.py --data-dir ./data/faithbench \\
        --output benchmarks/results/faithbench.json

    # Without dataset (uses synthetic data for testing)
    python benchmarks/run_faithbench.py --synthetic \\
        --output benchmarks/results/faithbench_synthetic.json

Evaluation protocol:
    1. Load dataset or generate synthetic examples.
    2. For each example, run athena-verify verify() on (question, answer, context).
    3. Compare predicted unsupported sentences against gold labels.
    4. Compute precision, recall, F1, ECE, latency (p50/p95), cost.
    5. All runs are deterministic (seed=42). Document hardware.

Success gate: Athena achieves >85% F1 on real faithfulness hallucinations.
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
    """Compute Expected Calibration Error.

    Args:
        predicted_scores: Predicted trust scores per sentence.
        actual_labels: True labels (True = faithful, False = hallucinated).
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


def load_faithbench(data_dir: Path) -> list[dict]:
    """Load FaithBench dataset.

    Expected format: JSONL files with fields:
    - question/prompt: the question
    - answer/response: the LLM response to verify
    - context/source: the source context
    - label: faithfulness label ('faithful', 'hallucinated', 'unsupported')
    - sentence_labels: per-sentence labels (optional)

    Args:
        data_dir: Directory containing FaithBench JSONL files.

    Returns:
        List of example dicts.
    """
    examples = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

    return examples


def generate_synthetic_faithbench(num_examples: int = 100) -> list[dict]:
    """Generate synthetic FaithBench-like examples for testing.

    Args:
        num_examples: Number of examples to generate.

    Returns:
        List of example dicts with question, answer, context, and labels.
    """
    examples = []

    faithful_data = [
        {
            "q": "What is the capital of France?",
            "a": "Paris is the capital of France.",
            "ctx": "Paris is the capital city of France located in north-central France.",
        },
        {
            "q": "When was Google founded?",
            "a": "Google was founded in 1998.",
            "ctx": "Google was established in 1998 by Larry Page and Sergey Brin.",
        },
        {
            "q": "What is the chemical symbol for Oxygen?",
            "a": "The chemical symbol for Oxygen is O.",
            "ctx": "The element Oxygen has the chemical symbol O.",
        },
        {
            "q": "Who was the first President of the United States?",
            "a": "George Washington was the first President of the United States.",
            "ctx": "George Washington served as the first President of the United States from 1789 to 1797.",
        },
        {
            "q": "What year did World War II end?",
            "a": "World War II ended in 1945.",
            "ctx": "World War II officially ended in 1945 with the surrender of Japan.",
        },
    ]

    hallucinated_data = [
        {
            "q": "What is the capital of France?",
            "a": "London is the capital of France.",
            "ctx": "Paris is the capital city of France located in north-central France.",
        },
        {
            "q": "When was Google founded?",
            "a": "Google was founded in 1985.",
            "ctx": "Google was established in 1998 by Larry Page and Sergey Brin.",
        },
        {
            "q": "What is the chemical symbol for Oxygen?",
            "a": "The chemical symbol for Oxygen is X.",
            "ctx": "The element Oxygen has the chemical symbol O.",
        },
        {
            "q": "Who was the first President of the United States?",
            "a": "Thomas Jefferson was the first President of the United States.",
            "ctx": "George Washington served as the first President of the United States from 1789 to 1797.",
        },
        {
            "q": "What year did World War II end?",
            "a": "World War II ended in 1950.",
            "ctx": "World War II officially ended in 1945 with the surrender of Japan.",
        },
    ]

    # Generate faithful examples
    for i in range(num_examples // 2):
        template = faithful_data[i % len(faithful_data)]
        examples.append(
            {
                "question": template["q"],
                "answer": template["a"],
                "context": [template["ctx"]],
                "label": "faithful",
                "domain": "general",
            }
        )

    # Generate hallucinated examples
    for i in range(num_examples - num_examples // 2):
        template = hallucinated_data[i % len(hallucinated_data)]
        examples.append(
            {
                "question": template["q"],
                "answer": template["a"],
                "context": [template["ctx"]],
                "label": "hallucinated",
                "domain": "general",
            }
        )

    return examples


def evaluate_faithbench(
    examples: list[dict],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    trust_threshold: float = 0.70,
) -> dict[str, Any]:
    """Run athena-verify on FaithBench examples.

    Args:
        examples: FaithBench examples.
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
        question = ex.get("question", ex.get("prompt", ""))
        answer = ex.get("answer", ex.get("response", ""))
        context = ex.get("context", [ex.get("source", "")])
        if isinstance(context, str):
            context = [context]

        gold_label = ex.get("label", "").lower()
        is_hallucinated = gold_label in ("hallucinated", "unsupported", "contradicted", "unfaithful")

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

        for sent in result.sentences:
            all_predicted_scores.append(sent.trust_score)
            all_actual_labels.append(not is_hallucinated)

        predicted_flag = not result.verification_passed

        if is_hallucinated and predicted_flag:
            true_positives += 1
        elif not is_hallucinated and predicted_flag:
            false_positives += 1
        elif is_hallucinated and not predicted_flag:
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
    parser = argparse.ArgumentParser(description="Run FaithBench benchmark")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="FaithBench data directory (optional)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real dataset",
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=100,
        help="Number of synthetic examples to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/faithbench.json"),
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

    if args.synthetic or not args.data_dir:
        print(f"Generating {args.num_synthetic} synthetic examples...")
        examples = generate_synthetic_faithbench(args.num_synthetic)
        print(f"Generated {len(examples)} examples")
    else:
        print(f"Loading FaithBench from {args.data_dir}...")
        examples = load_faithbench(args.data_dir)
        print(f"Loaded {len(examples)} examples")

    print("Running evaluation...")
    results = evaluate_faithbench(examples, nli_model=args.nli_model)

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
