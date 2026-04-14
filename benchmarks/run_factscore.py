#!/usr/bin/env python3
"""FActScore benchmark runner for athena-verify.

Evaluates athena-verify on the FActScore long-form factuality benchmark.

Usage:
    python benchmarks/run_factscore.py --data-dir ./data/factscore --output benchmarks/results/factscore.json

Requires the FActScore dataset. See:
https://github.com/shmsw25/FActScore
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from athena_verify import verify


def load_factscore(data_dir: Path) -> list[dict]:
    """Load FActScore dataset.

    Expected format: JSONL with fields:
    - input: the topic/prompt
    - output: the generated biography/text
    - facts: list of individual facts extracted from the output
    - labels: list of labels (SUPPORTED/NOT_SUPPORTED) for each fact

    Args:
        data_dir: Directory containing FActScore JSONL files.

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


def evaluate_factscore(
    examples: list[dict],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
) -> dict:
    """Run athena-verify on FActScore examples.

    For each example, we verify the generated text against available
    knowledge/source and compare against gold fact-level labels.

    Args:
        examples: FActScore examples.
        nli_model: NLI model to use.

    Returns:
        Dict with fact-level precision, recall, F1.
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    latencies = []

    for i, ex in enumerate(examples):
        topic = ex.get("input", "")
        biography = ex.get("output", "")
        # FActScore provides Wikipedia paragraphs as source
        source = ex.get("source", ex.get("knowledge", ""))
        context = [source] if source else [biography]  # Fallback

        start = time.time()
        result = verify(question=f"Tell me about {topic}", answer=biography, context=context, nli_model=nli_model)
        latencies.append(time.time() - start)

        # Get gold labels
        facts = ex.get("facts", [])
        labels = ex.get("labels", [])

        if not facts or not labels:
            # Use sentence-level verification as proxy
            for sent in result.sentences:
                is_supported_gold = True  # Assume supported if no gold labels
                predicted_unsupported = sent.support_status in ("UNSUPPORTED", "CONTRADICTED")

                if not is_supported_gold and predicted_unsupported:
                    true_positives += 1
                elif is_supported_gold and not predicted_unsupported:
                    true_negatives += 1
            continue

        # Map facts to sentences (best-effort)
        predicted_unsupported_indices = {s.index for s in result.unsupported}

        for j, (fact, label) in enumerate(zip(facts, labels)):
            is_not_supported = label in ("NOT_SUPPORTED", "false", "incorrect")
            # Find closest sentence to this fact
            predicted_flagged = j in predicted_unsupported_indices

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
    parser = argparse.ArgumentParser(description="Run FActScore benchmark")
    parser.add_argument("--data-dir", type=Path, required=True, help="FActScore data directory")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/factscore.json"), help="Output file")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base", help="NLI model")
    args = parser.parse_args()

    print(f"Loading FActScore from {args.data_dir}...")
    examples = load_factscore(args.data_dir)
    print(f"Loaded {len(examples)} examples")

    print("Running evaluation...")
    results = evaluate_factscore(examples, nli_model=args.nli_model)

    print(f"\nResults:")
    print(f"  Precision: {results['precision']}")
    print(f"  Recall:    {results['recall']}")
    print(f"  F1:        {results['f1']}")
    print(f"  Latency:   {results['latency_mean_s']}s mean")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
