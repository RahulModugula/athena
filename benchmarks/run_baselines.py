#!/usr/bin/env python3
"""Baseline comparison runner for athena-verify benchmarks.

Runs the same evaluation protocol used in run_ragtruth.py / run_halueval.py / run_factscore.py
but using baseline methods instead of athena-verify, enabling head-to-head comparison.

Supported baselines:
    - ragas: Ragas faithfulness metric (requires ragas + openai)
    - gpt4_judge: GPT-4-as-judge (requires openai)
    - vectara_hhem: Vectara HHEM cross-encoder (requires sentence-transformers)

Usage:
    python benchmarks/run_baselines.py --method ragas \\
        --data-dir ./data/ragtruth \\
        --output benchmarks/results/baseline_ragas.json
    python benchmarks/run_baselines.py --method gpt4_judge \\
        --data-dir ./data/ragtruth \\
        --output benchmarks/results/baseline_gpt4.json
    python benchmarks/run_baselines.py --method vectara_hhem \\
        --data-dir ./data/ragtruth \\
        --output benchmarks/results/baseline_hhem.json
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


def load_dataset(data_dir: Path) -> list[dict]:
    examples = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def _ragas_faithfulness(
    question: str,
    answer: str,
    context: list[str],
) -> float:
    """Score using Ragas faithfulness metric."""
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness
    except ImportError as e:
        raise ImportError("ragas is required for this baseline. pip install ragas") from e

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=context,
    )
    import asyncio

    scorer = Faithfulness()
    result = asyncio.get_event_loop().run_until_complete(scorer.single_turn_ascore(sample))
    return float(result)


def _gpt4_judge(
    question: str,
    answer: str,
    context: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> tuple[float, str]:
    """Score using GPT-4-as-judge."""
    import json as _json

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai is required for this baseline. pip install openai") from e

    client = OpenAI(api_key=api_key, timeout=60)

    prompt = (
        "You are a verification judge. Given context and a claim, "
        "determine whether the claim is supported by the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Claim: {answer}\n\n"
        f"Question (for context): {question}\n\n"
        "For each sentence in the claim, respond with a JSON object:\n"
        '{\n  "sentences": [\n'
        '    {"text": "sentence text", "supported": true/false, '
        '"confidence": 0.0-1.0}\n'
        "  ]\n}\n\n"
        "Be strict: only mark as supported if directly inferable from the context."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    text = response.choices[0].message.content or ""

    try:
        result = _json.loads(text)
        sentences = result.get("sentences", [])
        if not sentences:
            return 0.5, "parse_error_no_sentences"
        supported_count = sum(1 for s in sentences if s.get("supported", False))
        score = supported_count / len(sentences)
        return score, "ok"
    except (_json.JSONDecodeError, KeyError):
        return 0.5, "parse_error"


def _vectara_hhem(
    premise: str,
    hypothesis: str,
) -> float:
    """Score using Vectara HHEM model."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. pip install sentence-transformers"
        ) from e

    model = CrossEncoder("vectara/hallucination_evaluation_model")
    score = model.predict([[premise, hypothesis]])
    return float(score[0]) if hasattr(score, "__len__") else float(score)


def evaluate_ragas_baseline(
    examples: list[dict],
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run Ragas faithfulness baseline."""
    latencies: list[float] = []
    scores: list[float] = []
    total_cost_estimate = 0.0

    for i, ex in enumerate(examples):
        question = ex.get("prompt", ex.get("query", ""))
        answer = ex.get("response", ex.get("right_answer", ""))
        context = [ex.get("source", ex.get("context", ex.get("knowledge", "")))]

        start = time.time()
        score = _ragas_faithfulness(question, answer, context)
        latencies.append(time.time() - start)
        scores.append(score)

        total_cost_estimate += len(question.split()) * 0.00001 + len(answer.split()) * 0.00003

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    sorted_lat = sorted(latencies)
    return {
        "method": "ragas_faithfulness",
        "num_examples": len(examples),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0,
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0,
        "latency_p50_s": round(_percentile(sorted_lat, 0.50), 3),
        "latency_p95_s": round(_percentile(sorted_lat, 0.95), 3),
        "estimated_cost_usd": round(total_cost_estimate, 2),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": 42,
        },
    }


def evaluate_gpt4_baseline(
    examples: list[dict],
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run GPT-4-as-judge baseline."""
    latencies: list[float] = []
    scores: list[float] = []
    total_tokens_estimate = 0

    for i, ex in enumerate(examples):
        question = ex.get("prompt", ex.get("query", ""))
        answer = ex.get("response", ex.get("right_answer", ""))
        context = ex.get("source", ex.get("context", ex.get("knowledge", "")))

        start = time.time()
        score, status = _gpt4_judge(question, answer, context, model=model, api_key=api_key)
        latencies.append(time.time() - start)
        scores.append(score)

        total_tokens_estimate += len((question + answer + context).split()) * 2

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    sorted_lat = sorted(latencies)
    cost_per_1k_tokens = 0.00015 if "mini" in model else 0.005
    estimated_cost = total_tokens_estimate / 1000 * cost_per_1k_tokens

    return {
        "method": f"gpt4_judge_{model}",
        "num_examples": len(examples),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0,
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0,
        "latency_p50_s": round(_percentile(sorted_lat, 0.50), 3),
        "latency_p95_s": round(_percentile(sorted_lat, 0.95), 3),
        "estimated_cost_usd": round(estimated_cost, 4),
        "cost_per_1k_sentences": f"${cost_per_1k_tokens * 200:.4f} estimated",
        "environment": {
            "python": platform.python_version(),
            "model": model,
            "platform": platform.platform(),
            "seed": 42,
        },
    }


def evaluate_hhem_baseline(
    examples: list[dict],
) -> dict[str, Any]:
    """Run Vectara HHEM baseline."""
    latencies: list[float] = []
    scores: list[float] = []

    for i, ex in enumerate(examples):
        answer = ex.get("response", ex.get("right_answer", ""))
        context = ex.get("source", ex.get("context", ex.get("knowledge", "")))

        start = time.time()
        score = _vectara_hhem(context, answer)
        latencies.append(time.time() - start)
        scores.append(score)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    sorted_lat = sorted(latencies)

    return {
        "method": "vectara_hhem",
        "num_examples": len(examples),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0,
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0,
        "latency_p50_s": round(_percentile(sorted_lat, 0.50), 3),
        "latency_p95_s": round(_percentile(sorted_lat, 0.95), 3),
        "cost_per_1k_sentences": "local_model (no API cost)",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": 42,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument(
        "--method",
        required=True,
        choices=["ragas", "gpt4_judge", "vectara_hhem"],
        help="Baseline method to evaluate",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--output", type=Path, default=None, help="Output file")
    parser.add_argument("--api-key", default=None, help="API key (for ragas/gpt4)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model for GPT-4 judge")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)

    if args.output is None:
        args.output = Path(f"benchmarks/results/baseline_{args.method}.json")

    print(f"Loading dataset from {args.data_dir}...")
    examples = load_dataset(args.data_dir)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples")

    print(f"Running {args.method} baseline...")
    if args.method == "ragas":
        results = evaluate_ragas_baseline(examples, api_key=args.api_key)
    elif args.method == "gpt4_judge":
        results = evaluate_gpt4_baseline(examples, model=args.model, api_key=args.api_key)
    elif args.method == "vectara_hhem":
        results = evaluate_hhem_baseline(examples)

    print(f"\nResults ({args.method}):")
    for k, v in results.items():
        if k != "environment":
            print(f"  {k}: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
