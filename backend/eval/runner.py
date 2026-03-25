"""RAGAS evaluation runner for benchmarking chunking strategies and retrieval methods."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import structlog

DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logger = structlog.get_logger()


async def run_evaluation(
    dataset_name: str = "sample_qa",
    chunking_strategy: str = "recursive",
    retrieval_strategy: str = "hybrid",
    api_url: str = "http://localhost:8000",
) -> dict:
    """Run RAGAS evaluation against the live API."""
    import httpx

    dataset_path = DATASETS_DIR / f"{dataset_name}.json"
    with open(dataset_path) as f:
        samples = json.load(f)

    logger.info(
        "starting evaluation",
        dataset=dataset_name,
        chunking=chunking_strategy,
        retrieval=retrieval_strategy,
        samples=len(samples),
    )

    questions = []
    ground_truths = []
    answers = []
    contexts_list = []

    async with httpx.AsyncClient(base_url=api_url, timeout=60.0) as client:
        for sample in samples:
            try:
                resp = await client.post(
                    "/api/query",
                    json={
                        "question": sample["question"],
                        "strategy": retrieval_strategy,
                        "top_k": 5,
                    },
                )
                resp.raise_for_status()
                result = resp.json()

                questions.append(sample["question"])
                ground_truths.append(sample["ground_truth"])
                answers.append(result["answer"])
                contexts_list.append([s["content"] for s in result["sources"]])
            except Exception as e:
                logger.warning("sample failed", question=sample["question"][:50], error=str(e))

    if not questions:
        logger.error("no samples completed successfully")
        return {}

    metrics = await _compute_ragas_metrics(questions, answers, contexts_list, ground_truths)

    result_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": dataset_name,
        "chunking_strategy": chunking_strategy,
        "retrieval_strategy": retrieval_strategy,
        "sample_count": len(questions),
        "metrics": metrics,
    }

    result_file = RESULTS_DIR / f"{chunking_strategy}_{retrieval_strategy}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info("evaluation complete", metrics=metrics, output=str(result_file))
    return result_data


async def _compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
    ground_truths: list[str],
) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        return {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevance": round(float(result["answer_relevancy"]), 4),
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall": round(float(result["context_recall"]), 4),
        }
    except ImportError:
        logger.warning("ragas not installed, returning placeholder metrics")
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }


def benchmark_chunking_strategies() -> None:
    """Run evaluation across all 3 chunking strategies and print comparison table."""
    strategies = ["fixed", "recursive", "semantic"]
    results = []

    for strategy in strategies:
        print(f"\nEvaluating chunking strategy: {strategy}")
        result = asyncio.run(run_evaluation(
            chunking_strategy=strategy,
            retrieval_strategy="hybrid",
        ))
        if result:
            results.append(result)

    if not results:
        print("No results to compare.")
        return

    print("\n" + "=" * 80)
    print("CHUNKING STRATEGY BENCHMARK RESULTS")
    print("=" * 80)
    header = f"{'Strategy':<15} {'Faithfulness':>13} {'Answer Rel.':>12} {'Ctx Prec.':>10} {'Ctx Recall':>11}"
    print(header)
    print("-" * 80)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['chunking_strategy']:<15} "
            f"{m['faithfulness']:>13.4f} "
            f"{m['answer_relevance']:>12.4f} "
            f"{m['context_precision']:>10.4f} "
            f"{m['context_recall']:>11.4f}"
        )
    print("=" * 80)

    best_path = RESULTS_DIR / "benchmark_comparison.json"
    with open(best_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {best_path}")


def _parse_fail_below(args: list[str]) -> dict[str, float]:
    """Parse --fail-below metric=threshold ... from argv."""
    thresholds: dict[str, float] = {}
    collecting = False
    for arg in args:
        if arg == "--fail-below":
            collecting = True
            continue
        if collecting and "=" in arg and not arg.startswith("--"):
            metric, value = arg.split("=", 1)
            thresholds[metric.strip()] = float(value.strip())
        else:
            collecting = False
    return thresholds


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        # Run single-strategy eval with optional quality gate
        thresholds = _parse_fail_below(sys.argv)
        result = asyncio.run(run_evaluation())
        if result and thresholds:
            metrics = result.get("metrics", {})
            failures = []
            for metric, threshold in thresholds.items():
                actual = metrics.get(metric, 0.0)
                status = "PASS" if actual >= threshold else "FAIL"
                print(f"  {status}  {metric}: {actual:.4f} (threshold {threshold:.2f})")
                if actual < threshold:
                    failures.append(f"{metric}={actual:.4f} < {threshold:.2f}")
            if failures:
                print(f"\nQuality gate FAILED: {', '.join(failures)}")
                sys.exit(1)
            else:
                print("\nQuality gate PASSED")
        else:
            benchmark_chunking_strategies()
    else:
        result = asyncio.run(run_evaluation())
        print(json.dumps(result, indent=2))
