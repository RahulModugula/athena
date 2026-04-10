"""Run LegalBench-RAG evaluation on Athena."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


async def run_legalbench_eval(
    api_url: str = "http://localhost:8000",
    max_questions: int | None = None,
    with_verification: bool = True,
) -> dict[str, Any]:
    """Run Athena against LegalBench-RAG dataset.

    Args:
        api_url: Base URL of Athena API
        max_questions: Max questions to evaluate (None = all)
        with_verification: Include verification in evaluation

    Returns:
        Results dict with metrics
    """
    from eval.legalbench.loader import load_legalbench_qa

    qa_pairs = load_legalbench_qa()
    if max_questions:
        qa_pairs = qa_pairs[:max_questions]

    if not qa_pairs:
        logger.warning("no QA pairs to evaluate")
        return {}

    results = []
    metrics = {
        "total_questions": len(qa_pairs),
        "correct_at_1": 0,
        "correct_at_5": 0,
        "mean_trust_score": 0.0,
        "mean_latency_ms": 0.0,
    }

    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair.get("question", "")
        expected_answer = qa_pair.get("answer", "")

        logger.info(f"evaluating {i+1}/{len(qa_pairs)}", question=question[:80])

        start_time = time.time()
        try:
            # Query Athena
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{api_url}/api/query",
                    json={
                        "question": question,
                        "strategy": "hybrid",
                        "top_k": 5,
                    },
                    timeout=30,
                )
                if response.status_code != 200:
                    logger.error(f"query failed: {response.status_code}")
                    continue

                data = response.json()
                answer = data.get("answer", "")
                trust_score = data.get("trust_score", 0.0)
                sources = data.get("sources", [])

        except Exception as e:
            logger.error("evaluation error", error=str(e))
            continue

        elapsed_ms = (time.time() - start_time) * 1000

        # Simple precision check: does answer contain key terms from expected?
        answer_lower = answer.lower()
        expected_lower = expected_answer.lower()
        precision_at_1 = (
            1.0
            if any(
                term in answer_lower
                for term in expected_lower.split()[:5]
                if len(term) > 3
            )
            else 0.0
        )

        results.append({
            "question": question,
            "expected": expected_answer,
            "answer": answer,
            "precision_at_1": precision_at_1,
            "trust_score": trust_score,
            "latency_ms": elapsed_ms,
            "sources_count": len(sources),
        })

        metrics["correct_at_1"] += precision_at_1
        metrics["mean_trust_score"] += trust_score
        metrics["mean_latency_ms"] += elapsed_ms

    # Compute final metrics
    if results:
        metrics["correct_at_1"] /= len(results)
        metrics["mean_trust_score"] /= len(results)
        metrics["mean_latency_ms"] /= len(results)

    logger.info(
        "evaluation complete",
        precision_at_1=metrics["correct_at_1"],
        mean_trust_score=metrics["mean_trust_score"],
    )

    return {
        "metrics": metrics,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LegalBench-RAG evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    eval_results = await run_legalbench_eval(
        api_url=args.api_url,
        max_questions=args.max_questions,
    )

    # Save results
    output_path = args.output or f"eval/results/legalbench_{int(time.time())}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"\nMetrics:\n{json.dumps(eval_results['metrics'], indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
