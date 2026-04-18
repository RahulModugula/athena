#!/usr/bin/env python3
"""Hybrid NLI + LLM-judge benchmark for athena-verify.

Runs NLI first, then sends borderline sentences (NLI score 0.3–0.7) to
an LLM judge (gemma-4-31b-it via LM Studio at localhost:1234). Combines
NLI and LLM scores with configurable weights.

Usage:
    python benchmarks/run_hybrid_eval.py
    python benchmarks/run_hybrid_eval.py --nli-weight 0.6 --llm-weight 0.4
    python benchmarks/run_hybrid_eval.py --borderline-lo 0.3 --borderline-hi 0.7
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from athena_verify import verify
from athena_verify.nli import get_nli_model
from athena_verify.parser import split_sentences
from benchmarks.run_full_eval import build_dataset, compute_ece

LLM_JUDGE_URL = "http://localhost:1234/v1"
LLM_MODEL = "gemma-4-31b-it"
BORDERLINE_LO = 0.3
BORDERLINE_HI = 0.7
NLI_WEIGHT = 0.6
LLM_WEIGHT = 0.4
TRUST_THRESHOLD = 0.70


def query_llm_judge(context: str, claim: str) -> float:
    """Ask the LLM judge if a claim is supported. Returns 1.0 or 0.0."""
    from openai import OpenAI

    client = OpenAI(base_url=LLM_JUDGE_URL, api_key="lm-studio")

    prompt = (
        f"Is this claim fully supported by the context? "
        f"Context: {context}\nClaim: {claim}\n"
        f"Answer SUPPORTED or UNSUPPORTED."
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip().upper()
        if (
            "SUPPORTED" in text
            and "UNSUPPORTED" not in text
            and "NOT" not in text
            and text == "SUPPORTED"
        ):
            return 1.0
        return 0.0
    except Exception as e:
        print(f"  [LLM judge error] {e}")
        return -1.0


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def evaluate_hybrid(
    cases: list[dict],
    nli_model: str,
    nli_weight: float,
    llm_weight: float,
    border_lo: float,
    border_hi: float,
) -> dict[str, Any]:
    category_metrics: dict[str, dict[str, Any]] = {}
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tn = 0
    all_predicted_scores: list[float] = []
    all_actual_labels: list[bool] = []
    all_latencies: list[float] = []
    llm_calls = 0
    llm_time_total = 0.0

    print("Loading NLI model...")
    get_nli_model(nli_model)

    for i, case in enumerate(cases):
        question = case["question"]
        answer = case["answer"]
        context = case["context"]
        gold_labels = case["sentence_labels"]

        start = time.time()

        verify(
            question=question,
            answer=answer,
            context=context,
            nli_model=nli_model,
            trust_threshold=0.5,
        )

        sentences = split_sentences(answer)

        nli_pairs = []
        for sent in sentences:
            best_ctx = context[0] if context else ""
            best_score = -1.0
            for ctx in context:
                score = get_nli_model(nli_model).predict([[ctx, sent]])
                raw = float(score[0][0]) if hasattr(score[0], "__len__") else float(score[0])
                if raw > best_score:
                    best_score = raw
                    best_ctx = ctx
            nli_pairs.append((best_ctx, sent, best_score))

        final_flags: list[bool] = []
        final_scores: list[float] = []

        for best_ctx, sent, nli_raw in nli_pairs:
            if border_lo <= nli_raw <= border_hi:
                t0 = time.time()
                llm_score = query_llm_judge(best_ctx, sent)
                llm_dt = time.time() - t0
                llm_calls += 1
                llm_time_total += llm_dt

                if llm_score < 0:
                    combined = nli_raw
                else:
                    combined = nli_weight * nli_raw + llm_weight * llm_score
            else:
                combined = nli_raw

            final_scores.append(combined)
            is_unsupported = combined < 0.5
            final_flags.append(is_unsupported)

        latency = time.time() - start

        for j, gold in enumerate(gold_labels):
            if j >= len(final_flags):
                continue
            is_hallucinated = gold == "hallucinated"
            predicted_flag = final_flags[j]

            all_predicted_scores.append(final_scores[j] if j < len(final_scores) else 0.5)
            all_latencies.append(latency / max(len(final_flags), 1))
            all_actual_labels.append(not is_hallucinated)

            cat = case["category"]
            if cat not in category_metrics:
                category_metrics[cat] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            cm = category_metrics[cat]

            if is_hallucinated and predicted_flag:
                cm["tp"] += 1
                all_tp += 1
            elif not is_hallucinated and predicted_flag:
                cm["fp"] += 1
                all_fp += 1
            elif is_hallucinated and not predicted_flag:
                cm["fn"] += 1
                all_fn += 1
            else:
                cm["tn"] += 1
                all_tn += 1

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(cases)} cases (LLM calls so far: {llm_calls})")

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    per_category = {}
    for cat, cm in sorted(category_metrics.items()):
        p, r, f = _prf(cm["tp"], cm["fp"], cm["fn"])
        per_category[cat] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "confusion_matrix": dict(cm),
            "num_cases": sum(1 for c in cases if c["category"] == cat),
        }

    overall_p, overall_r, overall_f = _prf(all_tp, all_fp, all_fn)
    ece = compute_ece(all_predicted_scores, all_actual_labels)
    sorted_lat = sorted(all_latencies)

    return {
        "nli_model": nli_model,
        "llm_model": LLM_MODEL,
        "nli_weight": nli_weight,
        "llm_weight": llm_weight,
        "borderline_range": [border_lo, border_hi],
        "overall": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f, 4),
            "ece": round(ece, 4),
            "confusion_matrix": {
                "true_positives": all_tp,
                "false_positives": all_fp,
                "false_negatives": all_fn,
                "true_negatives": all_tn,
            },
            "num_cases": len(cases),
            "num_sentences": sum(len(c["sentence_labels"]) for c in cases),
            "latency_p50_ms": round(_percentile(sorted_lat, 0.50) * 1000, 1),
            "latency_p95_ms": round(_percentile(sorted_lat, 0.95) * 1000, 1),
            "llm_judge_calls": llm_calls,
            "llm_judge_avg_ms": round(llm_time_total / max(llm_calls, 1) * 1000, 1),
        },
        "per_category": per_category,
    }


def print_results(results: dict[str, Any]) -> None:
    o = results["overall"]
    print("\n## Hybrid NLI + LLM-Judge Results\n")
    print(f"NLI model:        {results['nli_model']}")
    print(f"LLM judge:        {results['llm_model']}")
    print(f"Borderline range: {results['borderline_range']}")
    print(f"Weights:          NLI={results['nli_weight']}, LLM={results['llm_weight']}")
    print()
    print(f"Overall Precision:  {o['precision']:.1%}")
    print(f"Overall Recall:     {o['recall']:.1%}")
    print(f"Overall F1:         {o['f1']:.1%}")
    print(f"ECE:                {o['ece']:.4f}")
    print(f"Latency p50:        {o['latency_p50_ms']:.0f}ms")
    print(f"Latency p95:        {o['latency_p95_ms']:.0f}ms")
    print(f"LLM judge calls:    {o['llm_judge_calls']}")
    print(f"LLM avg latency:    {o['llm_judge_avg_ms']:.0f}ms")

    print("\n### Per-Category Results\n")
    print("| Category | Precision | Recall | F1 | Cases |")
    print("|----------|-----------|--------|----|-------|")
    for cat, cm in sorted(results["per_category"].items()):
        print(
            f"| {cat} | {cm['precision']:.1%} "
            f"| {cm['recall']:.1%} | {cm['f1']:.1%} "
            f"| {cm['num_cases']} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run hybrid NLI + LLM-judge benchmark for athena-verify"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/hybrid_eval.json"),
        help="Output JSON file path",
    )
    parser.add_argument("--nli-weight", type=float, default=NLI_WEIGHT)
    parser.add_argument("--llm-weight", type=float, default=LLM_WEIGHT)
    parser.add_argument("--borderline-lo", type=float, default=BORDERLINE_LO)
    parser.add_argument("--borderline-hi", type=float, default=BORDERLINE_HI)
    parser.add_argument("--nli-model", type=str, default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass

    print("Building synthetic benchmark dataset...")
    cases = build_dataset()

    category_counts: dict[str, int] = {}
    for c in cases:
        category_counts[c["category"]] = category_counts.get(c["category"], 0) + 1
    print(f"Total test cases: {len(cases)}")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    total_sentences = sum(len(c["sentence_labels"]) for c in cases)
    print(f"Total sentences: {total_sentences}")

    print(f"\nRunning hybrid eval (NLI weight={args.nli_weight}, LLM weight={args.llm_weight})...")
    results = evaluate_hybrid(
        cases,
        nli_model=args.nli_model,
        nli_weight=args.nli_weight,
        llm_weight=args.llm_weight,
        border_lo=args.borderline_lo,
        border_hi=args.borderline_hi,
    )

    output_data = {
        "dataset": {
            "total_cases": len(cases),
            "total_sentences": total_sentences,
            "categories": category_counts,
        },
        "results": results,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": args.seed,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print_results(results)


if __name__ == "__main__":
    main()
