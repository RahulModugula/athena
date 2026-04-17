"""Natural Language Inference (NLI) model for entailment checking.

Uses a cross-encoder model to compute entailment probability between
a premise (context) and a hypothesis (claim/sentence).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import structlog

logger = structlog.get_logger()

NLI_MODEL_ALIASES: dict[str, str] = {
    "default": "cross-encoder/nli-deberta-v3-base",
    "lightweight": "cross-encoder/nli-MiniLM2-L6-H768",
    "vectara": "vectara/hallucination_evaluation_model",
    "deberta-base": "MoritzLaworr/NLI-deberta-base",
}

_nli_cache: dict[str, Any] = {}


def resolve_nli_model(model_name: str) -> str:
    """Resolve a model alias to a full HuggingFace model identifier.

    Args:
        model_name: A model name or alias (e.g. "lightweight").

    Returns:
        The resolved HuggingFace model identifier.
    """
    return NLI_MODEL_ALIASES.get(model_name, model_name)


def get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-base") -> Any:
    """Load the NLI cross-encoder model (lazy, cached).

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.

    Returns:
        CrossEncoder model instance.
    """
    resolved = resolve_nli_model(model_name)

    if resolved in _nli_cache:
        return _nli_cache[resolved]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for NLI scoring. "
            "Install with: pip install athena-verify[nli]"
        ) from e

    logger.info("loading_nli_model", model=resolved, alias=model_name)
    _nli_cache[resolved] = CrossEncoder(resolved)
    return _nli_cache[resolved]


def _softmax_entailment(logits: Any) -> float:
    """Convert 3-class NLI logits to entailment probability using softmax."""
    row = list(logits)
    max_val = max(row)
    exp_vals = [math.exp(v - max_val) for v in row]
    total = sum(exp_vals)
    return exp_vals[0] / total


def compute_entailment_score(
    premise: str,
    hypothesis: str,
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> float:
    """Compute entailment probability between premise and hypothesis.

    Args:
        premise: The supporting text (context chunk).
        hypothesis: The claim to verify (sentence).
        model_name: Cross-encoder model to use.

    Returns:
        Probability of entailment (0.0-1.0).
    """
    model = get_nli_model(model_name)
    scores = model.predict([[premise, hypothesis]])
    if hasattr(scores[0], "__len__") and len(scores[0]) >= 3:
        return _softmax_entailment(scores[0])
    return float(scores[0]) if not hasattr(scores[0], "__len__") else float(scores[0][0])


def batch_compute_entailment(
    pairs: list[tuple[str, str]],
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    batch_size: int = 32,
) -> list[float]:
    """Batch compute entailment scores for multiple premise-hypothesis pairs.

    Args:
        pairs: List of (premise, hypothesis) tuples.
        model_name: Cross-encoder model to use (or alias like "lightweight").
        batch_size: Number of pairs to process at once.

    Returns:
        List of entailment probabilities.
    """
    if not pairs:
        return []

    model = get_nli_model(model_name)
    results: list[float] = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        scores = model.predict(batch)

        for score_row in scores:
            if hasattr(score_row, "__len__") and len(score_row) >= 3:
                results.append(_softmax_entailment(score_row))
            else:
                results.append(float(score_row))

    return results


async def batch_compute_entailment_async(
    pairs: list[tuple[str, str]],
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    batch_size: int = 32,
) -> list[float]:
    """Async wrapper around batch NLI inference.

    Offloads the CPU-bound model inference to a thread pool so it
    doesn't block the event loop.

    Args:
        pairs: List of (premise, hypothesis) tuples.
        model_name: Cross-encoder model to use (or alias like "lightweight").
        batch_size: Number of pairs to process at once.

    Returns:
        List of entailment probabilities.
    """
    return await asyncio.to_thread(batch_compute_entailment, pairs, model_name, batch_size)
