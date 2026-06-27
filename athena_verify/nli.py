"""Natural Language Inference (NLI) model for entailment checking.

Uses a cross-encoder model to compute entailment probability between
a premise (context) and a hypothesis (claim/sentence).
"""

from __future__ import annotations

import asyncio
import math
from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger()

NLI_MODEL_ALIASES: dict[str, str] = {
    "default": "cross-encoder/nli-deberta-v3-base",
    "lightweight": "cross-encoder/nli-MiniLM2-L6-H768",
    "vectara": "vectara/hallucination_evaluation_model",
    "deberta-base": "MoritzLaworr/NLI-deberta-base",
}


def resolve_nli_model(model_name: str) -> str:
    """Resolve a model alias to a full HuggingFace model identifier.

    Args:
        model_name: A model name or alias (e.g. "lightweight").

    Returns:
        The resolved HuggingFace model identifier.
    """
    return NLI_MODEL_ALIASES.get(model_name, model_name)


@lru_cache(maxsize=32)
def get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-base") -> Any:
    """Load the NLI cross-encoder model (lazy, cached).

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.

    Returns:
        CrossEncoder model instance.
    """
    resolved = resolve_nli_model(model_name)

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for NLI scoring. "
            "Install with: pip install athena-verify[nli]"
        ) from e

    logger.info("loading_nli_model", model=resolved, alias=model_name)
    return CrossEncoder(resolved)


@lru_cache(maxsize=32)
def entailment_index(model_name: str) -> int | None:
    """Resolve the entailment class index from the model's label map.

    Different NLI checkpoints order their classes differently — e.g. the
    cross-encoder/nli-* family uses ``0=contradiction, 1=entailment,
    2=neutral`` while many MoritzLaurer/DeBERTa checkpoints use
    ``0=entailment``. Hardcoding the index silently scores the wrong class
    on non-default models, which reads as a flood of false positives.

    Returns the index of the class whose label contains "entail", or
    ``None`` for single-logit consistency models (e.g. Vectara HHEM) that
    have no label map.
    """
    model = get_nli_model(model_name)
    config = getattr(getattr(model, "model", None), "config", None) or getattr(
        model, "config", None
    )
    id2label = getattr(config, "id2label", None)
    if not isinstance(id2label, dict):
        return None
    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)
    return None


def _softmax_entailment(logits: Any, entail_idx: int) -> float:
    """Convert NLI logits to entailment probability via softmax.

    Args:
        logits: Per-class logits for one premise/hypothesis pair.
        entail_idx: Index of the entailment class for this model.
    """
    row = list(logits)
    max_val = max(row)
    exp_vals = [math.exp(v - max_val) for v in row]
    total = sum(exp_vals)
    idx = entail_idx if 0 <= entail_idx < len(row) else 1
    return exp_vals[idx] / total


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
    entail_idx = entailment_index(model_name)
    scores = model.predict([[premise, hypothesis]])
    row = scores[0]
    if hasattr(row, "__len__") and len(row) >= 3:
        return _softmax_entailment(row, entail_idx if entail_idx is not None else 1)
    # Single-logit consistency model (e.g. HHEM): score is already a probability.
    return float(row) if not hasattr(row, "__len__") else float(row[0])


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
    entail_idx = entailment_index(model_name)
    fallback_idx = entail_idx if entail_idx is not None else 1
    results: list[float] = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        scores = model.predict(batch)

        for score_row in scores:
            if hasattr(score_row, "__len__") and len(score_row) >= 3:
                results.append(_softmax_entailment(score_row, fallback_idx))
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
