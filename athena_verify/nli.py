"""Natural Language Inference (NLI) model for entailment checking.

Uses a cross-encoder model to compute entailment probability between
a premise (context) and a hypothesis (claim/sentence).
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger()

# Global NLI model cache — loaded once, reused across calls
_nli_model: Any = None
_nli_model_name: str = ""


def get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-base") -> Any:
    """Load the NLI cross-encoder model (lazy, cached).

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.

    Returns:
        CrossEncoder model instance.
    """
    global _nli_model, _nli_model_name

    if _nli_model is not None and _nli_model_name == model_name:
        return _nli_model

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for NLI scoring. "
            "Install with: pip install athena-verify[nli]"
        ) from e

    logger.info("loading_nli_model", model=model_name)
    _nli_model = CrossEncoder(model_name)
    _nli_model_name = model_name
    return _nli_model


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
    # CrossEncoder NLI models return [entail, neutral, contradict] logits
    # scores[0] is the array of logits for the first (only) pair
    if hasattr(scores, "shape") and len(scores.shape) > 1:
        # Softmax to get probabilities, take entailment (index 0)
        import numpy as np
        probs = np.exp(scores[0]) / np.exp(scores[0]).sum()
        return float(probs[0])
    # Some models return a single score
    return float(scores[0]) if not hasattr(scores[0], "__len__") else float(scores[0][0])


def batch_compute_entailment(
    pairs: list[tuple[str, str]],
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> list[float]:
    """Batch compute entailment scores for multiple premise-hypothesis pairs.

    Args:
        pairs: List of (premise, hypothesis) tuples.
        model_name: Cross-encoder model to use.

    Returns:
        List of entailment probabilities.
    """
    if not pairs:
        return []

    model = get_nli_model(model_name)
    scores = model.predict(pairs)

    import numpy as np

    results = []
    for score_row in scores:
        if hasattr(score_row, "__len__") and len(score_row) >= 3:
            # [entail, neutral, contradict] logits → softmax → entailment prob
            probs = np.exp(score_row) / np.exp(score_row).sum()
            results.append(float(probs[0]))
        else:
            # Single score (some models)
            results.append(float(score_row))

    return results


async def batch_compute_entailment_async(
    pairs: list[tuple[str, str]],
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> list[float]:
    """Async wrapper around batch NLI inference.

    Offloads the CPU-bound model inference to a thread pool so it
    doesn't block the event loop.

    Args:
        pairs: List of (premise, hypothesis) tuples.
        model_name: Cross-encoder model to use.

    Returns:
        List of entailment probabilities.
    """
    return await asyncio.to_thread(batch_compute_entailment, pairs, model_name)
