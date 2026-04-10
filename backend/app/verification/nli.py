"""Natural Language Inference (NLI) model for entailment checking."""

import structlog
from sentence_transformers import CrossEncoder

logger = structlog.get_logger()

# Global NLI model cache
_nli_model: CrossEncoder | None = None


def get_nli_model() -> CrossEncoder:
    """Load the NLI model (lazy load, cached)."""
    global _nli_model
    if _nli_model is None:
        logger.info("loading NLI model", model="cross-encoder/nli-deberta-v3-base")
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    return _nli_model


def compute_entailment_score(premise: str, hypothesis: str) -> float:
    """Compute entailment probability between premise and hypothesis.

    Args:
        premise: The supporting text (cited span)
        hypothesis: The claim we're verifying (sentence)

    Returns:
        Probability of entailment (0.0-1.0)
    """
    model = get_nli_model()
    # CrossEncoder expects (premise, hypothesis) pairs and returns 3 scores:
    # [entail, neutral, contradict]
    scores = model.predict([[premise, hypothesis]])[0]
    # scores[0] is entailment probability
    return float(scores[0])


def batch_compute_entailment(premise_hypothesis_pairs: list[tuple[str, str]]) -> list[float]:
    """Batch compute entailment scores for multiple pairs.

    Args:
        premise_hypothesis_pairs: List of (premise, hypothesis) tuples

    Returns:
        List of entailment probabilities
    """
    if not premise_hypothesis_pairs:
        return []

    model = get_nli_model()
    scores = model.predict(premise_hypothesis_pairs)
    return [float(s[0]) for s in scores]
