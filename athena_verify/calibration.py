"""Trust score calibration for the verification pipeline.

Combines multiple verification signals (NLI, lexical overlap, optional
LLM-judge) into a single calibrated trust score per sentence.
"""

from __future__ import annotations

from athena_verify.models import SentenceScore


# Default weights for the ensemble
DEFAULT_WEIGHTS = {
    "nli": 0.55,
    "overlap": 0.25,
    "llm_judge": 0.20,
}

# Thresholds for support status classification
SUPPORTED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.50
UNSUPPORTED_THRESHOLD = 0.30


def compute_trust_score(
    nli_score: float,
    lexical_overlap: float,
    llm_judge_score: float | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a calibrated trust score from individual signals.

    When LLM-judge score is not available, its weight is redistributed
    proportionally to NLI and overlap.

    Args:
        nli_score: NLI entailment probability (0.0-1.0).
        lexical_overlap: Token F1 overlap (0.0-1.0).
        llm_judge_score: Optional LLM-as-judge score (0.0-1.0).
        weights: Optional custom weights dict.

    Returns:
        Calibrated trust score (0.0-1.0).
    """
    w = weights or DEFAULT_WEIGHTS

    if llm_judge_score is not None:
        # All three signals available
        total_weight = w["nli"] + w["overlap"] + w["llm_judge"]
        trust = (
            w["nli"] * nli_score
            + w["overlap"] * lexical_overlap
            + w["llm_judge"] * llm_judge_score
        ) / total_weight
    else:
        # Only NLI + overlap — redistribute LLM-judge weight
        total_weight = w["nli"] + w["overlap"]
        trust = (
            w["nli"] * nli_score
            + w["overlap"] * lexical_overlap
        ) / total_weight

    return min(1.0, max(0.0, trust))


def classify_support(trust_score: float) -> str:
    """Classify a sentence's support status based on trust score.

    Args:
        trust_score: Calibrated trust score (0.0-1.0).

    Returns:
        One of: SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED.
    """
    if trust_score >= SUPPORTED_THRESHOLD:
        return "SUPPORTED"
    elif trust_score >= PARTIAL_THRESHOLD:
        return "PARTIAL"
    elif trust_score >= UNSUPPORTED_THRESHOLD:
        return "UNSUPPORTED"
    else:
        return "CONTRADICTED"


def compute_overall_trust(
    sentences: list[SentenceScore],
    trust_threshold: float = 0.70,
) -> tuple[float, bool]:
    """Compute overall trust score and whether verification passed.

    Args:
        sentences: List of sentence-level scores.
        trust_threshold: Minimum mean trust for verification to pass.

    Returns:
        Tuple of (overall trust score, verification passed).
    """
    if not sentences:
        return 0.0, False

    overall_trust = sum(s.trust_score for s in sentences) / len(sentences)
    unsupported_ratio = sum(
        1 for s in sentences if s.support_status in ("UNSUPPORTED", "CONTRADICTED")
    ) / len(sentences)

    # Verification passes if mean trust is above threshold AND
    # fewer than 30% of sentences are unsupported/contradicted
    passed = overall_trust >= trust_threshold and unsupported_ratio < 0.3

    return overall_trust, passed
