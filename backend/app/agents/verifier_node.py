"""Verifier node: check answer credibility and decide if retry is needed."""

import structlog

from app.agents.state import ResearchState
from app.verification import verify_answer

logger = structlog.get_logger()

VERIFICATION_THRESHOLD = 0.7
UNSUPPORTED_THRESHOLD = 0.3


async def verifier_node(state: ResearchState) -> dict:
    """Verify the writer's answer and decide if retry is needed.

    Verification flow:
    1. Parse answer and recover citations
    2. Run NLI entailment checks
    3. Compute trust score
    4. If trust < threshold OR >30% unsupported AND iteration < max:
       - Extract weak claims
       - Signal retry to researcher
    5. Otherwise, pass answer to output
    """
    final_answer = state.get("final_answer", "")
    chunks = state.get("retrieved_chunks", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1)

    logger.info(
        "verifier checking answer",
        iteration=iteration,
        chunks=len(chunks),
    )

    if not final_answer or not chunks:
        logger.warning("verifier: missing answer or chunks")
        return {
            "verified_sentences": [],
            "trust_score": 0.0,
            "verification_passed": False,
            "weak_claims": [],
        }

    # Run verification
    verified_answer = await verify_answer(final_answer, chunks)
    if verified_answer is None:
        logger.warning("verifier: answer verification failed")
        return {
            "verified_sentences": [],
            "trust_score": 0.0,
            "verification_passed": False,
            "weak_claims": [],
        }

    # Extract weak claims
    weak_claims = [
        s.text
        for s in verified_answer.sentences
        if s.support_status in ("UNSUPPORTED", "CONTRADICTED")
    ]
    unsupported_ratio = len(weak_claims) / len(
        verified_answer.sentences
    ) if verified_answer.sentences else 0

    # Determine if we should retry
    should_retry = (
        verified_answer.overall_trust_score < VERIFICATION_THRESHOLD
        or unsupported_ratio > UNSUPPORTED_THRESHOLD
    ) and iteration < max_iterations

    logger.info(
        "verifier done",
        trust_score=verified_answer.overall_trust_score,
        support_status=verified_answer.overall_support_status,
        weak_claims=len(weak_claims),
        should_retry=should_retry,
    )

    verified_sentences = [s.model_dump() for s in verified_answer.sentences]

    return {
        "verified_sentences": verified_sentences,
        "trust_score": verified_answer.overall_trust_score,
        "verification_passed": verified_answer.verification_passed,
        "weak_claims": weak_claims,
    }
