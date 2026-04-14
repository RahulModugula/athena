"""athena-verify: Runtime verification layer for RAG hallucination detection.

Drop-in guardrail that catches RAG hallucinations sentence-by-sentence
before they reach users.

Usage:
    from athena_verify import verify

    result = verify(
        question="What is the indemnification cap?",
        answer="The cap is $1M per incident.",
        context=retrieved_chunks,
    )

    print(result.trust_score)      # 0.0 - 1.0
    print(result.unsupported)      # sentences that failed verification
"""

from athena_verify.core import verify, verify_async, verified_completion
from athena_verify.models import Chunk, SentenceScore, VerificationResult

__all__ = [
    "verify",
    "verify_async",
    "verified_completion",
    "VerificationResult",
    "SentenceScore",
    "Chunk",
]

__version__ = "0.1.0"
