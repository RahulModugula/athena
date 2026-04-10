"""Citation verification module for hallucination detection."""

from app.verification.models import (
    VerifiedAnswer,
    VerifiedAnswerDraft,
    VerifiedSentence,
)
from app.verification.verifier import verify_answer

__all__ = ["verify_answer", "VerifiedAnswer", "VerifiedSentence", "VerifiedAnswerDraft"]
