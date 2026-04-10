"""Pydantic models for citation verification pipeline."""

from pydantic import BaseModel


class CitationSpan(BaseModel):
    """Reference to a span within a chunk."""
    chunk_id: str
    start: int
    end: int


class VerifiedSentence(BaseModel):
    """A sentence with its citations and verification status."""
    text: str
    citations: list[CitationSpan]
    support_status: str  # "SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"
    confidence: float  # 0.0-1.0
    trust_score: float  # 0.0-1.0
    nli_score: float  # 0.0-1.0 entailment probability
    lexical_overlap: float  # 0.0-1.0


class VerifiedAnswer(BaseModel):
    """Complete answer with per-sentence verification."""
    sentences: list[VerifiedSentence]
    overall_trust_score: float  # Mean of sentence trust_scores
    overall_support_status: str  # "SUPPORTED", "PARTIAL", "UNSUPPORTED"
    verification_passed: bool  # True if mean trust >= threshold


class VerifiedAnswerDraft(BaseModel):
    """Writer output: structured answer with citations."""
    sentences: list[dict]  # Each: {"text": str, "citations": [{"chunk_id": str, "start": int, "end": int}]}
