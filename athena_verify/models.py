"""Data models for the athena-verify verification pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrieved context chunk.

    Can be constructed from a plain string or from a dict with
    content and optional metadata.
    """

    content: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_input(cls, raw: str | Chunk | dict[str, Any]) -> Chunk:
        """Coerce a user-supplied value into a Chunk.

        Accepts:
            - str → Chunk(content=s)
            - dict with "content" key → Chunk(**d)
            - Chunk → passthrough
        """
        if isinstance(raw, Chunk):
            return raw
        if isinstance(raw, str):
            return cls(content=raw)
        if isinstance(raw, dict):
            return cls(**{k: v for k, v in raw.items() if k in ("content", "source", "metadata")})
        raise TypeError(f"Cannot coerce {type(raw)} into Chunk")


class SentenceScore(BaseModel):
    """Per-sentence verification result."""

    text: str
    index: int
    nli_score: float = Field(ge=0.0, le=1.0, description="NLI entailment probability")
    lexical_overlap: float = Field(
        ge=0.0,
        le=1.0,
        description="Token-level F1 overlap with best-matching context",
    )
    llm_judge_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM-as-judge score (optional)",
    )
    trust_score: float = Field(ge=0.0, le=1.0, description="Calibrated trust score")
    support_status: str = Field(description="SUPPORTED | PARTIAL | UNSUPPORTED | CONTRADICTED")
    best_matching_context: str | None = Field(
        default=None,
        description="The context chunk with highest overlap",
    )
    supporting_spans: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Spans in context that support this sentence",
    )


class StreamingResult(BaseModel):
    """Incremental verification result yielded by verify_stream().

    Each yield contains the sentences verified so far and an updated
    trust score. The final yield has ``is_final=True`` and a complete
    ``VerificationResult``-equivalent snapshot.
    """

    trust_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Running trust score (updated with each sentence)",
    )
    sentences: list[SentenceScore] = Field(
        default_factory=list,
        description="Sentences verified so far (grows incrementally)",
    )
    is_final: bool = Field(
        default=False,
        description="True on the last yield when the stream is fully processed",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (populated on final yield)",
    )


class VerificationResult(BaseModel):
    """Complete verification result returned by verify()."""

    question: str
    answer: str
    trust_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall calibrated trust score (0.0-1.0)",
    )
    sentences: list[SentenceScore] = Field(description="Per-sentence verification scores")
    unsupported: list[SentenceScore] = Field(
        default_factory=list,
        description="Sentences that failed verification",
    )
    supported: list[SentenceScore] = Field(
        default_factory=list,
        description="Sentences that passed verification",
    )
    suggested_revision: str | None = Field(
        default=None,
        description="LLM-generated corrected answer (if enabled)",
    )
    verification_passed: bool = Field(description="True if overall trust >= threshold")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model used, latency, etc.)",
    )
