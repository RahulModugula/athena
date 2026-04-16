"""Data models for the athena-verify verification pipeline."""

from __future__ import annotations

import time
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
    suggested_fix: str | None = Field(
        default=None,
        description="LLM-generated corrected sentence (if suggest_revisions enabled)",
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

    def to_json(self) -> str:
        """Serialize to a JSON string suitable for logging pipelines."""
        return self.model_dump_json()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return self.model_dump(mode="json")


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

    def to_json(self) -> str:
        """Serialize to a JSON string suitable for logging pipelines."""
        return self.model_dump_json()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return self.model_dump(mode="json")

    def to_otel_span(self) -> dict[str, Any]:
        """Return an OpenTelemetry-compatible span representation.

        Produces a dict that can be fed directly into an OTel Span
        as attributes.

        Returns:
            Dict with OTel span attributes.
        """
        return {
            "name": "athena.verify",
            "attributes": {
                "athena.trust_score": self.trust_score,
                "athena.verification_passed": self.verification_passed,
                "athena.num_sentences": len(self.sentences),
                "athena.num_unsupported": len(self.unsupported),
                "athena.num_supported": len(self.supported),
                "athena.question": self.question,
                **{
                    f"athena.metadata.{k}": v
                    for k, v in self.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            },
            "events": [
                {
                    "name": "sentence_score",
                    "timestamp": int(time.time() * 1e9),
                    "attributes": {
                        "text": s.text,
                        "trust_score": s.trust_score,
                        "support_status": s.support_status,
                        "nli_score": s.nli_score,
                        "lexical_overlap": s.lexical_overlap,
                        **({"suggested_fix": s.suggested_fix} if s.suggested_fix else {}),
                    },
                }
                for s in self.sentences
            ],
        }

    def to_langfuse_trace(self) -> dict[str, Any]:
        """Return a Langfuse-compatible trace/span representation.

        Returns:
            Dict that can be posted to the Langfuse API as a trace.
        """
        return {
            "name": "athena-verify",
            "metadata": self.metadata,
            "output": {
                "trust_score": self.trust_score,
                "verification_passed": self.verification_passed,
                "unsupported_count": len(self.unsupported),
                "supported_count": len(self.supported),
            },
            "scores": [
                {
                    "name": f"sentence_{s.index}",
                    "value": s.trust_score,
                    "comment": f"{s.support_status}: {s.text}"
                    + (f" → {s.suggested_fix}" if s.suggested_fix else ""),
                }
                for s in self.sentences
            ],
        }
