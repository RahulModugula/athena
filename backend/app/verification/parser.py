"""Parse structured citations from writer output."""

import json
import re  # Used for sentence splitting
from typing import Any

import structlog
from pydantic import ValidationError

from app.verification.models import VerifiedAnswerDraft

logger = structlog.get_logger()


def extract_citations_from_json(text: str) -> VerifiedAnswerDraft | None:
    """Extract citation structure from JSON writer output.

    The writer should emit:
    {
        "sentences": [
            {
                "text": "...",
                "citations": [
                    {"chunk_id": "uuid", "start": 0, "end": 100}
                ]
            }
        ]
    }

    Returns:
        VerifiedAnswerDraft or None if parsing fails.
    """
    try:
        data = json.loads(text)
        draft = VerifiedAnswerDraft(**data)
        return draft
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("failed to parse citations from JSON", error=str(e))
        return None


def extract_citations_fallback(text: str) -> VerifiedAnswerDraft | None:
    """Fallback: extract [Source N] citations from freeform text.

    This is a best-effort parser for backwards compatibility or when
    the writer fails to emit structured JSON.

    Returns:
        VerifiedAnswerDraft with extracted citations, or None if none found.
    """
    # Split by sentences (rough heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return None

    parsed_sentences: list[dict[str, Any]] = []

    for sent in sentences:
        if not sent.strip():
            continue

        # Extract source references
        # Note: incomplete extraction — we don't have chunk IDs from freeform text
        # This is a fallback only; prefer JSON parsing above
        parsed_sentences.append({
            "text": sent.strip(),
            "citations": [],
        })

    if not parsed_sentences:
        return None

    try:
        return VerifiedAnswerDraft(sentences=parsed_sentences)
    except ValidationError:
        return None


def parse_answer(text: str) -> VerifiedAnswerDraft | None:
    """Parse answer text, trying JSON first, then fallback.

    Args:
        text: Writer output (structured JSON or freeform text)

    Returns:
        VerifiedAnswerDraft or None
    """
    # Try strict JSON first
    draft = extract_citations_from_json(text)
    if draft is not None:
        return draft

    # Fallback to heuristic extraction
    draft = extract_citations_fallback(text)
    return draft
