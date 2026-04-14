"""Sentence splitting for verification pipeline.

Splits answer text into individual sentences for per-sentence verification.
Uses regex-based splitting with optional NLTK support for better accuracy.
"""

from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Uses a regex-based approach that handles common English sentence
    boundaries. For production use with non-English text, consider
    installing NLTK and using nltk.sent_tokenize().

    Args:
        text: The answer text to split.

    Returns:
        List of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = text.strip()

    # Split on sentence-ending punctuation followed by space or end-of-string.
    # Handles: period, exclamation, question mark.
    # Handles abbreviations poorly (e.g., "Dr. Smith" splits incorrectly).
    # This is acceptable for v1; NLTK integration is a future improvement.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Filter empty strings and strip whitespace
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            result.append(s)

    return result


def split_sentences_nltk(text: str) -> list[str]:
    """Split text into sentences using NLTK's Punkt tokenizer.

    Falls back to regex-based splitting if NLTK is not installed.

    Args:
        text: The answer text to split.

    Returns:
        List of non-empty sentence strings.
    """
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except (ImportError, LookupError):
        return split_sentences(text)
