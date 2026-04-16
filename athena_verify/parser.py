"""Sentence splitting for verification pipeline.

Splits answer text into individual sentences for per-sentence verification.
Uses regex-based splitting with optional NLTK support for better accuracy.
Also provides an async sentence buffer for streaming token input.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


async def sentence_buffer(
    token_stream: AsyncIterator[str],
    timeout: float = 2.0,
) -> AsyncIterator[str]:
    """Buffer streaming tokens and yield complete sentences.

    Accumulates tokens from *token_stream* until a sentence-ending
    boundary (``.``, ``!``, ``?`` followed by whitespace) is found,
    then yields the complete sentence.

    When the token stream is exhausted, any remaining buffered text is
    yielded as a final sentence (even if it doesn't end with
    punctuation).

    Args:
        token_stream: Async iterator yielding individual tokens/chunks.
        timeout: Not used in v1; reserved for future partial-yield logic.

    Yields:
        Complete sentence strings.
    """
    buf = ""
    async for token in token_stream:
        buf += token
        while True:
            m = SENTENCE_BOUNDARY.search(buf)
            if m is None:
                break
            sentence = buf[: m.end()].strip()
            buf = buf[m.end() :]
            if sentence:
                yield sentence

    if buf.strip():
        yield buf.strip()


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
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

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
