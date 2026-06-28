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
    """Split text into sentences using NLTK's Punkt tokenizer.

    Falls back to regex-based splitting if NLTK is not installed.
    Handles abbreviations like "Dr. Smith" and "U.S." correctly.

    Args:
        text: The answer text to split.

    Returns:
        List of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except (ImportError, LookupError):
        return _split_sentences_regex(text)


# Common abbreviations that end in a period but do not end a sentence. Kept
# lowercase and without the trailing period for matching. Covers titles, legal
# and academic citation forms, and Latin/measurement shorthands — the domains
# (legal, medical, technical) athena targets, where a wrong split fragments a
# claim and shows up as a false positive.
_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "rev", "hon", "sr", "jr", "st",
        "vs", "etc", "al", "cf", "eg", "ie", "ca", "approx",
        "inc", "ltd", "co", "corp", "llc", "plc",
        "no", "nos", "fig", "figs", "sec", "secs", "art", "para", "pp", "vol",
        "ch", "ed", "eds", "rep", "dept", "est", "min", "max",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
        # multi-dot forms, matched after stripping internal periods
        "us", "uk", "un", "eu", "am", "pm", "phd", "md", "ba", "ma", "bs",
    }
)


def _split_sentences_regex(text: str) -> list[str]:
    """Split text into sentences using regex (fallback for when NLTK is absent).

    Abbreviation-aware: a candidate boundary is rejected when the token before
    the period is a known abbreviation (``Dr.``, ``Inc.``), a single-letter
    initial, or a dotted acronym (``U.S.``), so claims in legal/medical text
    aren't fragmented.

    Args:
        text: The answer text to split.

    Returns:
        List of non-empty sentence strings.
    """
    text = text.strip()
    if not text:
        return []

    result: list[str] = []
    start = 0
    # A boundary is sentence-ending punctuation, an optional closing quote/paren,
    # then whitespace, followed by something that looks like a new sentence.
    for m in re.finditer(r"[.!?]+[\"')\]]?\s+(?=[A-Z0-9\"'(])", text):
        preceding = text[start : m.start()]
        last_token = preceding.split()[-1] if preceding.split() else ""
        # Normalize: drop internal/trailing periods so "U.S" -> "us", "Dr" -> "dr".
        normalized = last_token.replace(".", "").strip(",;:\"'()").lower()
        if normalized in _ABBREVIATIONS or len(normalized) == 1:
            continue
        sentence = text[start : m.end()].strip()
        if sentence:
            result.append(sentence)
        start = m.end()

    tail = text[start:].strip()
    if tail:
        result.append(tail)

    return result
