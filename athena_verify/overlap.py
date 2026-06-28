"""Lexical overlap computation for verification pipeline.

Computes token-level F1 overlap between a sentence and context chunks
to measure how well the sentence is grounded in the retrieved context.

This module fixes the bug in the original verifier.py which computed
overlap against chunk IDs instead of chunk content.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
# A number: digits with optional thousands separators / decimal part.
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Function words carry no grounding signal; excluding them keeps containment
# from being inflated by shared "the/of/is" tokens.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "for", "and", "or", "but", "with", "at", "by",
        "as", "that", "this", "these", "those", "it", "its", "from", "into",
        "than", "then", "also", "such", "which", "their", "they", "them",
        "has", "have", "had", "will", "shall", "may", "can", "any", "all",
        "not", "no", "only", "other", "more", "most", "some", "each", "both",
    }
)


def _normalize_number(token: str) -> str:
    """Strip thousands separators so '1,200' and '1200' compare equal."""
    return token.replace(",", "")


def containment_score(sentence: str, context_text: str) -> float:
    """Fraction of a sentence's content words that appear in the context.

    Unlike symmetric token F1 (which is penalised by long context), this is a
    precision-style measure of how much of the *claim* is lexically grounded.
    It is the signal used to rescue faithful paraphrases that standalone NLI
    scores as neutral.
    """
    ctx_tokens = set(_WORD_RE.findall(context_text.lower()))
    words = [
        w for w in _WORD_RE.findall(sentence.lower()) if len(w) > 2 and w not in _STOPWORDS
    ]
    if not words:
        return 0.0
    return sum(1 for w in words if w in ctx_tokens) / len(words)


def numeric_consistency(sentence: str, context_text: str) -> bool:
    """True if every number in the sentence also appears in the context.

    Comma-insensitive. Returns True when the sentence contains no numbers.
    This is the guard that keeps number-substitution hallucinations
    ("the cap is $5M" against a $2M context) from being rescued by lexical
    containment, since the swapped figure will be absent from the context.
    """
    ctx_nums = {_normalize_number(n) for n in _NUM_RE.findall(context_text)}
    sent_nums = [_normalize_number(n) for n in _NUM_RE.findall(sentence)]
    return all(n in ctx_nums for n in sent_nums)


def token_f1(text1: str, text2: str) -> float:
    """Compute token-level F1 overlap between two texts.

    Args:
        text1: First text (typically the sentence).
        text2: Second text (typically a context chunk).

    Returns:
        F1 score (0.0-1.0).
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)

    if intersection == 0:
        return 0.0

    precision = intersection / len(tokens1)
    recall = intersection / len(tokens2)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def best_overlap_score(
    sentence: str,
    context_chunks: list[str],
) -> tuple[float, str | None]:
    """Find the best-matching context chunk for a sentence.

    Computes token F1 between the sentence and each context chunk,
    returning the highest score and the matching chunk.

    Args:
        sentence: The sentence to check.
        context_chunks: List of context chunk strings.

    Returns:
        Tuple of (best F1 score, best matching chunk or None).
    """
    if not context_chunks:
        return 0.0, None

    best_score = 0.0
    best_chunk = None

    for chunk in context_chunks:
        score = token_f1(sentence, chunk)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_score, best_chunk


def batch_overlap_scores(
    sentences: list[str],
    context_chunks: list[str],
) -> list[tuple[float, str | None]]:
    """Compute best overlap score for each sentence against all context chunks.

    Args:
        sentences: List of sentences to check.
        context_chunks: List of context chunk strings.

    Returns:
        List of (best F1 score, best matching chunk) tuples.
    """
    return [best_overlap_score(s, context_chunks) for s in sentences]
