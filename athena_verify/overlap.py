"""Lexical overlap computation for verification pipeline.

Computes token-level F1 overlap between a sentence and context chunks
to measure how well the sentence is grounded in the retrieved context.

This module fixes the bug in the original verifier.py which computed
overlap against chunk IDs instead of chunk content.
"""

from __future__ import annotations


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
