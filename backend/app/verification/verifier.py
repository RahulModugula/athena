"""Orchestrate verification of citations against retrieved chunks."""

import hashlib
import structlog
from typing import Any

from app.verification.models import VerifiedAnswer, VerifiedSentence, CitationSpan
from app.verification.nli import batch_compute_entailment
from app.verification.parser import parse_answer

logger = structlog.get_logger()


def _lexical_overlap(text1: str, text2: str) -> float:
    """Compute token-level F1 overlap between two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    if union == 0:
        return 0.0

    return intersection / union


async def verify_answer(
    answer_text: str,
    retrieved_chunks: list[dict[str, Any]],
) -> VerifiedAnswer | None:
    """Verify an answer's citations against retrieved chunks.

    Args:
        answer_text: Writer output (JSON-structured or freeform)
        retrieved_chunks: List of chunks with content and offsets

    Returns:
        VerifiedAnswer with per-sentence verification, or None if parsing fails
    """
    # Parse the answer
    draft = parse_answer(answer_text)
    if draft is None:
        logger.warning("failed to parse answer for verification")
        return None

    # Build a map of chunk_id -> chunk for quick lookup
    chunks_by_id = {str(c.get("chunk_id", "")): c for c in retrieved_chunks}

    # Verify each sentence
    verified_sentences = []
    nli_pairs = []  # For batch NLI inference
    sentence_indices = []

    for i, sent_dict in enumerate(draft.sentences):
        sent_text = sent_dict.get("text", "").strip()
        citations = sent_dict.get("citations", [])

        if not sent_text:
            continue

        # Recover cited text from chunks
        cited_texts = []
        for cit in citations:
            chunk_id = str(cit.get("chunk_id", ""))
            start = cit.get("start", 0)
            end = cit.get("end", 0)

            chunk = chunks_by_id.get(chunk_id)
            if chunk and start >= 0 and end <= len(chunk.get("content", "")):
                cited_text = chunk["content"][start:end]
                cited_texts.append(cited_text)

        # If we recovered any cited text, add to NLI batch
        if cited_texts:
            combined_cited = " ".join(cited_texts)
            nli_pairs.append((combined_cited, sent_text))
            sentence_indices.append(i)
        else:
            # No valid citations
            nli_pairs.append(("", sent_text))
            sentence_indices.append(i)

    # Batch NLI inference
    if nli_pairs:
        nli_scores = await batch_compute_entailment_async(nli_pairs)
    else:
        nli_scores = []

    # Map NLI scores back to sentences and compute trust
    nli_score_by_idx = {}
    for idx, score in zip(sentence_indices, nli_scores, strict=False):
        nli_score_by_idx[idx] = score

    # Build verified sentences
    for i, sent_dict in enumerate(draft.sentences):
        sent_text = sent_dict.get("text", "").strip()
        citations = sent_dict.get("citations", [])

        if not sent_text:
            continue

        nli_score = nli_score_by_idx.get(i, 0.0)
        lexical = _lexical_overlap(sent_text, " ".join(c.get("chunk_id", "") for c in citations))

        # Determine support status based on NLI
        if nli_score >= 0.8:
            support_status = "SUPPORTED"
        elif nli_score >= 0.5:
            support_status = "PARTIAL"
        elif nli_score >= 0.3:
            support_status = "UNSUPPORTED"
        else:
            support_status = "CONTRADICTED"

        # Simple trust score (average of NLI and lexical)
        trust_score = (nli_score + lexical) / 2

        verified_sent = VerifiedSentence(
            text=sent_text,
            citations=[CitationSpan(**c) for c in citations],
            support_status=support_status,
            confidence=nli_score,
            trust_score=trust_score,
            nli_score=nli_score,
            lexical_overlap=lexical,
        )
        verified_sentences.append(verified_sent)

    # Compute overall metrics
    if not verified_sentences:
        return None

    overall_trust = sum(s.trust_score for s in verified_sentences) / len(verified_sentences)
    unsupported_count = sum(1 for s in verified_sentences if s.support_status == "UNSUPPORTED")
    unsupported_ratio = unsupported_count / len(verified_sentences)

    if overall_trust >= 0.7 and unsupported_ratio < 0.3:
        verification_passed = True
        overall_status = "SUPPORTED"
    elif unsupported_ratio > 0.5:
        verification_passed = False
        overall_status = "UNSUPPORTED"
    else:
        verification_passed = True
        overall_status = "PARTIAL"

    return VerifiedAnswer(
        sentences=verified_sentences,
        overall_trust_score=overall_trust,
        overall_support_status=overall_status,
        verification_passed=verification_passed,
    )


async def batch_compute_entailment_async(
    pairs: list[tuple[str, str]],
) -> list[float]:
    """Async wrapper around batch NLI inference."""
    # For now, just call the sync version
    # In production, this could be offloaded to a thread pool
    return batch_compute_entailment(pairs)
