"""Core verification function — the main entry point for athena-verify.

Provides verify() and verified_completion() for checking whether an LLM
answer is grounded in the provided context chunks.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from athena_verify.calibration import (
    classify_support,
    compute_overall_trust,
    compute_trust_score,
)
from athena_verify.llm_judge import LLMClient, batch_judge_sentences
from athena_verify.models import Chunk, SentenceScore, VerificationResult
from athena_verify.nli import batch_compute_entailment, batch_compute_entailment_async
from athena_verify.overlap import best_overlap_score
from athena_verify.parser import split_sentences

logger = structlog.get_logger()


def verify(
    question: str,
    answer: str,
    context: list[str] | list[Chunk] | list[dict[str, Any]],
    *,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    use_llm_judge: bool = False,
    llm_client: LLMClient | None = None,
    trust_threshold: float = 0.70,
    weights: dict[str, float] | None = None,
) -> VerificationResult:
    """Verify an LLM answer against retrieved context chunks.

    Splits the answer into sentences, scores each sentence for NLI
    entailment and lexical overlap against the context, optionally
    adds LLM-as-judge scoring, and returns per-sentence and overall
    trust scores.

    Args:
        question: The original question asked.
        answer: The LLM-generated answer to verify.
        context: Retrieved context chunks (list of strings, dicts, or Chunk objects).
        nli_model: Cross-encoder model name for NLI scoring.
        use_llm_judge: Whether to use LLM-as-judge for borderline cases.
        llm_client: LLM client instance (required if use_llm_judge is True).
        trust_threshold: Minimum trust score for verification to pass.
        weights: Custom weights for the trust score ensemble.

    Returns:
        VerificationResult with per-sentence scores and overall assessment.
    """
    start_time = time.time()

    # Normalize context to list of Chunk objects
    chunks = [Chunk.from_input(c) for c in context]
    chunk_texts = [c.content for c in chunks]

    # Split answer into sentences
    sentences = split_sentences(answer)

    if not sentences:
        return VerificationResult(
            question=question,
            answer=answer,
            trust_score=0.0,
            sentences=[],
            unsupported=[],
            supported=[],
            verification_passed=False,
            metadata={"error": "no_sentences_found", "latency_ms": 0},
        )

    # --- NLI scoring ---
    # Pair each sentence with the full context for entailment check
    nli_pairs = [
        (" ".join(chunk_texts), sentence)
        for sentence in sentences
    ]
    nli_scores = batch_compute_entailment(nli_pairs, model_name=nli_model)

    # --- Lexical overlap scoring ---
    overlap_results = [best_overlap_score(s, chunk_texts) for s in sentences]

    # --- Optional LLM-as-judge scoring ---
    llm_scores: list[float | None] = [None] * len(sentences)
    if use_llm_judge and llm_client is not None:
        combined_context = " ".join(chunk_texts)
        judge_results = batch_judge_sentences(sentences, combined_context, question, llm_client)
        llm_scores = [score for score, _ in judge_results]

    # --- Build per-sentence results ---
    sentence_scores: list[SentenceScore] = []
    for i, sentence in enumerate(sentences):
        nli = nli_scores[i] if i < len(nli_scores) else 0.0
        overlap, best_chunk = overlap_results[i]
        llm = llm_scores[i] if i < len(llm_scores) else None

        trust = compute_trust_score(nli, overlap, llm, weights)
        status = classify_support(trust)

        sentence_scores.append(
            SentenceScore(
                text=sentence,
                index=i,
                nli_score=nli,
                lexical_overlap=overlap,
                llm_judge_score=llm,
                trust_score=trust,
                support_status=status,
                best_matching_context=best_chunk,
            )
        )

    # --- Overall assessment ---
    overall_trust, passed = compute_overall_trust(sentence_scores, trust_threshold)

    supported = [
        s for s in sentence_scores if s.support_status in ("SUPPORTED", "PARTIAL")
    ]
    unsupported = [
        s for s in sentence_scores
        if s.support_status in ("UNSUPPORTED", "CONTRADICTED")
    ]

    latency_ms = (time.time() - start_time) * 1000

    return VerificationResult(
        question=question,
        answer=answer,
        trust_score=round(overall_trust, 4),
        sentences=sentence_scores,
        unsupported=unsupported,
        supported=supported,
        verification_passed=passed,
        metadata={
            "nli_model": nli_model,
            "num_chunks": len(chunks),
            "num_sentences": len(sentences),
            "latency_ms": round(latency_ms, 1),
            "llm_judge_used": use_llm_judge and llm_client is not None,
        },
    )


async def verify_async(
    question: str,
    answer: str,
    context: list[str] | list[Chunk] | list[dict[str, Any]],
    *,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    use_llm_judge: bool = False,
    llm_client: LLMClient | None = None,
    trust_threshold: float = 0.70,
    weights: dict[str, float] | None = None,
) -> VerificationResult:
    """Async version of verify().

    Offloads NLI model inference to a thread pool to avoid blocking
    the event loop.

    Args:
        Same as verify().

    Returns:
        Same as verify().
    """
    start_time = time.time()

    # Normalize context
    chunks = [Chunk.from_input(c) for c in context]
    chunk_texts = [c.content for c in chunks]

    # Split answer into sentences
    sentences = split_sentences(answer)

    if not sentences:
        return VerificationResult(
            question=question,
            answer=answer,
            trust_score=0.0,
            sentences=[],
            unsupported=[],
            supported=[],
            verification_passed=False,
            metadata={"error": "no_sentences_found", "latency_ms": 0},
        )

    # --- NLI scoring (async) ---
    nli_pairs = [
        (" ".join(chunk_texts), sentence)
        for sentence in sentences
    ]
    nli_scores = await batch_compute_entailment_async(nli_pairs, model_name=nli_model)

    # --- Lexical overlap scoring ---
    overlap_results = [best_overlap_score(s, chunk_texts) for s in sentences]

    # --- Optional LLM-as-judge scoring ---
    llm_scores: list[float | None] = [None] * len(sentences)
    if use_llm_judge and llm_client is not None:
        combined_context = " ".join(chunk_texts)
        judge_results = batch_judge_sentences(sentences, combined_context, question, llm_client)
        llm_scores = [score for score, _ in judge_results]

    # --- Build per-sentence results ---
    sentence_scores: list[SentenceScore] = []
    for i, sentence in enumerate(sentences):
        nli = nli_scores[i] if i < len(nli_scores) else 0.0
        overlap, best_chunk = overlap_results[i]
        llm = llm_scores[i] if i < len(llm_scores) else None

        trust = compute_trust_score(nli, overlap, llm, weights)
        status = classify_support(trust)

        sentence_scores.append(
            SentenceScore(
                text=sentence,
                index=i,
                nli_score=nli,
                lexical_overlap=overlap,
                llm_judge_score=llm,
                trust_score=trust,
                support_status=status,
                best_matching_context=best_chunk,
            )
        )

    # --- Overall assessment ---
    overall_trust, passed = compute_overall_trust(sentence_scores, trust_threshold)

    supported = [
        s for s in sentence_scores if s.support_status in ("SUPPORTED", "PARTIAL")
    ]
    unsupported = [
        s for s in sentence_scores
        if s.support_status in ("UNSUPPORTED", "CONTRADICTED")
    ]

    latency_ms = (time.time() - start_time) * 1000

    return VerificationResult(
        question=question,
        answer=answer,
        trust_score=round(overall_trust, 4),
        sentences=sentence_scores,
        unsupported=unsupported,
        supported=supported,
        verification_passed=passed,
        metadata={
            "nli_model": nli_model,
            "num_chunks": len(chunks),
            "num_sentences": len(sentences),
            "latency_ms": round(latency_ms, 1),
            "llm_judge_used": use_llm_judge and llm_client is not None,
        },
    )


def verified_completion(
    model: str,
    question: str,
    context: list[str] | list[Chunk] | list[dict[str, Any]],
    *,
    provider: str = "openai",
    api_key: str | None = None,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    trust_threshold: float = 0.70,
    **kwargs: Any,
) -> VerificationResult:
    """Generate an LLM completion and verify it against context.

    Convenience function that:
    1. Calls the LLM to generate an answer
    2. Verifies the answer against the context
    3. Returns the verification result

    Args:
        model: LLM model name (e.g., "gpt-4o", "claude-3-5-haiku-20241022").
        question: The question to ask.
        context: Retrieved context chunks.
        provider: "openai" or "anthropic".
        api_key: Optional API key.
        trust_threshold: Minimum trust score for verification to pass.
        **kwargs: Additional arguments passed to the LLM.

    Returns:
        VerificationResult with the generated answer verified.
    """
    # Generate answer
    chunks = [Chunk.from_input(c) for c in context]
    context_text = "\n\n".join(c.content for c in chunks)

    prompt = f"""Answer the following question based only on the provided context. \
If the context doesn't contain enough information, say so.

Context:
{context_text}

Question: {question}

Answer:"""

    if provider == "openai":
        from athena_verify.llm_judge import OpenAIJudge

        client = OpenAIJudge(model=model, api_key=api_key)
        answer = client.complete(prompt)
    elif provider == "anthropic":
        from athena_verify.llm_judge import AnthropicJudge

        client = AnthropicJudge(model=model, api_key=api_key)
        answer = client.complete(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    # Verify the answer
    return verify(
        question=question,
        answer=answer,
        context=context,
        nli_model=nli_model,
        trust_threshold=trust_threshold,
    )
