"""Core verification function — the main entry point for athena-verify.

Provides verify(), verify_async(), verify_stream(), and
verified_completion() for checking whether an LLM answer is grounded
in the provided context chunks.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from athena_verify.calibration import (
    classify_support,
    compute_overall_trust,
    compute_trust_score,
)
from athena_verify.llm_judge import LLMClient, batch_generate_revisions, batch_judge_sentences
from athena_verify.models import (
    Chunk,
    SentenceScore,
    StreamingResult,
    VerificationResult,
)
from athena_verify.nli import batch_compute_entailment, batch_compute_entailment_async
from athena_verify.overlap import best_overlap_score
from athena_verify.parser import sentence_buffer, split_sentences

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
    suggest_revisions: bool = False,
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
        nli_model: Cross-encoder model name for NLI scoring (or alias like "lightweight").
        use_llm_judge: Whether to use LLM-as-judge for borderline cases.
        llm_client: LLM client instance (required if use_llm_judge or suggest_revisions is True).
        trust_threshold: Minimum trust score for verification to pass.
        weights: Custom weights for the trust score ensemble.
        suggest_revisions: Whether to generate LLM-powered corrections for unsupported sentences.

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
    nli_pairs = [(" ".join(chunk_texts), sentence) for sentence in sentences]
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

    supported = [s for s in sentence_scores if s.support_status in ("SUPPORTED", "PARTIAL")]
    unsupported = [
        s for s in sentence_scores if s.support_status in ("UNSUPPORTED", "CONTRADICTED")
    ]

    # --- Optional revision suggestions ---
    if suggest_revisions and llm_client is not None and unsupported:
        combined_context = " ".join(chunk_texts)
        revisions = batch_generate_revisions(
            [s.text for s in unsupported],
            combined_context,
            question,
            llm_client,
        )
        for sent, revision in zip(unsupported, revisions, strict=True):
            sent.suggested_fix = revision

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
            "revisions_suggested": suggest_revisions and llm_client is not None,
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
    suggest_revisions: bool = False,
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
    nli_pairs = [(" ".join(chunk_texts), sentence) for sentence in sentences]
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

    supported = [s for s in sentence_scores if s.support_status in ("SUPPORTED", "PARTIAL")]
    unsupported = [
        s for s in sentence_scores if s.support_status in ("UNSUPPORTED", "CONTRADICTED")
    ]

    if suggest_revisions and llm_client is not None and unsupported:
        combined_context = " ".join(chunk_texts)
        revisions = batch_generate_revisions(
            [s.text for s in unsupported],
            combined_context,
            question,
            llm_client,
        )
        for sent, revision in zip(unsupported, revisions, strict=True):
            sent.suggested_fix = revision

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
            "revisions_suggested": suggest_revisions and llm_client is not None,
        },
    )


def verify_batch(
    questions: list[str] | str,
    answers: list[str] | str,
    contexts: list[str]
    | list[Chunk]
    | list[dict[str, Any]]
    | list[list[str] | list[Chunk] | list[dict[str, Any]]],
    *,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    use_llm_judge: bool = False,
    llm_client: LLMClient | None = None,
    trust_threshold: float = 0.70,
    weights: dict[str, float] | None = None,
    suggest_revisions: bool = False,
    batch_size: int = 32,
) -> list[VerificationResult]:
    """Verify multiple question-answer pairs in batch.

    Supports two modes:
    1. Multiple Q&A pairs: parallel lists of questions, answers, and contexts.
    2. Single Q&A with many context chunks: batch processes NLI inference.

    Args:
        questions: Single question string or list of questions.
        answers: Single answer string or list of answers.
        contexts: Shared context or list of per-question contexts.
        nli_model: Cross-encoder model (or alias like "lightweight").
        use_llm_judge: Whether to use LLM-as-judge.
        llm_client: LLM client for judge / revisions.
        trust_threshold: Minimum trust score for pass.
        weights: Custom weights for trust score ensemble.
        suggest_revisions: Generate corrections for unsupported sentences.
        batch_size: NLI batch size for model inference.

    Returns:
        List of VerificationResult objects.
    """
    single_question = isinstance(questions, str)
    if single_question:
        questions_list: list[str] = [questions]
        answers_list: list[str] = [answers]
        contexts_list: list[Any] = [contexts]
    else:
        questions_list = questions
        answers_list = answers
        if isinstance(contexts, list) and len(contexts_list := contexts) > 0:
            if isinstance(contexts_list[0], (str, dict, Chunk)):
                contexts_list = [contexts] * len(questions_list)
        else:
            contexts_list = [contexts] * len(questions_list)

    all_results: list[VerificationResult] = []

    start_time = time.time()

    all_chunks: list[list[Chunk]] = []
    all_sentences: list[list[str]] = []
    all_nli_pairs: list[tuple[str, str]] = []
    pair_offsets: list[int] = []

    for q_idx in range(len(questions_list)):
        chunks = [Chunk.from_input(c) for c in contexts_list[q_idx]]
        chunk_texts = [c.content for c in chunks]
        combined_context = " ".join(chunk_texts)
        sentences = split_sentences(answers_list[q_idx])

        all_chunks.append(chunks)
        all_sentences.append(sentences)

        offset = len(all_nli_pairs)
        pair_offsets.append(offset)

        for sentence in sentences:
            all_nli_pairs.append((combined_context, sentence))

    nli_scores_all = batch_compute_entailment(
        all_nli_pairs, model_name=nli_model, batch_size=batch_size
    )

    for q_idx in range(len(questions_list)):
        chunks = all_chunks[q_idx]
        chunk_texts = [c.content for c in chunks]
        sentences = all_sentences[q_idx]

        if not sentences:
            all_results.append(
                VerificationResult(
                    question=questions_list[q_idx],
                    answer=answers_list[q_idx],
                    trust_score=0.0,
                    sentences=[],
                    unsupported=[],
                    supported=[],
                    verification_passed=False,
                    metadata={"error": "no_sentences_found", "latency_ms": 0},
                )
            )
            continue

        offset = pair_offsets[q_idx]
        sentence_scores: list[SentenceScore] = []
        llm_scores: list[float | None] = [None] * len(sentences)

        if use_llm_judge and llm_client is not None:
            combined_context = " ".join(chunk_texts)
            judge_results = batch_judge_sentences(
                sentences, combined_context, questions_list[q_idx], llm_client
            )
            llm_scores = [score for score, _ in judge_results]

        for i, sentence in enumerate(sentences):
            nli = nli_scores_all[offset + i] if (offset + i) < len(nli_scores_all) else 0.0
            overlap, best_chunk = best_overlap_score(sentence, chunk_texts)
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

        overall_trust, passed = compute_overall_trust(sentence_scores, trust_threshold)

        supported = [s for s in sentence_scores if s.support_status in ("SUPPORTED", "PARTIAL")]
        unsupported = [
            s for s in sentence_scores if s.support_status in ("UNSUPPORTED", "CONTRADICTED")
        ]

        if suggest_revisions and llm_client is not None and unsupported:
            combined_context = " ".join(chunk_texts)
            revisions = batch_generate_revisions(
                [s.text for s in unsupported],
                combined_context,
                questions_list[q_idx],
                llm_client,
            )
            for sent, revision in zip(unsupported, revisions, strict=True):
                sent.suggested_fix = revision

        all_results.append(
            VerificationResult(
                question=questions_list[q_idx],
                answer=answers_list[q_idx],
                trust_score=round(overall_trust, 4),
                sentences=sentence_scores,
                unsupported=unsupported,
                supported=supported,
                verification_passed=passed,
                metadata={
                    "nli_model": nli_model,
                    "num_chunks": len(chunks),
                    "num_sentences": len(sentences),
                    "latency_ms": round((time.time() - start_time) * 1000, 1),
                    "llm_judge_used": use_llm_judge and llm_client is not None,
                    "revisions_suggested": suggest_revisions and llm_client is not None,
                },
            )
        )

    return all_results


async def verify_batch_async(
    questions: list[str] | str,
    answers: list[str] | str,
    contexts: list[str]
    | list[Chunk]
    | list[dict[str, Any]]
    | list[list[str] | list[Chunk] | list[dict[str, Any]]],
    *,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    use_llm_judge: bool = False,
    llm_client: LLMClient | None = None,
    trust_threshold: float = 0.70,
    weights: dict[str, float] | None = None,
    suggest_revisions: bool = False,
    batch_size: int = 32,
) -> list[VerificationResult]:
    """Async version of verify_batch().

    Offloads NLI model inference to a thread pool.

    Args:
        Same as verify_batch().

    Returns:
        List of VerificationResult objects.
    """
    single_question = isinstance(questions, str)
    if single_question:
        questions_list: list[str] = [questions]
        answers_list: list[str] = [answers]
        contexts_list: list[Any] = [contexts]
    else:
        questions_list = questions
        answers_list = answers
        if isinstance(contexts, list) and len(contexts) > 0:
            first = contexts[0]
            if isinstance(first, (str, dict, Chunk)):
                contexts_list = [contexts] * len(questions_list)
            else:
                contexts_list = contexts
        else:
            contexts_list = [contexts] * len(questions_list)

    all_results: list[VerificationResult] = []

    start_time = time.time()

    all_chunks: list[list[Chunk]] = []
    all_sentences: list[list[str]] = []
    all_nli_pairs: list[tuple[str, str]] = []
    pair_offsets: list[int] = []

    for q_idx in range(len(questions_list)):
        chunks = [Chunk.from_input(c) for c in contexts_list[q_idx]]
        chunk_texts = [c.content for c in chunks]
        combined_context = " ".join(chunk_texts)
        sentences = split_sentences(answers_list[q_idx])

        all_chunks.append(chunks)
        all_sentences.append(sentences)

        offset = len(all_nli_pairs)
        pair_offsets.append(offset)

        for sentence in sentences:
            all_nli_pairs.append((combined_context, sentence))

    nli_scores_all = await batch_compute_entailment_async(
        all_nli_pairs, model_name=nli_model, batch_size=batch_size
    )

    for q_idx in range(len(questions_list)):
        chunks = all_chunks[q_idx]
        chunk_texts = [c.content for c in chunks]
        sentences = all_sentences[q_idx]

        if not sentences:
            all_results.append(
                VerificationResult(
                    question=questions_list[q_idx],
                    answer=answers_list[q_idx],
                    trust_score=0.0,
                    sentences=[],
                    unsupported=[],
                    supported=[],
                    verification_passed=False,
                    metadata={"error": "no_sentences_found", "latency_ms": 0},
                )
            )
            continue

        offset = pair_offsets[q_idx]
        sentence_scores: list[SentenceScore] = []
        llm_scores: list[float | None] = [None] * len(sentences)

        if use_llm_judge and llm_client is not None:
            combined_context = " ".join(chunk_texts)
            judge_results = batch_judge_sentences(
                sentences, combined_context, questions_list[q_idx], llm_client
            )
            llm_scores = [score for score, _ in judge_results]

        for i, sentence in enumerate(sentences):
            nli = nli_scores_all[offset + i] if (offset + i) < len(nli_scores_all) else 0.0
            overlap, best_chunk = best_overlap_score(sentence, chunk_texts)
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

        overall_trust, passed = compute_overall_trust(sentence_scores, trust_threshold)

        supported = [s for s in sentence_scores if s.support_status in ("SUPPORTED", "PARTIAL")]
        unsupported = [
            s for s in sentence_scores if s.support_status in ("UNSUPPORTED", "CONTRADICTED")
        ]

        if suggest_revisions and llm_client is not None and unsupported:
            combined_context = " ".join(chunk_texts)
            revisions = batch_generate_revisions(
                [s.text for s in unsupported],
                combined_context,
                questions_list[q_idx],
                llm_client,
            )
            for sent, revision in zip(unsupported, revisions, strict=True):
                sent.suggested_fix = revision

        all_results.append(
            VerificationResult(
                question=questions_list[q_idx],
                answer=answers_list[q_idx],
                trust_score=round(overall_trust, 4),
                sentences=sentence_scores,
                unsupported=unsupported,
                supported=supported,
                verification_passed=passed,
                metadata={
                    "nli_model": nli_model,
                    "num_chunks": len(chunks),
                    "num_sentences": len(sentences),
                    "latency_ms": round((time.time() - start_time) * 1000, 1),
                    "llm_judge_used": use_llm_judge and llm_client is not None,
                    "revisions_suggested": suggest_revisions and llm_client is not None,
                },
            )
        )

    return all_results


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

        judge: LLMClient = OpenAIJudge(model=model, api_key=api_key)
        answer = judge.complete(prompt)
    elif provider == "anthropic":
        from athena_verify.llm_judge import AnthropicJudge

        judge = AnthropicJudge(model=model, api_key=api_key)
        answer = judge.complete(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    return verify(
        question=question,
        answer=answer,
        context=context,
        nli_model=nli_model,
        trust_threshold=trust_threshold,
    )


async def verify_stream(
    question: str,
    answer_stream: AsyncIterator[str],
    context: list[str] | list[Chunk] | list[dict[str, Any]],
    *,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    trust_threshold: float = 0.70,
    weights: dict[str, float] | None = None,
) -> AsyncIterator[StreamingResult]:
    """Verify a streaming LLM answer incrementally, sentence-by-sentence.

    Buffers incoming tokens until a sentence boundary is reached, then
    runs NLI + lexical-overlap verification on the completed sentence
    and yields an updated :class:`StreamingResult`.

    The final yield has ``is_final=True`` and contains the full set of
    scored sentences with a calibrated overall trust score.

    Args:
        question: The original question.
        answer_stream: Async iterator yielding LLM tokens.
        context: Retrieved context chunks.
        nli_model: Cross-encoder model for NLI scoring.
        trust_threshold: Minimum trust score for verification to pass.
        weights: Custom weights for the trust score ensemble.

    Yields:
        StreamingResult — one per completed sentence, plus a final result.
    """
    start_time = time.time()

    chunks = [Chunk.from_input(c) for c in context]
    chunk_texts = [c.content for c in chunks]
    combined_context = " ".join(chunk_texts)

    sentence_scores: list[SentenceScore] = []
    idx = 0

    async for sentence in sentence_buffer(answer_stream):
        nli_pairs = [(combined_context, sentence)]
        nli_scores = await batch_compute_entailment_async(nli_pairs, model_name=nli_model)
        nli = nli_scores[0] if nli_scores else 0.0

        overlap, best_chunk = best_overlap_score(sentence, chunk_texts)
        trust = compute_trust_score(nli, overlap, None, weights)
        status = classify_support(trust)

        score = SentenceScore(
            text=sentence,
            index=idx,
            nli_score=nli,
            lexical_overlap=overlap,
            trust_score=trust,
            support_status=status,
            best_matching_context=best_chunk,
        )
        sentence_scores.append(score)
        idx += 1

        overall_trust, _ = compute_overall_trust(sentence_scores, trust_threshold)

        yield StreamingResult(
            trust_score=round(overall_trust, 4),
            sentences=list(sentence_scores),
            is_final=False,
        )

    overall_trust, passed = compute_overall_trust(sentence_scores, trust_threshold)
    latency_ms = (time.time() - start_time) * 1000

    yield StreamingResult(
        trust_score=round(overall_trust, 4),
        sentences=list(sentence_scores),
        is_final=True,
        metadata={
            "nli_model": nli_model,
            "num_chunks": len(chunks),
            "num_sentences": len(sentence_scores),
            "latency_ms": round(latency_ms, 1),
            "verification_passed": passed,
        },
    )
