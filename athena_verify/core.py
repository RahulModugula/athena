"""Core verification function — the main entry point for athena-verify.

Provides verify(), verify_async(), verify_stream(), and
verified_completion() for checking whether an LLM answer is grounded
in the provided context chunks.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from athena_verify.calibration import (
    apply_grounding_rescue,
    classify_support,
    compute_overall_trust,
    compute_trust_score,
)
from athena_verify.claims import is_checkworthy
from athena_verify.llm_judge import LLMClient, batch_generate_revisions, batch_judge_sentences
from athena_verify.models import (
    Chunk,
    SentenceScore,
    StepResult,
    StreamingResult,
    SupportingSpan,
    VerificationResult,
)
from athena_verify.nli import batch_compute_nli
from athena_verify.overlap import best_overlap_score, containment_score, numeric_consistency
from athena_verify.parser import sentence_buffer, split_sentences

logger = structlog.get_logger()

# Span-level entailment threshold: a context unit must clear this to be
# reported as a supporting span for a sentence.
_SPAN_ENTAILMENT_THRESHOLD = 0.5

# Return shape of _ground_sentences: (entailment scores, contradiction scores,
# per-sentence span-unit scores, span unit texts, span unit (chunk_idx, start,
# end) locations).
_GroundResult = tuple[
    list[float], list[float], list[list[float]], list[str], list[tuple[int, int, int]]
]

# Leading tokens that signal a sentence depends on its predecessor for meaning
# (anaphora / discourse continuation). When an answer sentence starts with one
# of these, NLI scored on the sentence in isolation collapses to ~0 even when
# the claim is fully grounded, because the referent ("it", "this cap") is gone.
# We prepend the previous sentence to restore the antecedent before scoring.
_ANAPHORA_TOKENS = frozenset(
    {
        "it",
        "its",
        "it's",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "their",
        "theirs",
        "he",
        "she",
        "his",
        "her",
        "hers",
        "such",
        "also",
        "additionally",
        "moreover",
        "furthermore",
        "however",
        "therefore",
        "thus",
        "then",
        "there",
        "both",
        "neither",
        "either",
    }
)


def _starts_with_anaphor(sentence: str) -> bool:
    """True if a sentence opens with a pronoun/discourse marker needing context."""
    stripped = sentence.strip()
    if not stripped:
        return False
    first = stripped.split(maxsplit=1)[0].lower().strip(",.;:\"'()")
    return first in _ANAPHORA_TOKENS


def _word_tokens(text: str) -> list[str]:
    """Lowercase content tokens (length > 2) for lightweight topic matching."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]


def _build_context_units(
    chunk_texts: list[str],
) -> tuple[list[str], list[tuple[int, int, int]], list[bool]]:
    """Expand context chunks into NLI premise candidates.

    Returns parallel lists of:
      - unit text,
      - (chunk_idx, char_start, char_end) location into the original chunk,
      - is_span_unit flag (True for sentence-level units usable as precise
        supporting spans, False for whole-chunk fallback premises).

    Each chunk contributes its individual sentences (focused premises that
    avoid the long-premise "neutral" bias) plus, when it has more than one
    sentence, the full chunk text (so facts spread across several sentences
    are still entailed). NLI takes the max over all candidates, so adding the
    whole-chunk premise can only raise a faithful sentence's score.
    """
    units: list[str] = []
    locations: list[tuple[int, int, int]] = []
    is_span_unit: list[bool] = []
    for chunk_idx, chunk in enumerate(chunk_texts):
        sub = split_sentences(chunk) or [chunk]
        for unit in sub:
            char_start = chunk.find(unit)
            if char_start == -1:
                char_start = 0
            units.append(unit)
            locations.append((chunk_idx, char_start, char_start + len(unit)))
            is_span_unit.append(True)
        if len(sub) > 1:
            units.append(chunk)
            locations.append((chunk_idx, 0, len(chunk)))
            is_span_unit.append(False)
    return units, locations, is_span_unit


def _ground_sentences(
    sentences: list[str],
    chunk_texts: list[str],
    nli_model: str,
) -> _GroundResult:
    """Score how well each answer sentence is grounded in the context.

    Splits context into focused premise candidates, applies anaphora windowing
    to each hypothesis, and returns:
      - entail_scores: best entailment per sentence over all premise candidates,
      - contra_scores: strongest contradiction per sentence over all candidates,
      - span_scores: per-sentence entailment over the span-eligible units only,
      - span_units / span_locations: the span-eligible units these align to.
    """
    units, locations, is_span_unit = _build_context_units(chunk_texts)
    span_units = [u for u, keep in zip(units, is_span_unit, strict=True) if keep]
    span_locations = [
        loc for loc, keep in zip(locations, is_span_unit, strict=True) if keep
    ]

    if not units or not sentences:
        empty_spans: list[list[float]] = [[] for _ in sentences]
        zeros = [0.0] * len(sentences)
        return zeros, list(zeros), empty_spans, span_units, span_locations

    # Anaphora windowing: prepend the previous sentence when the current one
    # opens with a referent, so the NLI hypothesis carries its antecedent.
    hypotheses: list[str] = []
    for i, sentence in enumerate(sentences):
        if i > 0 and _starts_with_anaphor(sentence):
            hypotheses.append(f"{sentences[i - 1]} {sentence}")
        else:
            hypotheses.append(sentence)

    nli_pairs = [(unit, hyp) for hyp in hypotheses for unit in units]
    flat = batch_compute_nli(nli_pairs, model_name=nli_model)

    # Token sets per unit, for picking the on-topic unit for the contradiction
    # signal (see below).
    unit_tokens = [set(_word_tokens(u)) for u in units]

    entail_scores: list[float] = []
    contra_scores: list[float] = []
    span_scores: list[list[float]] = []
    n_units = len(units)
    for i in range(len(sentences)):
        rows = flat[i * n_units : (i + 1) * n_units]
        entails = [e for e, _ in rows]
        entail_scores.append(max(entails) if entails else 0.0)
        # Contradiction is read from the unit most lexically on-topic with the
        # claim, not the global max. An unrelated context unit frequently
        # "contradicts" a claim it has nothing to do with (negations, sibling
        # clauses), which would veto a faithful sentence; the on-topic unit
        # still fires for genuine contradictions (number swaps, reversals)
        # because those reuse the same vocabulary.
        hyp_tokens = set(_word_tokens(hypotheses[i]))
        if rows and hyp_tokens:
            relevance = [len(hyp_tokens & ut) for ut in unit_tokens]
            topic = max(range(len(rows)), key=lambda k: (relevance[k], entails[k]))
            contra_scores.append(rows[topic][1])
        else:
            contra_scores.append(0.0)
        span_scores.append(
            [e for e, keep in zip(entails, is_span_unit, strict=True) if keep]
        )
    return entail_scores, contra_scores, span_scores, span_units, span_locations


def _trust_and_status(
    *,
    entailment: float,
    contradiction: float,
    overlap: float,
    sentence: str,
    context_text: str,
    llm: float | None,
    weights: dict[str, float] | None,
) -> tuple[float, str]:
    """Combine signals into a trust score, apply the grounding rescue, classify.

    Shared by every verify entry point so they score identically.
    """
    trust = compute_trust_score(entailment, overlap, llm, weights)
    trust = apply_grounding_rescue(
        trust,
        entailment=entailment,
        contradiction=contradiction,
        containment=containment_score(sentence, context_text),
        numeric_ok=numeric_consistency(sentence, context_text),
    )
    # Questions and meta/refusal statements aren't verifiable claims — never
    # flag them as hallucinations (they're usually the honest, correct response).
    if not is_checkworthy(sentence):
        return trust, "NOT_A_CLAIM"
    return trust, classify_support(trust)


def verify_step(
    claim: str,
    evidence: str | list[str],
    threshold: float = 0.7,
    *,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
) -> StepResult:
    """Verify a single claim against evidence for multi-step agents.

    Treats the claim as both question and answer, verifying it against
    the provided evidence. Returns a StepResult suitable for circuit-breaker
    patterns where agents halt on fabricated intermediate claims.

    Args:
        claim: The factual claim to verify.
        evidence: Evidence supporting or refuting the claim (str or list of str).
        threshold: Minimum trust score for claim to pass (default 0.7).
        nli_model: Cross-encoder model name for NLI scoring.

    Returns:
        StepResult with passed flag, trust_score, and action ("continue" or "halt").
    """
    if isinstance(evidence, str):
        evidence = [evidence]

    result = verify(
        question=claim,
        answer=claim,
        context=evidence,
        nli_model=nli_model,
        trust_threshold=threshold,
    )

    return StepResult(
        passed=result.verification_passed,
        trust_score=result.trust_score,
        action="continue" if result.verification_passed else "halt",
        sentences=result.sentences,
    )


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
    latency_budget_ms: int | None = None,
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
        latency_budget_ms: Latency budget in milliseconds. None (default): current behavior.
            <= 100: skip LLM judge. > 100: only escalate borderline if budget covers it.

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
    # NLI works best on short, focused premises with hypotheses that carry
    # their own referents. _ground_sentences handles both: it scores each
    # sentence against individual context sentences plus the whole chunk
    # (max wins), and prepends the prior sentence when a hypothesis opens with
    # an anaphor. See athena_verify.core helpers for details.
    entail_scores, contra_scores, per_sentence_unit_scores, span_units, span_locations = (
        _ground_sentences(sentences, chunk_texts, nli_model)
    )
    context_text = " ".join(chunk_texts)

    # --- Lexical overlap scoring ---
    overlap_results = [best_overlap_score(s, chunk_texts) for s in sentences]

    # --- Optional LLM-as-judge scoring with latency budget ---
    llm_scores: list[float | None] = [None] * len(sentences)
    budget_exceeded = False
    llm_judge_avg_ms = 2000.0

    should_use_llm = use_llm_judge and llm_client is not None
    if latency_budget_ms is not None and latency_budget_ms <= 100:
        should_use_llm = False

    if should_use_llm and latency_budget_ms is not None and latency_budget_ms > 100:
        elapsed_so_far = (time.time() - start_time) * 1000
        remaining_budget = latency_budget_ms - elapsed_so_far

        if remaining_budget < llm_judge_avg_ms:
            should_use_llm = False
            budget_exceeded = True

    if should_use_llm:
        combined_context = " ".join(chunk_texts)
        judge_start = time.time()
        judge_results = batch_judge_sentences(sentences, combined_context, question, llm_client)
        llm_scores = [score for score, _ in judge_results]
        llm_judge_avg_ms = (
            (time.time() - judge_start) * 1000 / len(sentences) if sentences else 2000.0
        )

    # --- Build per-sentence results ---
    sentence_scores: list[SentenceScore] = []
    for i, sentence in enumerate(sentences):
        nli = entail_scores[i] if i < len(entail_scores) else 0.0
        contra = contra_scores[i] if i < len(contra_scores) else 0.0
        overlap, best_chunk = overlap_results[i]
        llm = llm_scores[i] if i < len(llm_scores) else None

        trust, status = _trust_and_status(
            entailment=nli,
            contradiction=contra,
            overlap=overlap,
            sentence=sentence,
            context_text=context_text,
            llm=llm,
            weights=weights,
        )

        unit_scores_i = per_sentence_unit_scores[i] if i < len(per_sentence_unit_scores) else []
        supporting_spans = [
            SupportingSpan(
                chunk_idx=span_locations[j][0],
                start=span_locations[j][1],
                end=span_locations[j][2],
                text=span_units[j],
            )
            for j, score in enumerate(unit_scores_i)
            if score >= _SPAN_ENTAILMENT_THRESHOLD
        ]

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
                supporting_spans=supporting_spans,
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

    metadata_dict: dict[str, Any] = {
        "nli_model": nli_model,
        "num_chunks": len(chunks),
        "num_sentences": len(sentences),
        "latency_ms": round(latency_ms, 1),
        "llm_judge_used": should_use_llm,
        "revisions_suggested": suggest_revisions and llm_client is not None,
    }
    if latency_budget_ms is not None:
        metadata_dict["budget_exceeded"] = budget_exceeded

    result = VerificationResult(
        question=question,
        answer=answer,
        trust_score=round(overall_trust, 4),
        sentences=sentence_scores,
        unsupported=unsupported,
        supported=supported,
        verification_passed=passed,
        metadata=metadata_dict,
    )

    if os.getenv("ATHENA_OTEL_ENABLED") == "1":
        otel_span = result.to_otel_span()
        logger.info("otel_span_generated", span=otel_span)

    if os.getenv("ATHENA_LANGFUSE_ENABLED") == "1":
        langfuse_trace = result.to_langfuse_trace()
        logger.info("langfuse_trace_generated", trace=langfuse_trace)

    return result


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
    latency_budget_ms: int | None = None,
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
    # Offload the same grounding logic used by verify() to a thread so we get
    # per-unit + whole-chunk premises and anaphora windowing here too, instead
    # of the old concatenate-all-chunks premise that silently truncated at the
    # model's token limit.
    (
        entail_scores,
        contra_scores,
        per_sentence_unit_scores,
        span_units,
        span_locations,
    ) = await asyncio.to_thread(_ground_sentences, sentences, chunk_texts, nli_model)
    context_text = " ".join(chunk_texts)

    # --- Lexical overlap scoring ---
    overlap_results = [best_overlap_score(s, chunk_texts) for s in sentences]

    # --- Optional LLM-as-judge scoring with latency budget ---
    llm_scores: list[float | None] = [None] * len(sentences)
    budget_exceeded = False
    llm_judge_avg_ms = 2000.0

    should_use_llm = use_llm_judge and llm_client is not None
    if latency_budget_ms is not None and latency_budget_ms <= 100:
        should_use_llm = False

    if should_use_llm and latency_budget_ms is not None and latency_budget_ms > 100:
        elapsed_so_far = (time.time() - start_time) * 1000
        remaining_budget = latency_budget_ms - elapsed_so_far

        if remaining_budget < llm_judge_avg_ms:
            should_use_llm = False
            budget_exceeded = True

    if should_use_llm:
        combined_context = " ".join(chunk_texts)
        judge_start = time.time()
        judge_results = batch_judge_sentences(sentences, combined_context, question, llm_client)
        llm_scores = [score for score, _ in judge_results]
        llm_judge_avg_ms = (
            (time.time() - judge_start) * 1000 / len(sentences) if sentences else 2000.0
        )

    # --- Build per-sentence results ---
    sentence_scores: list[SentenceScore] = []
    for i, sentence in enumerate(sentences):
        nli = entail_scores[i] if i < len(entail_scores) else 0.0
        contra = contra_scores[i] if i < len(contra_scores) else 0.0
        overlap, best_chunk = overlap_results[i]
        llm = llm_scores[i] if i < len(llm_scores) else None

        trust, status = _trust_and_status(
            entailment=nli,
            contradiction=contra,
            overlap=overlap,
            sentence=sentence,
            context_text=context_text,
            llm=llm,
            weights=weights,
        )

        unit_scores_i = per_sentence_unit_scores[i] if i < len(per_sentence_unit_scores) else []
        supporting_spans = [
            SupportingSpan(
                chunk_idx=span_locations[j][0],
                start=span_locations[j][1],
                end=span_locations[j][2],
                text=span_units[j],
            )
            for j, score in enumerate(unit_scores_i)
            if score >= _SPAN_ENTAILMENT_THRESHOLD
        ]

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
                supporting_spans=supporting_spans,
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

    metadata_dict: dict[str, Any] = {
        "nli_model": nli_model,
        "num_chunks": len(chunks),
        "num_sentences": len(sentences),
        "latency_ms": round(latency_ms, 1),
        "llm_judge_used": should_use_llm,
        "revisions_suggested": suggest_revisions and llm_client is not None,
    }
    if latency_budget_ms is not None:
        metadata_dict["budget_exceeded"] = budget_exceeded

    result = VerificationResult(
        question=question,
        answer=answer,
        trust_score=round(overall_trust, 4),
        sentences=sentence_scores,
        unsupported=unsupported,
        supported=supported,
        verification_passed=passed,
        metadata=metadata_dict,
    )

    if os.getenv("ATHENA_OTEL_ENABLED") == "1":
        otel_span = result.to_otel_span()
        logger.info("otel_span_generated", span=otel_span)

    if os.getenv("ATHENA_LANGFUSE_ENABLED") == "1":
        langfuse_trace = result.to_langfuse_trace()
        logger.info("langfuse_trace_generated", trace=langfuse_trace)

    return result


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
        questions_list: list[str] = [questions]  # type: ignore[list-item]
        answers_list: list[str] = [answers]  # type: ignore[list-item]
        contexts_list: list[Any] = [contexts]
    else:
        questions_list = list(questions)
        answers_list = list(answers)
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

    for q_idx in range(len(questions_list)):
        chunks = [Chunk.from_input(c) for c in contexts_list[q_idx]]
        sentences = split_sentences(answers_list[q_idx])
        all_chunks.append(chunks)
        all_sentences.append(sentences)

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

        entail_scores, contra_scores, per_sentence_unit_scores, span_units, span_locations = (
            _ground_sentences(sentences, chunk_texts, nli_model)
        )
        context_text = " ".join(chunk_texts)
        sentence_scores: list[SentenceScore] = []
        llm_scores: list[float | None] = [None] * len(sentences)

        if use_llm_judge and llm_client is not None:
            judge_results = batch_judge_sentences(
                sentences, context_text, questions_list[q_idx], llm_client
            )
            llm_scores = [score for score, _ in judge_results]

        for i, sentence in enumerate(sentences):
            nli = entail_scores[i] if i < len(entail_scores) else 0.0
            contra = contra_scores[i] if i < len(contra_scores) else 0.0
            overlap, best_chunk = best_overlap_score(sentence, chunk_texts)
            llm = llm_scores[i] if i < len(llm_scores) else None

            trust, status = _trust_and_status(
                entailment=nli,
                contradiction=contra,
                overlap=overlap,
                sentence=sentence,
                context_text=context_text,
                llm=llm,
                weights=weights,
            )

            unit_scores_i = per_sentence_unit_scores[i] if i < len(per_sentence_unit_scores) else []
            supporting_spans = [
                SupportingSpan(
                    chunk_idx=span_locations[j][0],
                    start=span_locations[j][1],
                    end=span_locations[j][2],
                    text=span_units[j],
                )
                for j, score in enumerate(unit_scores_i)
                if score >= _SPAN_ENTAILMENT_THRESHOLD
            ]

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
                    supporting_spans=supporting_spans,
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
        questions_list: list[str] = [questions]  # type: ignore[list-item]
        answers_list: list[str] = [answers]  # type: ignore[list-item]
        contexts_list: list[Any] = [contexts]
    else:
        questions_list = list(questions)
        answers_list = list(answers)
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

    for q_idx in range(len(questions_list)):
        chunks = [Chunk.from_input(c) for c in contexts_list[q_idx]]
        sentences = split_sentences(answers_list[q_idx])
        all_chunks.append(chunks)
        all_sentences.append(sentences)

    # Ground every question with the shared per-unit + windowing logic, offloaded
    # to a single worker thread so we don't block the event loop.
    def _ground_all() -> list[_GroundResult]:
        out: list[_GroundResult] = []
        for q_idx in range(len(questions_list)):
            sents = all_sentences[q_idx]
            if not sents:
                out.append(([], [], [], [], []))
                continue
            texts = [c.content for c in all_chunks[q_idx]]
            out.append(_ground_sentences(sents, texts, nli_model))
        return out

    grounding = await asyncio.to_thread(_ground_all)

    for q_idx in range(len(questions_list)):
        try:
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

            (
                entail_scores,
                contra_scores,
                per_sentence_unit_scores,
                span_units,
                span_locations,
            ) = grounding[q_idx]
            context_text = " ".join(chunk_texts)
            sentence_scores: list[SentenceScore] = []
            llm_scores: list[float | None] = [None] * len(sentences)

            if use_llm_judge and llm_client is not None:
                judge_results = batch_judge_sentences(
                    sentences, context_text, questions_list[q_idx], llm_client
                )
                llm_scores = [score for score, _ in judge_results]

            for i, sentence in enumerate(sentences):
                nli = entail_scores[i] if i < len(entail_scores) else 0.0
                contra = contra_scores[i] if i < len(contra_scores) else 0.0
                overlap, best_chunk = best_overlap_score(sentence, chunk_texts)
                llm = llm_scores[i] if i < len(llm_scores) else None

                trust, status = _trust_and_status(
                    entailment=nli,
                    contradiction=contra,
                    overlap=overlap,
                    sentence=sentence,
                    context_text=context_text,
                    llm=llm,
                    weights=weights,
                )

                unit_scores_i = (
                    per_sentence_unit_scores[i] if i < len(per_sentence_unit_scores) else []
                )
                supporting_spans = [
                    SupportingSpan(
                        chunk_idx=span_locations[j][0],
                        start=span_locations[j][1],
                        end=span_locations[j][2],
                        text=span_units[j],
                    )
                    for j, score in enumerate(unit_scores_i)
                    if score >= _SPAN_ENTAILMENT_THRESHOLD
                ]

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
                        supporting_spans=supporting_spans,
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
        except Exception as e:
            logger.exception("batch_item_processing_error", q_idx=q_idx, error=str(e))
            all_results.append(
                VerificationResult(
                    question=questions_list[q_idx],
                    answer=answers_list[q_idx],
                    trust_score=0.0,
                    sentences=[],
                    unsupported=[],
                    supported=[],
                    verification_passed=False,
                    metadata={
                        "error": f"batch_processing_failed: {str(e)}",
                        "latency_ms": round((time.time() - start_time) * 1000, 1),
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

    sentence_scores: list[SentenceScore] = []
    idx = 0

    async for sentence in sentence_buffer(answer_stream):
        # Ground each completed sentence with the same per-unit + whole-chunk
        # premises as verify(). When the sentence opens with an anaphor, include
        # the previous one so the referent is present, then keep the current
        # sentence's score (the last entry).
        prev_text = sentence_scores[-1].text if sentence_scores else None
        if prev_text and _starts_with_anaphor(sentence):
            ground_input = [prev_text, sentence]
        else:
            ground_input = [sentence]

        entail_scores, contra_scores, span_scores, span_units, span_locations = (
            await asyncio.to_thread(_ground_sentences, ground_input, chunk_texts, nli_model)
        )
        nli = entail_scores[-1] if entail_scores else 0.0
        contra = contra_scores[-1] if contra_scores else 0.0
        unit_scores_i = span_scores[-1] if span_scores else []

        overlap, best_chunk = best_overlap_score(sentence, chunk_texts)
        trust, status = _trust_and_status(
            entailment=nli,
            contradiction=contra,
            overlap=overlap,
            sentence=sentence,
            context_text=" ".join(chunk_texts),
            llm=None,
            weights=weights,
        )

        supporting_spans = [
            SupportingSpan(
                chunk_idx=span_locations[j][0],
                start=span_locations[j][1],
                end=span_locations[j][2],
                text=span_units[j],
            )
            for j, score in enumerate(unit_scores_i)
            if score >= _SPAN_ENTAILMENT_THRESHOLD
        ]

        score = SentenceScore(
            text=sentence,
            index=idx,
            nli_score=nli,
            lexical_overlap=overlap,
            trust_score=trust,
            support_status=status,
            best_matching_context=best_chunk,
            supporting_spans=supporting_spans,
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
