"""Tests for new features: streaming, NLI aliases, revisions, batch, observability.

Uses mock NLI model to avoid requiring sentence-transformers in CI.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from athena_verify import verify, verify_batch, verify_batch_async, verify_stream
from athena_verify.llm_judge import (
    batch_generate_revisions,
    generate_revision,
)
from athena_verify.models import (
    SentenceScore,
    StreamingResult,
    VerificationResult,
)
from athena_verify.nli import NLI_MODEL_ALIASES, resolve_nli_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class StubLLMClient:
    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["The cap is $1M per incident."]
        self._call_count = 0

    def complete(self, prompt: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


class MockCrossEncoder:
    def predict(self, pairs):
        return [[0.9, 0.05, 0.05]] * len(pairs)


@pytest.fixture(autouse=True)
def _mock_nli():
    with (
        patch("athena_verify.core.batch_compute_entailment", return_value=[0.85]),
        patch("athena_verify.core.batch_compute_entailment_async", return_value=[0.85]),
    ):
        yield


async def _token_stream(tokens: list[str]):
    for token in tokens:
        yield token


def _make_result(**kwargs) -> VerificationResult:
    defaults = dict(
        question="What is the cap?",
        answer="The cap is $1M.",
        context=["The agreement sets a cap of $1M per incident."],
    )
    defaults.update(kwargs)
    return verify(**defaults)


# ---------------------------------------------------------------------------
# 1. Lightweight NLI Fallback
# ---------------------------------------------------------------------------


class TestNLIModelAliases:
    def test_default_alias(self):
        assert resolve_nli_model("default") == "cross-encoder/nli-deberta-v3-base"

    def test_lightweight_alias(self):
        assert resolve_nli_model("lightweight") == "cross-encoder/nli-MiniLM2-L6-H768"

    def test_vectara_alias(self):
        assert resolve_nli_model("vectara") == "vectara/hallucination_evaluation_model"

    def test_deberta_base_alias(self):
        assert resolve_nli_model("deberta-base") == "MoritzLaworr/NLI-deberta-base"

    def test_full_model_name_passthrough(self):
        model = "cross-encoder/nli-deberta-v3-base"
        assert resolve_nli_model(model) == model

    def test_unknown_alias_passthrough(self):
        assert resolve_nli_model("some-unknown-model") == "some-unknown-model"

    def test_aliases_dict_has_expected_keys(self):
        for key in ("default", "lightweight", "vectara", "deberta-base"):
            assert key in NLI_MODEL_ALIASES


# ---------------------------------------------------------------------------
# 2. Suggested Revision
# ---------------------------------------------------------------------------


class TestSuggestedRevision:
    def test_generate_revision_basic(self):
        client = StubLLMClient(responses=["The cap is $1M per incident."])
        revision = generate_revision(
            sentence="The cap is $5M per incident.",
            context="The agreement sets a cap of $1M per incident.",
            question="What is the cap?",
            client=client,
        )
        assert revision == "The cap is $1M per incident."

    def test_generate_revision_insufficient_context(self):
        client = StubLLMClient(responses=["INSUFFICIENT_CONTEXT"])
        revision = generate_revision(
            sentence="The cap is $5M.",
            context="Unrelated.",
            question="What?",
            client=client,
        )
        assert revision is None

    def test_generate_revision_empty_response(self):
        client = StubLLMClient(responses=[""])
        revision = generate_revision(sentence="Test.", context="Ctx.", question="Q?", client=client)
        assert revision is None

    def test_generate_revision_exception(self):
        client = MagicMock()
        client.complete.side_effect = RuntimeError("boom")
        revision = generate_revision("S.", "C.", "Q?", client)
        assert revision is None

    def test_batch_generate_revisions(self):
        client = StubLLMClient(responses=["Fixed 1.", "Fixed 2.", "Fixed 3."])
        revisions = batch_generate_revisions(["Bad 1.", "Bad 2.", "Bad 3."], "Ctx.", "Q?", client)
        assert revisions == ["Fixed 1.", "Fixed 2.", "Fixed 3."]

    def test_batch_generate_revisions_mixed(self):
        client = StubLLMClient(responses=["Fixed.", "INSUFFICIENT_CONTEXT"])
        revisions = batch_generate_revisions(["Bad 1.", "Bad 2."], "Ctx.", "Q?", client)
        assert revisions[0] == "Fixed."
        assert revisions[1] is None

    def test_verify_suggest_revisions_no_client(self):
        result = _make_result(suggest_revisions=True, llm_client=None)
        assert isinstance(result, VerificationResult)
        for s in result.unsupported:
            assert s.suggested_fix is None
        assert result.metadata["revisions_suggested"] is False

    def test_verify_suggest_revisions_with_client(self):
        client = StubLLMClient(responses=["Corrected sentence."])
        result = verify(
            question="What?",
            answer="Completely wrong statement here.",
            context=["Something entirely different is described."],
            suggest_revisions=True,
            llm_client=client,
        )
        assert isinstance(result, VerificationResult)
        assert result.metadata["revisions_suggested"] is True

    def test_sentence_score_suggested_fix_default_none(self):
        score = SentenceScore(
            text="T.",
            index=0,
            nli_score=0.5,
            lexical_overlap=0.5,
            trust_score=0.5,
            support_status="SUPPORTED",
        )
        assert score.suggested_fix is None

    def test_sentence_score_suggested_fix_set(self):
        score = SentenceScore(
            text="T.",
            index=0,
            nli_score=0.2,
            lexical_overlap=0.1,
            trust_score=0.15,
            support_status="UNSUPPORTED",
            suggested_fix="Corrected.",
        )
        assert score.suggested_fix == "Corrected."


# ---------------------------------------------------------------------------
# 3. Batch Verification API
# ---------------------------------------------------------------------------


class TestVerifyBatch:
    def test_batch_single_question(self):
        results = verify_batch(
            questions="What is the cap?",
            answers="The cap is $1M.",
            contexts=["The cap is $1M per incident."],
        )
        assert len(results) == 1
        assert isinstance(results[0], VerificationResult)
        assert results[0].question == "What is the cap?"

    def test_batch_multiple_questions(self):
        results = verify_batch(
            questions=["What is the cap?", "What is Python?"],
            answers=["The cap is $1M.", "Python is a language."],
            contexts=[["The cap is $1M."], ["Python is a programming language."]],
        )
        assert len(results) == 2
        assert results[0].question == "What is the cap?"
        assert results[1].question == "What is Python?"

    def test_batch_shared_context(self):
        ctx = ["The cap is $1M. Python is a programming language."]
        results = verify_batch(
            questions=["Q1?", "Q2?"],
            answers=["The cap is $1M.", "Python is a language."],
            contexts=ctx,
        )
        assert len(results) == 2

    def test_batch_empty_answer(self):
        results = verify_batch(
            questions="What?",
            answers="",
            contexts=["Some context"],
        )
        assert len(results) == 1
        assert results[0].trust_score == 0.0

    def test_batch_metadata(self):
        results = verify_batch(
            questions="What?",
            answers="Some answer.",
            contexts=["Ctx."],
        )
        assert "latency_ms" in results[0].metadata
        assert "num_chunks" in results[0].metadata

    def test_batch_with_revisions(self):
        client = StubLLMClient(responses=["Corrected."])
        results = verify_batch(
            questions="What?",
            answers="Wrong answer here.",
            contexts=["Different context entirely."],
            suggest_revisions=True,
            llm_client=client,
        )
        assert results[0].metadata["revisions_suggested"] is True

    @pytest.mark.asyncio
    async def test_batch_async_single(self):
        results = await verify_batch_async(
            questions="What is the cap?",
            answers="The cap is $1M.",
            contexts=["The cap is $1M per incident."],
        )
        assert len(results) == 1
        assert isinstance(results[0], VerificationResult)

    @pytest.mark.asyncio
    async def test_batch_async_multiple(self):
        results = await verify_batch_async(
            questions=["Q1?", "Q2?"],
            answers=["A1.", "A2."],
            contexts=[["Ctx 1."], ["Ctx 2."]],
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_batch_async_empty(self):
        results = await verify_batch_async(
            questions="What?",
            answers="",
            contexts=["Ctx."],
        )
        assert results[0].trust_score == 0.0


# ---------------------------------------------------------------------------
# 4. Streaming Support
# ---------------------------------------------------------------------------


class TestVerifyStream:
    @pytest.mark.asyncio
    async def test_stream_single_sentence(self):
        results = []
        async for r in verify_stream(
            question="What color is the sky?",
            answer_stream=_token_stream(["The sky is blue."]),
            context=["The sky appears blue during the day."],
        ):
            results.append(r)

        assert len(results) >= 1
        final = results[-1]
        assert isinstance(final, StreamingResult)
        assert final.is_final is True
        assert len(final.sentences) == 1

    @pytest.mark.asyncio
    async def test_stream_multiple_sentences(self):
        results = []
        async for r in verify_stream(
            question="About Python.",
            answer_stream=_token_stream(["Python is a language.", " It was created by Guido."]),
            context=[
                "Python is a high-level programming language.",
                "Python was created by Guido van Rossum.",
            ],
        ):
            results.append(r)

        assert len(results) >= 2
        final = results[-1]
        assert final.is_final is True
        assert len(final.sentences) == 2

    @pytest.mark.asyncio
    async def test_stream_trust_score_range(self):
        async for r in verify_stream(
            question="What?",
            answer_stream=_token_stream(["Some answer."]),
            context=["Some context."],
        ):
            assert 0.0 <= r.trust_score <= 1.0

    @pytest.mark.asyncio
    async def test_stream_incremental_growth(self):
        tokens = ["First sentence.", " Second sentence.", " Third sentence."]
        results = []
        async for r in verify_stream(
            question="Q?",
            answer_stream=_token_stream(tokens),
            context=["First second third sentence context."],
        ):
            results.append(r)

        non_final = [r for r in results if not r.is_final]
        assert len(non_final) >= 1
        for i, r in enumerate(non_final):
            assert len(r.sentences) == i + 1

    @pytest.mark.asyncio
    async def test_stream_final_metadata(self):
        async for r in verify_stream(
            question="What?",
            answer_stream=_token_stream(["Answer."]),
            context=["Context."],
        ):
            if r.is_final:
                assert "latency_ms" in r.metadata
                assert "num_chunks" in r.metadata
                assert "num_sentences" in r.metadata
                assert "verification_passed" in r.metadata

    @pytest.mark.asyncio
    async def test_stream_empty(self):
        results = []
        async for r in verify_stream(
            question="What?",
            answer_stream=_token_stream([]),
            context=["Context."],
        ):
            results.append(r)
        if results:
            assert results[-1].is_final is True
            assert len(results[-1].sentences) == 0

    @pytest.mark.asyncio
    async def test_stream_to_json(self):
        async for r in verify_stream(
            question="What?",
            answer_stream=_token_stream(["Answer."]),
            context=["Context."],
        ):
            parsed = json.loads(r.to_json())
            assert "trust_score" in parsed
            assert "sentences" in parsed

    @pytest.mark.asyncio
    async def test_stream_to_dict(self):
        async for r in verify_stream(
            question="What?",
            answer_stream=_token_stream(["Answer."]),
            context=["Context."],
        ):
            d = r.to_dict()
            assert isinstance(d, dict)
            assert "trust_score" in d


# ---------------------------------------------------------------------------
# 5. Structured Logging / Observability
# ---------------------------------------------------------------------------


class TestObservabilityOutput:
    def test_to_json_valid(self):
        result = _make_result()
        parsed = json.loads(result.to_json())
        assert "trust_score" in parsed
        assert "question" in parsed
        assert "sentences" in parsed

    def test_to_dict_plain_dict(self):
        result = _make_result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["question"] == "What is the cap?"

    def test_to_json_roundtrip(self):
        result = _make_result()
        parsed = json.loads(result.to_json())
        assert abs(parsed["trust_score"] - result.trust_score) < 0.001

    def test_to_otel_span_structure(self):
        span = _make_result().to_otel_span()
        assert span["name"] == "athena.verify"
        assert "attributes" in span
        assert "events" in span
        attrs = span["attributes"]
        assert "athena.trust_score" in attrs
        assert "athena.verification_passed" in attrs
        assert "athena.num_sentences" in attrs
        assert "athena.question" in attrs

    def test_to_otel_span_metadata(self):
        span = _make_result().to_otel_span()
        assert "athena.metadata.latency_ms" in span["attributes"]
        assert "athena.metadata.num_chunks" in span["attributes"]

    def test_to_otel_span_events_per_sentence(self):
        result = _make_result()
        span = result.to_otel_span()
        assert len(span["events"]) == len(result.sentences)
        for ev in span["events"]:
            assert ev["name"] == "sentence_score"
            assert "text" in ev["attributes"]
            assert "trust_score" in ev["attributes"]

    def test_to_otel_span_suggested_fix(self):
        score = SentenceScore(
            text="Bad.",
            index=0,
            nli_score=0.2,
            lexical_overlap=0.1,
            trust_score=0.15,
            support_status="CONTRADICTED",
            suggested_fix="Good.",
        )
        result = VerificationResult(
            question="Q?",
            answer="Bad.",
            trust_score=0.15,
            sentences=[score],
            unsupported=[score],
            supported=[],
            verification_passed=False,
        )
        ev = result.to_otel_span()["events"][0]
        assert ev["attributes"]["suggested_fix"] == "Good."

    def test_to_otel_span_no_fix_omitted(self):
        score = SentenceScore(
            text="Ok.",
            index=0,
            nli_score=0.9,
            lexical_overlap=0.9,
            trust_score=0.9,
            support_status="SUPPORTED",
        )
        result = VerificationResult(
            question="Q?",
            answer="Ok.",
            trust_score=0.9,
            sentences=[score],
            unsupported=[],
            supported=[score],
            verification_passed=True,
        )
        ev = result.to_otel_span()["events"][0]
        assert "suggested_fix" not in ev["attributes"]

    def test_to_langfuse_structure(self):
        trace = _make_result().to_langfuse_trace()
        assert trace["name"] == "athena-verify"
        assert "metadata" in trace
        assert "output" in trace
        assert "scores" in trace

    def test_to_langfuse_output(self):
        output = _make_result().to_langfuse_trace()["output"]
        assert "trust_score" in output
        assert "verification_passed" in output
        assert "unsupported_count" in output
        assert "supported_count" in output

    def test_to_langfuse_scores_per_sentence(self):
        result = _make_result()
        trace = result.to_langfuse_trace()
        assert len(trace["scores"]) == len(result.sentences)
        for sc in trace["scores"]:
            assert "name" in sc
            assert "value" in sc
            assert "comment" in sc

    def test_to_langfuse_suggested_fix_in_comment(self):
        score = SentenceScore(
            text="Bad.",
            index=0,
            nli_score=0.2,
            lexical_overlap=0.1,
            trust_score=0.15,
            support_status="UNSUPPORTED",
            suggested_fix="Good.",
        )
        result = VerificationResult(
            question="Q?",
            answer="Bad.",
            trust_score=0.15,
            sentences=[score],
            unsupported=[score],
            supported=[],
            verification_passed=False,
        )
        comment = result.to_langfuse_trace()["scores"][0]["comment"]
        assert "Good." in comment

    def test_streaming_result_to_json(self):
        sr = StreamingResult(trust_score=0.85, sentences=[], is_final=False)
        parsed = json.loads(sr.to_json())
        assert parsed["trust_score"] == 0.85
        assert parsed["is_final"] is False

    def test_streaming_result_to_dict(self):
        sr = StreamingResult(trust_score=0.85, is_final=True, metadata={"k": "v"})
        d = sr.to_dict()
        assert d["trust_score"] == 0.85
        assert d["metadata"]["k"] == "v"

    def test_empty_result_otel_span(self):
        result = VerificationResult(
            question="Q?",
            answer="",
            trust_score=0.0,
            sentences=[],
            unsupported=[],
            supported=[],
            verification_passed=False,
        )
        span = result.to_otel_span()
        assert span["attributes"]["athena.num_sentences"] == 0
        assert span["events"] == []

    def test_empty_result_langfuse_trace(self):
        result = VerificationResult(
            question="Q?",
            answer="",
            trust_score=0.0,
            sentences=[],
            unsupported=[],
            supported=[],
            verification_passed=False,
        )
        assert result.to_langfuse_trace()["scores"] == []
