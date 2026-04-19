"""Tests that SentenceScore.supporting_spans is populated correctly by verify()."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from athena_verify import verify
from athena_verify.models import SupportingSpan


CHUNK_0 = "The sky is blue during the day."
CHUNK_1 = "Photosynthesis occurs in plant cells."

# 2 context units (one per chunk), 2 sentences in answer → 4 NLI pairs:
#   (unit0, sent0), (unit1, sent0), (unit0, sent1), (unit1, sent1)
# Scores: sent0 supported by unit0 (chunk 0), sent1 supported by unit1 (chunk 1).
_NLI_SCORES = [0.9, 0.1, 0.1, 0.85]


@pytest.fixture()
def _mock_nli():
    with patch("athena_verify.core.batch_compute_entailment", return_value=_NLI_SCORES):
        yield


@pytest.mark.usefixtures("_mock_nli")
class TestSupportingSpans:
    def test_supported_sentences_have_spans(self):
        result = verify(
            question="What color is the sky?",
            answer="The sky appears blue. Photosynthesis happens in plants.",
            context=[CHUNK_0, CHUNK_1],
        )
        assert len(result.sentences) == 2
        for sentence in result.sentences:
            assert len(sentence.supporting_spans) >= 1, (
                f"Sentence '{sentence.text}' has no supporting spans"
            )

    def test_span_types(self):
        result = verify(
            question="What color is the sky?",
            answer="The sky appears blue. Photosynthesis happens in plants.",
            context=[CHUNK_0, CHUNK_1],
        )
        for sentence in result.sentences:
            for span in sentence.supporting_spans:
                assert isinstance(span, SupportingSpan)

    def test_span_chunk_indices(self):
        result = verify(
            question="What color is the sky?",
            answer="The sky appears blue. Photosynthesis happens in plants.",
            context=[CHUNK_0, CHUNK_1],
        )
        sent0_chunks = {s.chunk_idx for s in result.sentences[0].supporting_spans}
        sent1_chunks = {s.chunk_idx for s in result.sentences[1].supporting_spans}
        assert 0 in sent0_chunks
        assert 1 in sent1_chunks

    def test_span_offsets_are_valid(self):
        result = verify(
            question="What color is the sky?",
            answer="The sky appears blue. Photosynthesis happens in plants.",
            context=[CHUNK_0, CHUNK_1],
        )
        chunks = [CHUNK_0, CHUNK_1]
        for sentence in result.sentences:
            for span in sentence.supporting_spans:
                chunk_text = chunks[span.chunk_idx]
                assert 0 <= span.start < span.end <= len(chunk_text), (
                    f"Span offsets [{span.start}:{span.end}] out of range for chunk of length {len(chunk_text)}"
                )

    def test_span_text_matches_slice(self):
        result = verify(
            question="What color is the sky?",
            answer="The sky appears blue. Photosynthesis happens in plants.",
            context=[CHUNK_0, CHUNK_1],
        )
        chunks = [CHUNK_0, CHUNK_1]
        for sentence in result.sentences:
            for span in sentence.supporting_spans:
                chunk_text = chunks[span.chunk_idx]
                assert chunk_text[span.start : span.end] == span.text, (
                    f"Span text '{span.text}' does not match slice '{chunk_text[span.start:span.end]}'"
                )

    def test_no_spans_below_threshold(self):
        # All NLI scores are 0.1 — below the 0.5 threshold, so no spans.
        with patch("athena_verify.core.batch_compute_entailment", return_value=[0.1, 0.1, 0.1, 0.1]):
            result = verify(
                question="What color is the sky?",
                answer="The sky appears blue. Photosynthesis happens in plants.",
                context=[CHUNK_0, CHUNK_1],
            )
        for sentence in result.sentences:
            assert sentence.supporting_spans == []
