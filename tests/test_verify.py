"""Tests for the core verify() function."""

from __future__ import annotations

import pytest

from athena_verify import verify
from athena_verify.models import Chunk, VerificationResult


class TestVerifyBasic:
    """Basic verification tests with mock-friendly inputs."""

    def test_verify_returns_result(self):
        """verify() returns a VerificationResult for any input."""
        result = verify(
            question="What color is the sky?",
            answer="The sky is blue.",
            context=["The sky appears blue during the day due to Rayleigh scattering."],
        )
        assert isinstance(result, VerificationResult)
        assert result.question == "What color is the sky?"
        assert result.answer == "The sky is blue."
        assert len(result.sentences) == 1

    def test_verify_empty_answer(self):
        """verify() handles empty answer gracefully."""
        result = verify(
            question="What?",
            answer="",
            context=["Some context"],
        )
        assert isinstance(result, VerificationResult)
        assert result.trust_score == 0.0
        assert result.verification_passed is False

    def test_verify_empty_context(self):
        """verify() handles empty context gracefully."""
        result = verify(
            question="What?",
            answer="Some answer.",
            context=[],
        )
        assert isinstance(result, VerificationResult)

    def test_verify_string_context(self):
        """verify() accepts list[str] as context."""
        result = verify(
            question="What is Python?",
            answer="Python is a programming language.",
            context=["Python is a high-level programming language."],
        )
        assert isinstance(result, VerificationResult)

    def test_verify_chunk_context(self):
        """verify() accepts list[Chunk] as context."""
        result = verify(
            question="What is Python?",
            answer="Python is a programming language.",
            context=[Chunk(content="Python is a high-level programming language.")],
        )
        assert isinstance(result, VerificationResult)

    def test_verify_dict_context(self):
        """verify() accepts list[dict] as context."""
        result = verify(
            question="What is Python?",
            answer="Python is a programming language.",
            context=[{"content": "Python is a high-level programming language."}],
        )
        assert isinstance(result, VerificationResult)

    def test_verify_multiple_sentences(self):
        """verify() splits answer into multiple sentences."""
        result = verify(
            question="Tell me about Python.",
            answer="Python is a programming language. It was created by Guido van Rossum. Python is dynamically typed.",
            context=[
                "Python is a high-level programming language.",
                "Python was created by Guido van Rossum and first released in 1991.",
                "Python uses dynamic typing.",
            ],
        )
        assert len(result.sentences) == 3

    def test_verify_has_trust_score(self):
        """Result has a trust score between 0 and 1."""
        result = verify(
            question="What?",
            answer="Some answer.",
            context=["Some context"],
        )
        assert 0.0 <= result.trust_score <= 1.0

    def test_verify_has_metadata(self):
        """Result includes metadata."""
        result = verify(
            question="What?",
            answer="Some answer.",
            context=["Some context"],
        )
        assert "latency_ms" in result.metadata
        assert "num_chunks" in result.metadata
        assert "num_sentences" in result.metadata


class TestVerifySupported:
    """Test that well-supported answers get high trust scores."""

    def test_supported_answer(self):
        """Answer fully grounded in context should be SUPPORTED."""
        result = verify(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            context=["Paris is the capital city of France, located in the north-central part of the country."],
        )
        assert result.verification_passed is True
        assert result.trust_score > 0.5

    def test_unsupported_answer(self):
        """Answer not grounded in context should be flagged."""
        result = verify(
            question="What is the capital of France?",
            answer="The capital of France is London.",
            context=["Paris is the capital city of France, located in the north-central part of the country."],
        )
        # London is contradicted by the context
        assert len(result.unsupported) > 0 or result.trust_score < 0.8


class TestVerifyEdgeCases:
    """Edge case tests."""

    def test_single_word_answer(self):
        """verify() handles single-word answers."""
        result = verify(
            question="What is 2+2?",
            answer="Four.",
            context=["2 + 2 equals 4."],
        )
        assert isinstance(result, VerificationResult)

    def test_very_long_context(self):
        """verify() handles very long context."""
        long_context = " ".join(["Python is a programming language."] * 100)
        result = verify(
            question="What is Python?",
            answer="Python is a programming language.",
            context=[long_context],
        )
        assert isinstance(result, VerificationResult)

    def test_multiple_context_chunks(self):
        """verify() handles multiple context chunks."""
        result = verify(
            question="What are Python's features?",
            answer="Python supports multiple programming paradigms and has dynamic typing.",
            context=[
                "Python supports multiple programming paradigms including object-oriented and functional.",
                "Python has dynamic typing and automatic memory management.",
            ],
        )
        assert isinstance(result, VerificationResult)
        assert result.metadata["num_chunks"] == 2
