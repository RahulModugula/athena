"""Tests for the sentence parser module."""

from __future__ import annotations

from athena_verify.parser import split_sentences


class TestSplitSentences:
    """Tests for sentence splitting."""

    def test_single_sentence(self):
        """Single sentence should return one element."""
        result = split_sentences("This is a sentence.")
        assert len(result) == 1
        assert result[0] == "This is a sentence."

    def test_two_sentences(self):
        """Two sentences should be split correctly."""
        result = split_sentences("First sentence. Second sentence.")
        assert len(result) == 2

    def test_question_marks(self):
        """Question marks should split sentences."""
        result = split_sentences("What is this? This is a test.")
        assert len(result) == 2

    def test_exclamation_marks(self):
        """Exclamation marks should split sentences."""
        result = split_sentences("Hello world! How are you?")
        assert len(result) == 2

    def test_empty_string(self):
        """Empty string should return empty list."""
        result = split_sentences("")
        assert result == []

    def test_whitespace_only(self):
        """Whitespace-only string should return empty list."""
        result = split_sentences("   ")
        assert result == []

    def test_no_punctuation(self):
        """Text without sentence-ending punctuation should return as-is."""
        result = split_sentences("no punctuation here")
        assert len(result) == 1
        assert result[0] == "no punctuation here"

    def test_multiple_spaces(self):
        """Extra whitespace should be handled."""
        result = split_sentences("First.   Second.")
        assert len(result) == 2
