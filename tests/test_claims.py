"""Tests for check-worthiness filtering of non-claim sentences."""

from __future__ import annotations

from athena_verify.claims import is_checkworthy, is_question


class TestIsQuestion:
    def test_question_mark(self):
        assert is_question("What is the indemnification cap?")

    def test_statement(self):
        assert not is_question("The cap is $2 million.")


class TestCheckworthy:
    def test_factual_claim_is_checkworthy(self):
        assert is_checkworthy("The liability cap is $2 million per incident.")

    def test_question_not_checkworthy(self):
        assert not is_checkworthy("How are technicians paid?")

    def test_refusal_not_checkworthy(self):
        assert not is_checkworthy("Unable to answer based on the given passages.")

    def test_not_mentioned_not_checkworthy(self):
        assert not is_checkworthy("The passages do not mention the warranty period.")
        assert not is_checkworthy("There is no information about pricing in the context.")

    def test_insufficient_context_not_checkworthy(self):
        assert not is_checkworthy("Insufficient information to determine the answer.")

    def test_empty_not_checkworthy(self):
        assert not is_checkworthy("   ")

    def test_normal_sentence_with_no_keyword_is_checkworthy(self):
        assert is_checkworthy("Automotive technicians can be paid hourly or on commission.")
