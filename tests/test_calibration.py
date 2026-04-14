"""Tests for the calibration module."""

from __future__ import annotations

from athena_verify.calibration import classify_support, compute_overall_trust, compute_trust_score
from athena_verify.models import SentenceScore


class TestComputeTrustScore:
    """Tests for trust score computation."""

    def test_high_scores(self):
        """High NLI and overlap should give high trust."""
        trust = compute_trust_score(nli_score=0.95, lexical_overlap=0.9)
        assert trust > 0.8

    def test_low_scores(self):
        """Low NLI and overlap should give low trust."""
        trust = compute_trust_score(nli_score=0.1, lexical_overlap=0.1)
        assert trust < 0.3

    def test_with_llm_judge(self):
        """LLM judge score should be incorporated."""
        trust_no_judge = compute_trust_score(nli_score=0.5, lexical_overlap=0.5)
        trust_with_judge = compute_trust_score(nli_score=0.5, lexical_overlap=0.5, llm_judge_score=0.9)
        # Higher judge score should increase trust
        assert trust_with_judge > trust_no_judge

    def test_clamped_to_range(self):
        """Trust score should be clamped to [0.0, 1.0]."""
        trust = compute_trust_score(nli_score=1.0, lexical_overlap=1.0, llm_judge_score=1.0)
        assert trust <= 1.0

    def test_zero_scores(self):
        """Zero scores should give zero trust."""
        trust = compute_trust_score(nli_score=0.0, lexical_overlap=0.0)
        assert trust == 0.0


class TestClassifySupport:
    """Tests for support status classification."""

    def test_supported(self):
        """High trust should be SUPPORTED."""
        assert classify_support(0.85) == "SUPPORTED"

    def test_partial(self):
        """Medium trust should be PARTIAL."""
        assert classify_support(0.6) == "PARTIAL"

    def test_unsupported(self):
        """Low trust should be UNSUPPORTED."""
        assert classify_support(0.35) == "UNSUPPORTED"

    def test_contradicted(self):
        """Very low trust should be CONTRADICTED."""
        assert classify_support(0.1) == "CONTRADICTED"


class TestComputeOverallTrust:
    """Tests for overall trust computation."""

    def _make_sentence(self, trust: float, status: str) -> SentenceScore:
        return SentenceScore(
            text="test",
            index=0,
            nli_score=trust,
            lexical_overlap=trust,
            trust_score=trust,
            support_status=status,
        )

    def test_all_supported(self):
        """All supported sentences should pass verification."""
        sentences = [
            self._make_sentence(0.9, "SUPPORTED"),
            self._make_sentence(0.85, "SUPPORTED"),
        ]
        trust, passed = compute_overall_trust(sentences)
        assert passed is True
        assert trust > 0.7

    def test_all_unsupported(self):
        """All unsupported sentences should fail verification."""
        sentences = [
            self._make_sentence(0.1, "CONTRADICTED"),
            self._make_sentence(0.2, "UNSUPPORTED"),
        ]
        trust, passed = compute_overall_trust(sentences)
        assert passed is False

    def test_empty_sentences(self):
        """Empty sentence list should return (0.0, False)."""
        trust, passed = compute_overall_trust([])
        assert trust == 0.0
        assert passed is False
