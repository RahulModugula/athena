"""Tests for the lexical overlap module."""

from __future__ import annotations

from athena_verify.overlap import best_overlap_score, token_f1


class TestTokenF1:
    """Tests for token-level F1 computation."""

    def test_identical_strings(self):
        """Identical strings should have F1 = 1.0."""
        assert token_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0

    def test_completely_different(self):
        """Completely different strings should have F1 = 0.0."""
        assert token_f1("cat dog bird", "fish whale shark") == 0.0

    def test_partial_overlap(self):
        """Partially overlapping strings should have 0 < F1 < 1."""
        score = token_f1("the cat sat on the mat", "the cat sat on the rug")
        assert 0.0 < score < 1.0

    def test_empty_string(self):
        """Empty strings should return 0.0."""
        assert token_f1("", "some text") == 0.0
        assert token_f1("some text", "") == 0.0
        assert token_f1("", "") == 0.0

    def test_case_insensitive(self):
        """Comparison should be case-insensitive."""
        assert token_f1("The Cat Sat", "the cat sat") == 1.0

    def test_single_word(self):
        """Single word overlap should work."""
        score = token_f1("cat", "cat")
        assert score == 1.0

    def test_subset(self):
        """Subset tokens should give partial F1."""
        score = token_f1("cat", "cat dog")
        # precision=1.0, recall=0.5, F1=2*1.0*0.5/1.5=0.667
        assert abs(score - 2 / 3) < 0.01


class TestBestOverlapScore:
    """Tests for best-matching context chunk finding."""

    def test_best_match(self):
        """Should find the best matching chunk."""
        chunks = [
            "The capital of France is London.",
            "The capital of France is Paris.",
            "Python is a programming language.",
        ]
        score, best = best_overlap_score("The capital of France is Paris.", chunks)
        assert best == "The capital of France is Paris."
        assert score == 1.0

    def test_no_chunks(self):
        """Empty chunk list should return (0.0, None)."""
        score, best = best_overlap_score("some text", [])
        assert score == 0.0
        assert best is None

    def test_no_match(self):
        """No matching chunks should return low score."""
        chunks = ["fish whale shark", "ocean sea water"]
        score, best = best_overlap_score("cat dog bird", chunks)
        assert score == 0.0
