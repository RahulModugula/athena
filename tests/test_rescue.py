"""Tests for the grounding-rescue path: containment, numeric gate, and the
contradiction-vetoed rescue that recovers faithful paraphrases NLI scores low.
"""

from __future__ import annotations

from athena_verify.calibration import (
    RESCUE_TRUST,
    apply_grounding_rescue,
    classify_support,
)
from athena_verify.overlap import containment_score, numeric_consistency


class TestContainment:
    def test_full_containment(self):
        score = containment_score(
            "reference counting primary mechanism",
            "Python memory management uses reference counting as the primary mechanism.",
        )
        assert score == 1.0

    def test_partial_containment(self):
        score = containment_score(
            "olive oil drizzled before baking",
            "Ingredients: pizza dough, tomato sauce, mozzarella, olive oil.",
        )
        assert 0.0 < score < 0.6

    def test_stopwords_ignored(self):
        # Only function words overlap -> no grounding signal.
        assert containment_score("the and of is", "the cat and the dog") == 0.0

    def test_empty_sentence(self):
        assert containment_score("the of", "anything here") == 0.0


class TestNumericConsistency:
    def test_no_numbers_is_ok(self):
        assert numeric_consistency("the cap applies broadly", "context with no figures")

    def test_matching_number(self):
        assert numeric_consistency("the cap is 2 million", "indemnification cap of 2 million")

    def test_comma_insensitive(self):
        assert numeric_consistency("about 1200 SEK", "approximately SEK 1,200 per tonne")

    def test_substituted_number_fails(self):
        assert not numeric_consistency("the cap is 5 million", "the cap is 2 million")


class TestGroundingRescue:
    def _neutral_paraphrase(self, **over):
        kwargs = dict(
            entailment=0.10, contradiction=0.05, containment=0.9, numeric_ok=True
        )
        kwargs.update(over)
        return apply_grounding_rescue(0.2, **kwargs)

    def test_rescues_neutral_grounded_paraphrase(self):
        trust = self._neutral_paraphrase()
        assert trust >= RESCUE_TRUST
        assert classify_support(trust) in ("SUPPORTED", "PARTIAL")

    def test_contradiction_blocks_rescue(self):
        # A real contradiction (e.g. subtle reversal) must never be rescued.
        trust = self._neutral_paraphrase(contradiction=0.9)
        assert trust == 0.2

    def test_numeric_mismatch_blocks_rescue(self):
        # Number substitution: lexically grounded but a figure is wrong.
        trust = self._neutral_paraphrase(numeric_ok=False)
        assert trust == 0.2

    def test_low_containment_not_rescued(self):
        trust = self._neutral_paraphrase(containment=0.2)
        assert trust == 0.2

    def test_rescue_never_lowers_trust(self):
        # Already-high trust is left untouched.
        assert apply_grounding_rescue(
            0.9, entailment=0.85, contradiction=0.0, containment=1.0, numeric_ok=True
        ) == 0.9
