"""Tests for the NLI module — model aliases and score computation.

Uses mock CrossEncoder to avoid requiring sentence-transformers in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from athena_verify.nli import (
    NLI_MODEL_ALIASES,
    batch_compute_entailment,
    compute_entailment_score,
    resolve_nli_model,
)


class TestResolveNLIModel:
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

    def test_unknown_string_passthrough(self):
        assert resolve_nli_model("some-unknown-model") == "some-unknown-model"

    def test_aliases_has_expected_keys(self):
        for key in ("default", "lightweight", "vectara", "deberta-base"):
            assert key in NLI_MODEL_ALIASES


class _MockArray:
    def __init__(self, data: list[float]):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def sum(self) -> float:
        return sum(self._data)


class MockCrossEncoder:
    def __init__(self, entailment_score: float = 0.9):
        self.entailment_score = entailment_score

    def predict(self, pairs):
        return [_MockArray([self.entailment_score, 0.05, 0.05]) for _ in pairs]


@pytest.fixture
def mock_model_cache():
    with patch("athena_verify.nli._nli_cache", {}) as cache:
        yield cache


@pytest.fixture
def mock_cross_encoder(mock_model_cache):
    model = MockCrossEncoder(entailment_score=0.85)
    mock_model_cache["cross-encoder/nli-deberta-v3-base"] = model
    return model


class TestComputeEntailmentScore:
    def test_basic_score(self, mock_cross_encoder):
        score = compute_entailment_score("The cat sat on the mat.", "A cat was on the mat.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_single_pair_returns_float(self, mock_cross_encoder):
        score = compute_entailment_score("Premise.", "Hypothesis.")
        assert isinstance(score, float)

    def test_with_alias(self, mock_model_cache):
        model = MockCrossEncoder(entailment_score=0.75)
        resolved = resolve_nli_model("default")
        mock_model_cache[resolved] = model
        score = compute_entailment_score("P.", "H.", model_name="default")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestBatchComputeEntailment:
    def test_empty_pairs(self):
        result = batch_compute_entailment([])
        assert result == []

    def test_single_pair(self, mock_cross_encoder):
        result = batch_compute_entailment([("Premise.", "Hypothesis.")])
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_multiple_pairs(self, mock_cross_encoder):
        pairs = [
            ("The sky is blue.", "The sky appears blue."),
            ("Paris is the capital.", "The capital is London."),
            ("Python is a language.", "Python is a programming language."),
        ]
        result = batch_compute_entailment(pairs)
        assert len(result) == 3
        for score in result:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_batch_size_respected(self, mock_model_cache):
        model = MockCrossEncoder(entailment_score=0.8)
        mock_model_cache["cross-encoder/nli-deberta-v3-base"] = model
        pairs = [("P.", f"H{i}.") for i in range(10)]
        result = batch_compute_entailment(pairs, batch_size=3)
        assert len(result) == 10
        for score in result:
            assert isinstance(score, float)

    def test_no_double_predict(self, mock_model_cache):
        mock_model = MagicMock()
        mock_model.predict.return_value = [_MockArray([0.9, 0.05, 0.05])]
        mock_model_cache["cross-encoder/nli-deberta-v3-base"] = mock_model

        batch_compute_entailment([("P.", "H.")], batch_size=32)

        predict_calls = mock_model.predict.call_count
        assert predict_calls == 1, f"predict() called {predict_calls} times, expected 1"

    def test_large_batch_single_call(self, mock_model_cache):
        mock_model = MagicMock()
        mock_model.predict.return_value = [_MockArray([0.9, 0.05, 0.05])] * 5
        mock_model_cache["cross-encoder/nli-deberta-v3-base"] = mock_model

        pairs = [("P.", f"H{i}.") for i in range(5)]
        batch_compute_entailment(pairs, batch_size=32)

        assert mock_model.predict.call_count == 1

    def test_large_batch_split_calls(self, mock_model_cache):
        mock_model = MagicMock()
        mock_model.predict.return_value = [_MockArray([0.9, 0.05, 0.05])] * 3
        mock_model_cache["cross-encoder/nli-deberta-v3-base"] = mock_model

        pairs = [("P.", f"H{i}.") for i in range(7)]
        batch_compute_entailment(pairs, batch_size=3)

        assert mock_model.predict.call_count == 3


class TestModelCaching:
    def test_model_cached_after_first_load(self, mock_model_cache):
        mock_model = MockCrossEncoder()
        mock_model_cache["cross-encoder/nli-deberta-v3-base"] = mock_model

        batch_compute_entailment([("P1.", "H1.")])
        batch_compute_entailment([("P2.", "H2.")])

        assert "cross-encoder/nli-deberta-v3-base" in mock_model_cache
