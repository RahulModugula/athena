"""Tests for integrations (LangChain and LlamaIndex wrappers).

Uses mocks so the actual frameworks don't need to be installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from athena_verify.integrations.langchain import VerifyingLLM
from athena_verify.integrations.llamaindex import VerifyingPostprocessor
from athena_verify.models import SentenceScore, VerificationResult


def _make_verification_result(passed: bool = True, trust: float = 0.85) -> VerificationResult:
    score = SentenceScore(
        text="The cap is $1M.",
        index=0,
        nli_score=trust,
        lexical_overlap=trust,
        trust_score=trust,
        support_status="SUPPORTED" if passed else "UNSUPPORTED",
    )
    return VerificationResult(
        question="What is the cap?",
        answer="The cap is $1M.",
        trust_score=trust,
        sentences=[score],
        unsupported=[] if passed else [score],
        supported=[score] if passed else [],
        verification_passed=passed,
    )


class TestVerifyingLLM:
    def test_predict_no_context(self):
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "The cap is $1M."

        vllm = VerifyingLLM(mock_llm)
        result = vllm.predict("What is the cap?")

        assert result == "The cap is $1M."
        mock_llm.predict.assert_called_once_with("What is the cap?")

    def test_predict_with_context_passes(self):
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "The cap is $1M."

        mock_result = _make_verification_result(passed=True)

        vllm = VerifyingLLM(mock_llm, on_unsupported="flag")

        with patch("athena_verify.integrations.langchain.verify", return_value=mock_result):
            result = vllm.predict(
                "What is the cap?",
                context=["The agreement sets a cap of $1M."],
            )

        assert "The cap is $1M." in result
        assert vllm.last_verification is not None
        assert vllm.last_verification.verification_passed is True

    def test_predict_flag_unsupported(self):
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "The cap is $5M."

        mock_result = _make_verification_result(passed=False, trust=0.3)

        vllm = VerifyingLLM(mock_llm, on_unsupported="flag")

        with patch("athena_verify.integrations.langchain.verify", return_value=mock_result):
            result = vllm.predict(
                "What is the cap?",
                context=["The agreement sets a cap of $1M."],
            )

        assert "Verification warning" in result

    def test_predict_reject_unsupported(self):
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "The cap is $5M."

        mock_result = _make_verification_result(passed=False, trust=0.3)

        vllm = VerifyingLLM(mock_llm, on_unsupported="reject")

        with patch("athena_verify.integrations.langchain.verify", return_value=mock_result):
            result = vllm.predict(
                "What is the cap?",
                context=["The agreement sets a cap of $1M."],
            )

        assert result == ""

    def test_predict_warn_unsupported(self):
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "The cap is $5M."

        mock_result = _make_verification_result(passed=False, trust=0.3)

        vllm = VerifyingLLM(mock_llm, on_unsupported="warn")

        with patch("athena_verify.integrations.langchain.verify", return_value=mock_result):
            result = vllm.predict(
                "What is the cap?",
                context=["The agreement sets a cap of $1M."],
            )

        assert result == "The cap is $5M."

    def test_getattr_passthrough(self):
        mock_llm = MagicMock()
        mock_llm.custom_attr = "test_value"

        vllm = VerifyingLLM(mock_llm)
        assert vllm.custom_attr == "test_value"

    def test_predict_messages_with_context(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The cap is $1M."
        mock_llm.predict_messages.return_value = mock_response

        mock_result = _make_verification_result(passed=True)

        vllm = VerifyingLLM(mock_llm)

        with patch("athena_verify.integrations.langchain.verify", return_value=mock_result):
            response = vllm.predict_messages(
                [MagicMock()],
                context=["The agreement sets a cap of $1M."],
                question="What is the cap?",
            )

        assert response.content == "The cap is $1M."


class TestVerifyingPostprocessor:
    def test_postprocess_nodes_passthrough(self):
        pp = VerifyingPostprocessor()
        nodes = [MagicMock()]
        result = pp.postprocess_nodes(nodes)
        assert result is nodes

    def test_process_response_with_source_nodes(self):
        pp = VerifyingPostprocessor(flag_unsupported=True)

        mock_node = MagicMock()
        mock_node.text = "The agreement sets a cap of $1M per incident."

        mock_response = MagicMock()
        mock_response.source_nodes = [mock_node]
        mock_response.metadata = {}
        mock_response.__str__ = MagicMock(return_value="The cap is $1M per incident.")

        mock_result = _make_verification_result(passed=True)

        with patch("athena_verify.integrations.llamaindex.verify", return_value=mock_result):
            response = pp.process_response(mock_response, query="What is the cap?")

        assert "athena_verification" in response.metadata
        assert pp.last_verification is not None

    def test_process_response_no_context(self):
        pp = VerifyingPostprocessor()

        mock_response = MagicMock()
        mock_response.source_nodes = None
        mock_response.__str__ = MagicMock(return_value="")

        pp.process_response(mock_response)
        assert pp.last_verification is None

    def test_process_response_flag_unsupported(self):
        pp = VerifyingPostprocessor(flag_unsupported=True)

        mock_node = MagicMock()
        mock_node.text = "The cap is $1M."

        mock_response = MagicMock()
        mock_response.source_nodes = [mock_node]
        mock_response.metadata = {}
        mock_response.response = "The cap is $5M."
        mock_response.__str__ = MagicMock(return_value="The cap is $5M.")

        mock_result = _make_verification_result(passed=False, trust=0.3)

        with patch("athena_verify.integrations.llamaindex.verify", return_value=mock_result):
            response = pp.process_response(mock_response, query="What?")

        assert "Verification" in response.response
