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

    def test_re_retrieve_success_on_second_attempt(self):
        mock_llm = MagicMock()
        mock_llm.predict.side_effect = [
            "The cap is $5M.",  # First attempt (bad answer)
            "The cap is $1M.",  # Second attempt (good answer after re-retrieve)
        ]

        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "The agreement sets a cap of $1M."
        mock_retriever.get_relevant_documents.return_value = [mock_doc]

        failed_result = _make_verification_result(passed=False, trust=0.3)
        passed_result = _make_verification_result(passed=True, trust=0.85)

        vllm = VerifyingLLM(
            mock_llm,
            retriever=mock_retriever,
            max_retries=2,
            on_unsupported="re-retrieve",
        )

        with patch(
            "athena_verify.integrations.langchain.verify",
            side_effect=[failed_result, passed_result],
        ):
            result = vllm.predict(
                "What is the cap?",
                context=["Some initial context."],
            )

        assert result == "The cap is $1M."
        assert vllm.retry_count == 1
        assert vllm.last_verification.verification_passed is True
        mock_retriever.get_relevant_documents.assert_called()

    def test_re_retrieve_exhausts_max_retries(self):
        mock_llm = MagicMock()
        mock_llm.predict.side_effect = [
            "The cap is $5M.",
            "The cap is $5M.",
            "The cap is $5M.",
        ]

        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Extra context."
        mock_retriever.get_relevant_documents.return_value = [mock_doc]

        failed_result = _make_verification_result(passed=False, trust=0.3)

        vllm = VerifyingLLM(
            mock_llm,
            retriever=mock_retriever,
            max_retries=2,
            on_unsupported="re-retrieve",
        )

        with patch(
            "athena_verify.integrations.langchain.verify",
            return_value=failed_result,
        ):
            result = vllm.predict(
                "What is the cap?",
                context=["Initial context."],
            )

        assert result == "The cap is $5M."
        assert vllm.retry_count == 2
        mock_retriever.get_relevant_documents.assert_called()

    def test_re_retrieve_without_retriever_falls_back(self):
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "The cap is $5M."

        failed_result = _make_verification_result(passed=False, trust=0.3)

        vllm = VerifyingLLM(
            mock_llm,
            retriever=None,
            on_unsupported="re-retrieve",
        )

        with patch(
            "athena_verify.integrations.langchain.verify",
            return_value=failed_result,
        ):
            result = vllm.predict(
                "What is the cap?",
                context=["Initial context."],
            )

        assert result == "The cap is $5M."
        assert vllm.retry_count == 0

    def test_re_retrieve_early_exit_on_pass(self):
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "The cap is $1M."

        mock_retriever = MagicMock()

        passed_result = _make_verification_result(passed=True, trust=0.85)

        vllm = VerifyingLLM(
            mock_llm,
            retriever=mock_retriever,
            max_retries=2,
            on_unsupported="re-retrieve",
        )

        with patch(
            "athena_verify.integrations.langchain.verify",
            return_value=passed_result,
        ):
            result = vllm.predict(
                "What is the cap?",
                context=["Initial context."],
            )

        assert result == "The cap is $1M."
        assert vllm.retry_count == 0
        mock_retriever.get_relevant_documents.assert_not_called()


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
