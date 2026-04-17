"""LangChain integration for athena-verify.

Provides VerifyingLLM, a thin wrapper around any LangChain LLM that
automatically verifies generated answers against retrieved context.

Usage:
    from athena_verify.integrations.langchain import VerifyingLLM

    chain = RetrievalQA.from_llm(VerifyingLLM(llm), retriever=retriever)
    result = chain.run("What is the indemnification cap?")
    # result contains both the answer and verification metadata
"""

from __future__ import annotations

from typing import Any

from athena_verify import verify
from athena_verify.models import VerificationResult


class VerifyingLLM:
    """LangChain LLM wrapper that verifies answers against context.

    Wraps any LangChain LLM and automatically runs athena-verify on
    the generated answer using the retrieved context documents.

    This is designed to work with LangChain's RetrievalQA chain.
    The retriever provides context documents, the LLM generates an
    answer, and VerifyingLLM verifies it before returning.

    Args:
        llm: A LangChain LLM instance (e.g., ChatOpenAI, ChatAnthropic).
        trust_threshold: Minimum trust score for verification to pass.
        nli_model: Cross-encoder model for NLI scoring.
        on_unsupported: What to do when verification fails.
            "warn" — log a warning, return the answer anyway.
            "flag" — attach verification metadata to the answer.
            "reject" — return None instead of the answer.
    """

    def __init__(
        self,
        llm: Any,
        trust_threshold: float = 0.70,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        on_unsupported: str = "flag",
    ):
        self.llm = llm
        self.trust_threshold = trust_threshold
        self.nli_model = nli_model
        self.on_unsupported = on_unsupported
        self._last_verification: VerificationResult | None = None

    @property
    def last_verification(self) -> VerificationResult | None:
        """Get the verification result from the last call."""
        return self._last_verification

    def predict(
        self,
        text: str,
        *,
        context: list[str] | None = None,
        question: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate and verify an answer.

        This method is designed to be compatible with LangChain's
        chain interface. For direct usage, pass context explicitly.

        Args:
            text: The prompt or question.
            context: Retrieved context chunks (strings).
            question: The original question (defaults to text).
            **kwargs: Additional arguments passed to the LLM.

        Returns:
            The LLM answer, potentially annotated with verification info.
        """
        if hasattr(self.llm, "predict"):
            answer = self.llm.predict(text, **kwargs)
        else:
            answer = str(self.llm(text))

        if context and len(context) > 0:
            result = verify(
                question=question or text,
                answer=answer,
                context=context,
                nli_model=self.nli_model,
                trust_threshold=self.trust_threshold,
            )
            self._last_verification = result

            if not result.verification_passed:
                if self.on_unsupported == "reject":
                    return ""
                elif self.on_unsupported == "flag":
                    unsupp = [s.text for s in result.unsupported]
                    n = len(result.unsupported)
                    flag = f"\n\n⚠️ Verification warning: {n} unsupported claims."
                    if unsupp:
                        flag += f"\nUnsupported sentences: {unsupp}"
                    return answer + flag  # type: ignore[no-any-return]

        return answer  # type: ignore[no-any-return]

    def predict_messages(
        self,
        messages: list[Any],
        *,
        context: list[str] | None = None,
        question: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate and verify using message-style interface.

        Args:
            messages: List of LangChain message objects.
            context: Retrieved context chunks.
            question: The original question.
            **kwargs: Additional arguments.

        Returns:
            The LLM response.
        """
        response = self.llm.predict_messages(messages, **kwargs)

        if context and len(context) > 0:
            answer_text = response.content if hasattr(response, "content") else str(response)
            result = verify(
                question=question or str(messages[-1]),
                answer=answer_text,
                context=context,
                nli_model=self.nli_model,
                trust_threshold=self.trust_threshold,
            )
            self._last_verification = result

        return response

    # Pass through attribute access to the underlying LLM
    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)
