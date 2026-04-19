"""LangChain integration for athena-verify.

Provides VerifyingLLM, a thin wrapper around any LangChain LLM that
automatically verifies generated answers against retrieved context.

Supports an optional self-healing retry/re-retrieve loop (see on_unsupported="re-retrieve").

Usage:
    from athena_verify.integrations.langchain import VerifyingLLM

    chain = RetrievalQA.from_llm(VerifyingLLM(llm), retriever=retriever)
    result = chain.run("What is the indemnification cap?")
    # result contains both the answer and verification metadata

With re-retrieve mode:
    vllm = VerifyingLLM(llm, retriever=retriever, max_retries=2,
                        on_unsupported="re-retrieve")
    result = vllm.predict(text, context=context, question=question)
    # On verification failure, unsupported sentences are re-queried via retriever
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

    Optionally implements a self-healing retry/re-retrieve loop
    (when on_unsupported="re-retrieve" and a retriever is provided).
    On verification failure, unsupported sentences are used as new
    retrieval queries, new chunks are appended to context, and the
    answer is regenerated and re-verified, up to max_retries times.
    See LangChain issue #33191 for the motivation.

    Args:
        llm: A LangChain LLM instance (e.g., ChatOpenAI, ChatAnthropic).
        trust_threshold: Minimum trust score for verification to pass.
        nli_model: Cross-encoder model for NLI scoring.
        on_unsupported: What to do when verification fails.
            "warn" — log a warning, return the answer anyway.
            "flag" — attach verification metadata to the answer.
            "reject" — return None instead of the answer.
            "re-retrieve" — attempt self-healing retry loop (requires retriever).
        retriever: Optional LangChain BaseRetriever instance for re-retrieve mode.
        max_retries: Max retry attempts in re-retrieve mode (default 2).
    """

    def __init__(
        self,
        llm: Any,
        trust_threshold: float = 0.70,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        on_unsupported: str = "flag",
        retriever: Any | None = None,
        max_retries: int = 2,
    ):
        self.llm = llm
        self.trust_threshold = trust_threshold
        self.nli_model = nli_model
        self.on_unsupported = on_unsupported
        self.retriever = retriever
        self.max_retries = max_retries
        self._last_verification: VerificationResult | None = None
        self._retry_count: int = 0

    @property
    def last_verification(self) -> VerificationResult | None:
        """Get the verification result from the last call."""
        return self._last_verification

    @property
    def retry_count(self) -> int:
        """Number of re-retrieve retries performed in the last call."""
        return self._retry_count

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

        When on_unsupported="re-retrieve", implements a self-healing loop:
        on verification failure, unsupported sentences are used as retrieval
        queries to fetch additional context, the answer is regenerated, and
        verification is re-run (up to max_retries times).

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
            context_list = list(context)
            result = None

            for attempt in range(self.max_retries + 1):
                result = verify(
                    question=question or text,
                    answer=answer,
                    context=context_list,
                    nli_model=self.nli_model,
                    trust_threshold=self.trust_threshold,
                )
                self._last_verification = result
                self._retry_count = attempt

                if result.verification_passed:
                    break

                if (
                    self.on_unsupported == "re-retrieve"
                    and self.retriever is not None
                    and attempt < self.max_retries
                ):
                    new_chunks = []
                    for sentence in result.unsupported:
                        docs = self.retriever.get_relevant_documents(sentence.text)
                        new_chunks.extend([doc.page_content for doc in docs])

                    for chunk in new_chunks:
                        if chunk not in context_list:
                            context_list.append(chunk)

                    expanded_prompt = f"Context:\n{chr(10).join(context_list)}\n\n{text}"
                    if hasattr(self.llm, "predict"):
                        answer = self.llm.predict(expanded_prompt, **kwargs)
                    else:
                        answer = str(self.llm(expanded_prompt))
                else:
                    break

            if result and not result.verification_passed:
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
