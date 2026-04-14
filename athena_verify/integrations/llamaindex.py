"""LlamaIndex integration for athena-verify.

Provides VerifyingPostprocessor, a LlamaIndex response postprocessor
that verifies generated answers against retrieved context.

Usage:
    from athena_verify.integrations.llamaindex import VerifyingPostprocessor

    postprocessor = VerifyingPostprocessor()
    engine = index.as_query_engine(
        response_postprocessors=[postprocessor]
    )
    response = engine.query("What is the indemnification cap?")
    # response.metadata contains verification results
"""

from __future__ import annotations

from typing import Any, Optional

from athena_verify import verify
from athena_verify.models import VerificationResult


class VerifyingPostprocessor:
    """LlamaIndex response postprocessor that verifies answers.

    When added to a LlamaIndex query engine's response_postprocessors,
    this intercepts the generated response and verifies each sentence
    against the retrieved context nodes.

    The verification result is attached to the response metadata
    under the key "athena_verification".

    Args:
        trust_threshold: Minimum trust score for verification to pass.
        nli_model: Cross-encoder model for NLI scoring.
        flag_unsupported: If True, append warning to response text
            when unsupported claims are found.
    """

    def __init__(
        self,
        trust_threshold: float = 0.70,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        flag_unsupported: bool = True,
    ):
        self.trust_threshold = trust_threshold
        self.nli_model = nli_model
        self.flag_unsupported = flag_unsupported
        self._last_verification: VerificationResult | None = None

    @property
    def last_verification(self) -> VerificationResult | None:
        """Get the verification result from the last query."""
        return self._last_verification

    def postprocess_nodes(
        self,
        nodes: list[Any],
        query: Optional[Any] = None,
        response: Optional[Any] = None,
    ) -> list[Any]:
        """Postprocess retrieved nodes (before synthesis).

        This is a no-op — verification happens after synthesis.

        Args:
            nodes: Retrieved nodes.
            query: The query.
            response: The response (not yet generated at this stage).

        Returns:
            Unchanged nodes.
        """
        return nodes

    def process_response(
        self,
        response: Any,
        query: str | None = None,
        context_nodes: list[Any] | None = None,
    ) -> Any:
        """Verify a LlamaIndex response against context nodes.

        Call this after the query engine generates a response.

        Args:
            response: LlamaIndex response object.
            query: The original query string.
            context_nodes: Retrieved context nodes (if available).

        Returns:
            The response with verification metadata attached.
        """
        # Extract answer text
        answer_text = str(response) if response else ""
        question = query or ""

        # Extract context from response source nodes or provided nodes
        context_texts: list[str] = []
        source_nodes = getattr(response, "source_nodes", None) or context_nodes or []

        for node in source_nodes:
            if hasattr(node, "text"):
                context_texts.append(node.text)
            elif hasattr(node, "node") and hasattr(node.node, "text"):
                context_texts.append(node.node.text)
            elif isinstance(node, str):
                context_texts.append(node)

        if not context_texts or not answer_text:
            return response

        # Run verification
        result = verify(
            question=question,
            answer=answer_text,
            context=context_texts,
            nli_model=self.nli_model,
            trust_threshold=self.trust_threshold,
        )
        self._last_verification = result

        # Attach verification to response metadata
        if hasattr(response, "metadata"):
            if response.metadata is None:
                response.metadata = {}
            response.metadata["athena_verification"] = result.model_dump()

        # Optionally flag unsupported claims in the response text
        if self.flag_unsupported and result.unsupported:
            unsupported_texts = [s.text for s in result.unsupported]
            warning = (
                f"\n\n⚠️ Verification: {len(result.unsupported)} unsupported claim(s) detected. "
                f"Trust score: {result.trust_score:.2f}"
            )
            if hasattr(response, "response"):
                response.response = response.response + warning
            elif hasattr(response, "text"):
                response.text = response.text + warning

        return response
