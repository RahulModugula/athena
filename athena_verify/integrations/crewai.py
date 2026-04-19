"""CrewAI integration for athena-verify.

Provides a CrewAI tool for verifying claims in agent workflows.
"""

from __future__ import annotations

from typing import Any

from athena_verify.core import verify_step

try:
    from crewai.tools import BaseTool

    _CREWAI_AVAILABLE = True
except ImportError:
    BaseTool = object  # type: ignore[misc,assignment]
    _CREWAI_AVAILABLE = False


class AthenaVerifyTool(BaseTool):
    """CrewAI tool for verifying factual claims against evidence.

    Verify whether a claim is supported by the given evidence.
    Returns a string summary of pass/fail status and trust score.
    """

    name: str = "athena_verify_step"
    description: str = (
        "Verify whether a factual claim is supported by the given evidence. "
        "Returns passed (bool), trust_score (float), and action ('continue' or 'halt')."
    )
    threshold: float = 0.7
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    def _run(self, claim: str, evidence: str) -> str:
        """Verify the claim against evidence.

        Args:
            claim: The factual claim to verify.
            evidence: Evidence to check the claim against.

        Returns:
            String summary of verification result.
        """
        if not _CREWAI_AVAILABLE:
            raise ImportError("Install athena-verify[crewai] to use AthenaVerifyTool.")

        result = verify_step(
            claim=claim,
            evidence=evidence,
            threshold=self.threshold,
            nli_model=self.nli_model,
        )
        return (
            f"passed={result.passed}, trust_score={result.trust_score:.3f}, "
            f"action={result.action}"
        )
