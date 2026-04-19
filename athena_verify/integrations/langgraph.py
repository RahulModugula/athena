"""LangGraph integration for athena-verify.

Provides a node factory for building circuit-breaker patterns in LangGraph agents.
"""

from __future__ import annotations

from typing import Any, Callable

from athena_verify.core import verify_step
from athena_verify.models import StepResult


def make_verify_node(
    claim_key: str = "claim",
    evidence_key: str = "evidence",
    threshold: float = 0.7,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    result_key: str = "verify_result",
    halt_key: str = "halt",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a LangGraph node that verifies a claim against evidence.

    The returned function is a LangGraph-compatible node that reads
    claim_key and evidence_key from state, verifies the claim, and
    returns an updated state dict with the result and a halt flag.

    Args:
        claim_key: State key containing the claim to verify.
        evidence_key: State key containing evidence (str or list[str]).
        threshold: Minimum trust score for claim to pass.
        nli_model: Cross-encoder model name for NLI scoring.
        result_key: State key to store the StepResult.
        halt_key: State key to store the halt flag (True = halt, False = continue).

    Returns:
        A callable that accepts state dict and returns updated state dict.

    Example:
        >>> from langgraph.graph import StateGraph
        >>> verify_node = make_verify_node()
        >>> graph = StateGraph(dict)
        >>> graph.add_node("verify", verify_node)
        >>> graph.add_edge("generate", "verify")
        >>> graph.add_conditional_edges("verify", lambda s: "end" if s["halt"] else "next")
    """

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        result: StepResult = verify_step(
            claim=state[claim_key],
            evidence=state[evidence_key],
            threshold=threshold,
            nli_model=nli_model,
        )
        return {**state, result_key: result, halt_key: not result.passed}

    return _node
