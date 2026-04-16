"""Optional LLM-as-judge scoring for borderline verification cases.

Uses an LLM (OpenAI or Anthropic) to evaluate whether a sentence is
supported by the provided context. This is an optional signal that
improves accuracy on ambiguous cases where NLI and lexical overlap
disagree.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import structlog

logger = structlog.get_logger()

JUDGE_PROMPT = """You are a verification judge. Given a context and a sentence, determine whether the sentence is supported by the context.

Context:
{context}

Sentence: {sentence}

Question (for context): {question}

Respond with a JSON object:
{{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Be strict: only mark as supported if the sentence can be directly inferred from the context."""


class LLMClient(Protocol):
    """Protocol for LLM clients (OpenAI, Anthropic, etc.)."""

    def complete(self, prompt: str) -> str: ...


_DEFAULT_TIMEOUT = 60


class OpenAIJudge:
    """LLM judge using OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        self._client: object | None = None

    def _get_client(self) -> Any:
        from openai import OpenAI

        if self._client is None:
            kwargs: dict[str, Any] = {"timeout": _DEFAULT_TIMEOUT}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return response.choices[0].message.content or ""


class AnthropicJudge:
    """LLM judge using Anthropic API."""

    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        self._client: object | None = None

    def _get_client(self) -> Any:
        import anthropic

        if self._client is None:
            kwargs: dict[str, Any] = {"timeout": _DEFAULT_TIMEOUT}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""


def judge_sentence(
    sentence: str,
    context: str,
    question: str,
    client: LLMClient | None = None,
) -> tuple[float, str]:
    """Use an LLM to judge whether a sentence is supported by context.

    Args:
        sentence: The sentence to verify.
        context: The context chunks joined into a single string.
        question: The original question (for additional context).
        client: LLM client to use. If None, returns (0.0, "no client").

    Returns:
        Tuple of (score 0.0-1.0, reasoning string).
    """
    if client is None:
        return 0.0, "no_llm_client"

    prompt = JUDGE_PROMPT.format(context=context, sentence=sentence, question=question)

    try:
        response = client.complete(prompt)
        result = json.loads(response)
        supported = result.get("supported", False)
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        score = confidence if supported else (1.0 - confidence)
        return score, reasoning

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("llm_judge_parse_error", error=str(e))
        return 0.5, f"parse_error: {e}"
    except Exception as e:
        logger.warning("llm_judge_error", error=str(e))
        return 0.5, f"error: {e}"


def batch_judge_sentences(
    sentences: list[str],
    context: str,
    question: str,
    client: LLMClient | None = None,
) -> list[tuple[float, str]]:
    """Judge multiple sentences against the same context.

    Args:
        sentences: List of sentences to verify.
        context: The context chunks joined into a single string.
        question: The original question.
        client: LLM client to use.

    Returns:
        List of (score, reasoning) tuples.
    """
    return [judge_sentence(s, context, question, client) for s in sentences]
