"""Tests for the optional OpenTelemetry / Langfuse emitters.

The default path (no opentelemetry SDK configured) must never raise and must
fall back to structured logging. When the SDK is installed, a real span is
recorded and carries athena's attributes.
"""

from __future__ import annotations

import pytest

from athena_verify.models import SentenceScore, VerificationResult
from athena_verify.observability import emit_langfuse_trace, emit_otel_span


def _result() -> VerificationResult:
    sentences = [
        SentenceScore(
            text="The cap is $1M.",
            index=0,
            nli_score=0.1,
            lexical_overlap=0.4,
            trust_score=0.2,
            support_status="UNSUPPORTED",
        ),
        SentenceScore(
            text="Signed on 2024-01-15.",
            index=1,
            nli_score=0.9,
            lexical_overlap=0.8,
            trust_score=0.9,
            support_status="SUPPORTED",
        ),
    ]
    return VerificationResult(
        question="What are the terms?",
        answer="The cap is $1M. Signed on 2024-01-15.",
        trust_score=0.55,
        sentences=sentences,
        unsupported=[sentences[0]],
        supported=[sentences[1]],
        verification_passed=False,
        metadata={"nli_model": "test", "latency_ms": 12.3},
    )


def test_langfuse_emit_never_raises() -> None:
    # Pure structured-log emitter; must be a safe no-op that returns None.
    assert emit_langfuse_trace(_result()) is None


def test_otel_emit_falls_back_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no opentelemetry importable, emit_otel_span returns False, no raise."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert emit_otel_span(_result()) is False


def test_otel_emit_records_real_span_when_sdk_present() -> None:
    """When the SDK is installed, a real `athena.verify` span is recorded."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # set_tracer_provider only takes effect once per process; get_tracer_provider
    # is fine to read back. Use the provider directly to stay isolated.
    tracer = provider.get_tracer("athena_verify")
    monkeypatched = trace.get_tracer
    try:
        trace.get_tracer = lambda *a, **k: tracer  # type: ignore[assignment]
        assert emit_otel_span(_result()) is True
    finally:
        trace.get_tracer = monkeypatched  # type: ignore[assignment]

    spans = exporter.get_finished_spans()
    assert any(s.name == "athena.verify" for s in spans)
    span = next(s for s in spans if s.name == "athena.verify")
    assert span.attributes is not None
    assert span.attributes["athena.trust_score"] == pytest.approx(0.55)
    assert span.attributes["athena.num_unsupported"] == 1
