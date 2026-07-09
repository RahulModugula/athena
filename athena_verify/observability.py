"""Optional OpenTelemetry / Langfuse emission for verification results.

Both emitters are opt-in via environment flag (``ATHENA_OTEL_ENABLED=1`` /
``ATHENA_LANGFUSE_ENABLED=1``) and are called from ``verify()`` /
``verify_async()``. They never raise: if the OpenTelemetry SDK is not installed
(or no tracer/exporter is configured), OTel emission falls back to a structured
log line, so enabling the flag is always safe.

To send real spans to Grafana / Datadog / Jaeger, install the extra and
configure an exporter in your app:

    pip install "athena-verify[otel]"

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    trace.set_tracer_provider(TracerProvider())  # + your span processor/exporter

    import os; os.environ["ATHENA_OTEL_ENABLED"] = "1"
    verify(...)  # each call now records an `athena.verify` span
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from athena_verify.models import VerificationResult

logger = structlog.get_logger(__name__)

# OpenTelemetry span/event attribute values must be primitives (or sequences of
# them). We filter to these before setting anything on a span.
_PRIMITIVE = (str, bool, int, float)


def _primitives(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if isinstance(v, _PRIMITIVE)}


def emit_otel_span(result: VerificationResult) -> bool:
    """Record a real OpenTelemetry span for a verification result.

    Returns True if a real span was recorded, False if it fell back to logging
    (SDK missing, no tracer, or an error). Never raises.
    """
    span_data = result.to_otel_span()
    try:
        from opentelemetry import trace
    except ImportError:
        logger.info("otel_span_generated", span=span_data)
        return False

    try:
        tracer = trace.get_tracer("athena_verify")
        with tracer.start_as_current_span(span_data["name"]) as span:
            for key, value in _primitives(span_data["attributes"]).items():
                span.set_attribute(key, value)
            for event in span_data["events"]:
                span.add_event(event["name"], attributes=_primitives(event["attributes"]))
        return True
    except Exception as exc:  # noqa: BLE001 - telemetry must never break verify()
        logger.warning("otel_span_failed", error=str(exc))
        logger.info("otel_span_generated", span=span_data)
        return False


def emit_langfuse_trace(result: VerificationResult) -> None:
    """Emit a Langfuse-compatible trace (best-effort structured log).

    Kept as a structured log line rather than a hard Langfuse client dependency;
    pipe the ``langfuse_trace_generated`` event to the Langfuse ingestion API.
    Never raises.
    """
    logger.info("langfuse_trace_generated", trace=result.to_langfuse_trace())
