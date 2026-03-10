"""OpenTelemetry distributed tracing configuration.

Traces are exported via OTLP (gRPC) to any compatible backend:
- Jaeger  (docker run jaegertracing/all-in-one)
- Grafana Tempo
- Honeycomb / Datadog / cloud providers

Each pipeline turn creates a root span with child spans per stage.
Trace IDs are correlated with structured log entries via ``structlog``
context vars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = structlog.get_logger(__name__)


def configure_tracing(app: "FastAPI") -> None:
    """Attach OpenTelemetry instrumentation to a FastAPI application.

    Called during app lifespan startup when ``TRACING_ENABLED=true``.
    No-ops silently if opentelemetry packages are missing.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from src.config.settings import get_settings
        settings = get_settings()

        resource = Resource.create(
            {
                "service.name": "multimodal-assistant",
                "service.version": "1.0.0",
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("tracing_configured", endpoint=settings.otlp_endpoint)

    except ImportError:
        logger.warning("opentelemetry_not_installed", note="Tracing disabled")
    except Exception as exc:
        logger.error("tracing_setup_failed", error=str(exc))
