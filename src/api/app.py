"""FastAPI application factory with lifespan management.

Startup sequence
================
1. Configure structlog (JSON or console based on LOG_FORMAT)
2. Pre-load Whisper models into RAM (avoids cold-start on first request)
3. Initialise Prometheus registry
4. Mount all routers
5. Add CORS, request-ID, and timing middleware

Health probes (Kubernetes-ready)
=================================
- GET /health         → liveness  (always 200 if process is alive)
- GET /health/ready   → readiness (200 only when models are fully loaded)
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.audio import router as audio_router
from src.api.routes.health import router as health_router
from src.api.routes.stream import router as stream_router
from src.api.websocket.handler import router as ws_router
from src.audio.transcription import TranscriptionService
from src.config.settings import get_settings
from src.monitoring.metrics import PipelineMetrics
from src.monitoring.tracing import configure_tracing
from src.pipeline.orchestrator import PipelineOrchestrator

settings = get_settings()

# ── Logging configuration ──────────────────────────────────────────────────────


def _configure_logging() -> None:
    import logging
    import sys

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    )


# ── Application state ──────────────────────────────────────────────────────────


class AppState:
    """Shared application state attached to ``app.state``."""

    orchestrator: PipelineOrchestrator
    stt_service: TranscriptionService
    ready: bool = False


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    _configure_logging()
    log = structlog.get_logger(__name__)
    log.info("startup_begin", version="1.0.0")

    # Boot services
    stt = TranscriptionService()
    await stt.load_models()

    orchestrator = PipelineOrchestrator()

    # Attach to app state for dependency injection
    app.state.stt_service = stt
    app.state.orchestrator = orchestrator
    app.state.ready = True

    if settings.tracing_enabled:
        configure_tracing(app)

    PipelineMetrics.instance()  # initialise registry

    log.info("startup_complete", whisper_model=settings.whisper_model_size)
    yield

    # Shutdown
    app.state.ready = False
    log.info("shutdown_complete")


# ── Application factory ────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    application = FastAPI(
        title="Real-Time Multimodal Assistant",
        description=(
            "Voice + vision streaming pipeline with explicit latency budget enforcement, "
            "circuit breakers, and graceful degradation."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request-ID + latency middleware ───────────────────────────────────────
    @application.middleware("http")
    async def add_request_id(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    application.include_router(health_router)
    application.include_router(audio_router, prefix="/api/v1/audio", tags=["Audio"])
    application.include_router(stream_router, prefix="/api/v1/stream", tags=["Stream"])
    application.include_router(ws_router, tags=["WebSocket"])

    # ── Prometheus scrape endpoint ─────────────────────────────────────────
    if settings.metrics_enabled:
        from prometheus_client import make_asgi_app

        metrics_app = make_asgi_app()
        application.mount("/metrics", metrics_app)

    return application


app = create_app()
