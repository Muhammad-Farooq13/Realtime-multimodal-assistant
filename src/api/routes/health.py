"""Health check endpoints — liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.pipeline.circuit_breaker import CircuitBreakerRegistry

router = APIRouter()


@router.get("/health", summary="Liveness probe")
async def liveness():
    """Returns 200 if the process is alive."""
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness(request: Request):
    """Returns 200 only when all models are loaded and services are ready."""
    ready = getattr(request.app.state, "ready", False)
    if not ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Models still loading"},
        )

    registry = CircuitBreakerRegistry.instance()
    breaker_statuses = registry.all_statuses()
    open_breakers = [b for b in breaker_statuses if b["state"] == "open"]

    return {
        "status": "ready",
        "circuit_breakers": breaker_statuses,
        "degraded": len(open_breakers) > 0,
        "open_circuits": [b["service"] for b in open_breakers],
    }
