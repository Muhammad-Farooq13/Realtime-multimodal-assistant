"""Timeout Handling and Graceful Degradation for the pipeline.

Philosophy
==========
Each pipeline stage has TWO safeguards:

1. **Hard timeout** — ``async_timeout`` raises ``asyncio.TimeoutError`` after
   the stage's allocated budget.
2. **Graceful fallback** — instead of propagating the error, the stage calls
   a pre-defined fallback function and emits a ``DEGRADED`` response.

Degradation tiers
=================
| Stage exceeded   | Fallback strategy                             |
|-----------------|-----------------------------------------------|
| STT > 600ms     | Switch to tiny Whisper model for this turn    |
| LLM > 1 200ms   | Return pre-canned "processing..." response    |
| TTS > 400ms     | Return text-only (skip audio synthesis)       |
| Total > 3 500ms | Return error with Retry-After header          |

The ``with_timeout`` decorator / context manager wraps any async function
and applies these strategies transparently.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar, Awaitable, Any

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class DegradationLevel(str, Enum):
    NONE = "none"           # All stages within budget
    MINOR = "minor"         # One non-critical stage exceeded budget
    MODERATE = "moderate"   # Critical stage (STT/LLM) used fallback
    SEVERE = "severe"       # Multiple stages degraded
    FAILED = "failed"       # Could not produce any useful response


@dataclass
class TimeoutResult:
    """Wraps a stage result, indicating whether it used a fallback."""

    value: Any
    timed_out: bool = False
    degradation_level: DegradationLevel = DegradationLevel.NONE
    fallback_used: str = ""


class StageTimeoutError(Exception):
    """Raised when a stage exceeds its budget and no fallback is configured."""

    def __init__(self, stage: str, budget_ms: int) -> None:
        self.stage = stage
        self.budget_ms = budget_ms
        super().__init__(f"Stage '{stage}' exceeded {budget_ms}ms budget with no fallback.")


def with_timeout(
    budget_s: float,
    stage: str,
    fallback: Callable[..., Awaitable[Any]] | None = None,
    degradation_level: DegradationLevel = DegradationLevel.MODERATE,
) -> Callable:
    """Decorator that applies a timeout to an async function.

    If the function exceeds ``budget_s`` seconds:
    - If ``fallback`` is provided: calls fallback(*args, **kwargs) and returns
      the result wrapped in a ``TimeoutResult(timed_out=True)``.
    - Otherwise: raises ``StageTimeoutError``.

    Args:
        budget_s:           Timeout in seconds.
        stage:              Stage name for logging.
        fallback:           Async callable with the same signature as the
                            decorated function, providing a degraded response.
        degradation_level:  Reported severity when fallback is invoked.

    Usage::

        @with_timeout(budget_s=0.6, stage="speech_to_text",
                      fallback=stt_tiny_model.transcribe)
        async def transcribe_audio(audio: bytes) -> str:
            return await stt_large_model.transcribe(audio)
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[TimeoutResult]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> TimeoutResult:
            try:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=budget_s)
                return TimeoutResult(value=result)
            except asyncio.TimeoutError:
                logger.warning(
                    "stage_timeout",
                    stage=stage,
                    budget_s=budget_s,
                    has_fallback=fallback is not None,
                )
                if fallback is not None:
                    try:
                        fallback_result = await fallback(*args, **kwargs)
                        return TimeoutResult(
                            value=fallback_result,
                            timed_out=True,
                            degradation_level=degradation_level,
                            fallback_used=getattr(fallback, "__name__", "unknown"),
                        )
                    except Exception as fb_exc:
                        logger.error(
                            "fallback_failed",
                            stage=stage,
                            error=str(fb_exc),
                        )
                        raise StageTimeoutError(stage, int(budget_s * 1000)) from fb_exc
                raise StageTimeoutError(stage, int(budget_s * 1000))

        return wrapper

    return decorator


async def run_with_timeout(
    coro: Awaitable[T],
    budget_s: float,
    stage: str,
    fallback_value: T | None = None,
) -> TimeoutResult:
    """Functional form of timeout handling — no decorator needed.

    Args:
        coro:           The coroutine to run.
        budget_s:       Timeout in seconds.
        stage:          Stage name for logging.
        fallback_value: Value to return on timeout. If ``None`` and timeout
                        occurs, raises ``StageTimeoutError``.

    Returns:
        ``TimeoutResult`` wrapping the real or fallback value.

    Example::

        result = await run_with_timeout(
            stt_model.transcribe(audio),
            budget_s=settings.stt_timeout_s,
            stage="speech_to_text",
            fallback_value=TranscriptionResult(text="", confidence=0.0),
        )
        if result.timed_out:
            metrics.increment("stt_timeout_total")
    """
    try:
        value = await asyncio.wait_for(coro, timeout=budget_s)
        return TimeoutResult(value=value)
    except asyncio.TimeoutError:
        logger.warning("stage_timeout_functional", stage=stage, budget_s=budget_s)
        if fallback_value is not None:
            return TimeoutResult(
                value=fallback_value,
                timed_out=True,
                degradation_level=DegradationLevel.MODERATE,
                fallback_used="static_fallback",
            )
        raise StageTimeoutError(stage, int(budget_s * 1000))


def track_degradation(results: list[TimeoutResult]) -> DegradationLevel:
    """Compute overall degradation level from a list of stage results.

    Rules:
    - All OK            → NONE
    - 1 timed out (non-critical) → MINOR
    - Any critical timed out     → MODERATE
    - ≥3 timed out             → SEVERE
    """
    timed_out = [r for r in results if r.timed_out]
    if not timed_out:
        return DegradationLevel.NONE
    if len(timed_out) >= 3:
        return DegradationLevel.SEVERE
    worst = max(r.degradation_level for r in timed_out)
    return worst
