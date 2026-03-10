"""Circuit Breaker — prevents cascading failures in the multimodal pipeline.

State machine
=============

              ┌──── failure_count >= threshold ────┐
              │                                     ▼
          CLOSED                                  OPEN
              ▲                                     │
              │                          recovery_time elapsed
              │                                     ▼
   success_count >= threshold ──────────         HALF-OPEN
                                 failure            │
                                   │                │ success
                                   └──── reopen ────┘

CLOSED   → Normal operation. Failures are counted.
OPEN     → Fast-fail: all calls raise ``CircuitOpenError`` immediately.
HALF-OPEN → A single probe call is allowed through to test recovery.

Each service (STT, LLM, TTS) gets its own ``CircuitBreaker`` instance so
that a slow TTS provider cannot affect LLM availability.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Callable, TypeVar, Awaitable, Any

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""

    def __init__(self, service: str, retry_after: float) -> None:
        self.service = service
        self.retry_after = retry_after
        super().__init__(
            f"Circuit OPEN for '{service}'. Retry after {retry_after:.1f}s."
        )


class CircuitBreaker:
    """Async-safe circuit breaker for a single downstream service.

    Args:
        service:            Human-readable name (used in logs and errors).
        failure_threshold:  Number of consecutive failures before opening.
        success_threshold:  Consecutive successes in HALF-OPEN before closing.
        recovery_seconds:   Seconds the breaker stays OPEN before HALF-OPEN.
        excluded_exceptions: Exception types that do *not* count as failures
                             (e.g. ``asyncio.CancelledError``).

    Usage::

        breaker = CircuitBreaker("openai-llm", failure_threshold=5)

        try:
            result = await breaker.call(llm_pipeline.generate, prompt)
        except CircuitOpenError as e:
            # return cached / degraded response
            ...
    """

    def __init__(
        self,
        service: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_seconds: float = 30.0,
        excluded_exceptions: tuple[type[Exception], ...] = (asyncio.CancelledError,),
    ) -> None:
        self.service = service
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_seconds = recovery_seconds
        self.excluded_exceptions = excluded_exceptions

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute ``func(*args, **kwargs)`` through the breaker.

        Raises:
            CircuitOpenError: If the circuit is OPEN and recovery has not elapsed.
            Any exception raised by ``func``: recorded as a failure.
        """
        await self._check_state()
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.excluded_exceptions:
            raise
        except Exception as exc:
            await self._on_failure(exc)
            raise

    def as_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    def as_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def status_dict(self) -> dict:
        return {
            "service": self.service,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "seconds_until_recovery": max(
                0.0,
                self.recovery_seconds - (time.monotonic() - self._opened_at),
            )
            if self._state == CircuitState.OPEN
            else 0.0,
        }

    async def reset(self) -> None:
        """Manually reset to CLOSED — useful in tests or admin endpoints."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info("circuit_reset", service=self.service)

    # ── Internal state machine ────────────────────────────────────────────────

    async def _check_state(self) -> None:
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.recovery_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info("circuit_half_open", service=self.service)
                else:
                    retry_after = self.recovery_seconds - elapsed
                    raise CircuitOpenError(self.service, retry_after)

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("circuit_closed", service=self.service)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # reset on any success

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._failure_count += 1
            logger.warning(
                "circuit_failure_recorded",
                service=self.service,
                failure_count=self._failure_count,
                threshold=self.failure_threshold,
                error=str(exc),
            )
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed → reopen immediately
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.error("circuit_reopened", service=self.service)
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.error(
                    "circuit_opened",
                    service=self.service,
                    failure_count=self._failure_count,
                )


# ── Registry ─────────────────────────────────────────────────────────────────


class CircuitBreakerRegistry:
    """Singleton registry of all circuit breakers in the application.

    Usage::

        registry = CircuitBreakerRegistry.instance()
        stt_breaker = registry.get("stt")
    """

    _instance: CircuitBreakerRegistry | None = None

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    @classmethod
    def instance(cls) -> CircuitBreakerRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(
        self,
        service: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_seconds: float = 30.0,
    ) -> CircuitBreaker:
        """Return existing breaker or create a new one for ``service``."""
        if service not in self._breakers:
            self._breakers[service] = CircuitBreaker(
                service,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                recovery_seconds=recovery_seconds,
            )
        return self._breakers[service]

    def all_statuses(self) -> list[dict]:
        return [b.status_dict() for b in self._breakers.values()]
