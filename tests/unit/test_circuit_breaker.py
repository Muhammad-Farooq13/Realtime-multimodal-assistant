"""Unit tests for CircuitBreaker."""

from __future__ import annotations

import asyncio

import pytest

from src.pipeline.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


async def _ok() -> str:
    return "success"


async def _fail() -> None:
    raise ValueError("service error")


@pytest.mark.asyncio
class TestCircuitBreakerClosed:
    async def test_successful_call_returns_value(self):
        cb = CircuitBreaker("test-svc", failure_threshold=3)
        result = await cb.call(_ok)
        assert result == "success"

    async def test_stays_closed_on_success(self):
        cb = CircuitBreaker("test-svc", failure_threshold=3)
        await cb.call(_ok)
        assert cb.state == CircuitState.CLOSED

    async def test_failure_increments_count(self):
        cb = CircuitBreaker("test-svc", failure_threshold=3)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb._failure_count == 1
        assert cb.state == CircuitState.CLOSED

    async def test_opens_after_threshold(self):
        cb = CircuitBreaker("test-svc", failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state == CircuitState.OPEN

    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test-svc", failure_threshold=3)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        await cb.call(_ok)
        assert cb._failure_count == 0


@pytest.mark.asyncio
class TestCircuitBreakerOpen:
    async def test_open_circuit_fast_fails(self):
        cb = CircuitBreaker("test-svc", failure_threshold=1)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(_ok)
        assert exc_info.value.service == "test-svc"

    async def test_circuit_open_error_has_retry_after(self):
        cb = CircuitBreaker("test-svc", failure_threshold=1, recovery_seconds=30.0)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(_ok)
        assert exc_info.value.retry_after <= 30.0
        assert exc_info.value.retry_after > 0.0

    async def test_transitions_to_half_open_after_recovery(self):
        cb = CircuitBreaker("test-svc", failure_threshold=1, recovery_seconds=0.05)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.06)
        # Trigger the state check
        await cb._check_state()
        assert cb.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
class TestCircuitBreakerHalfOpen:
    async def _open_breaker(self, cb: CircuitBreaker) -> CircuitBreaker:
        with pytest.raises(ValueError):
            await cb.call(_fail)
        return cb

    async def test_successful_probe_closes_circuit(self):
        cb = CircuitBreaker(
            "test-svc",
            failure_threshold=1,
            success_threshold=1,
            recovery_seconds=0.05,
        )
        await self._open_breaker(cb)
        await asyncio.sleep(0.06)
        result = await cb.call(_ok)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    async def test_failed_probe_reopens_circuit(self):
        cb = CircuitBreaker(
            "test-svc",
            failure_threshold=1,
            success_threshold=2,
            recovery_seconds=0.05,
        )
        await self._open_breaker(cb)
        await asyncio.sleep(0.06)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN

    async def test_multiple_successes_required_to_close(self):
        cb = CircuitBreaker(
            "test-svc",
            failure_threshold=1,
            success_threshold=2,
            recovery_seconds=0.05,
        )
        await self._open_breaker(cb)
        await asyncio.sleep(0.06)
        await cb.call(_ok)
        assert cb.state == CircuitState.HALF_OPEN  # still half-open after 1
        await cb.call(_ok)
        assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
class TestCircuitBreakerReset:
    async def test_manual_reset_closes_circuit(self):
        cb = CircuitBreaker("test-svc", failure_threshold=1)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN
        await cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    async def test_cancelled_error_not_counted(self):
        cb = CircuitBreaker("test-svc", failure_threshold=2)

        async def cancelled():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await cb.call(cancelled)
        assert cb._failure_count == 0
