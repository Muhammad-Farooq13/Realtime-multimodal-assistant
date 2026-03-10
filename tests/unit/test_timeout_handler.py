"""Unit tests for TimeoutHandler and graceful degradation."""

from __future__ import annotations

import asyncio

import pytest

from src.pipeline.timeout_handler import (
    DegradationLevel,
    StageTimeoutError,
    TimeoutResult,
    run_with_timeout,
    track_degradation,
    with_timeout,
)


# ── run_with_timeout ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRunWithTimeout:
    async def test_returns_value_when_within_budget(self):
        async def fast():
            return 42

        result = await run_with_timeout(fast(), budget_s=1.0, stage="test")
        assert result.value == 42
        assert not result.timed_out

    async def test_returns_fallback_on_timeout(self):
        async def slow():
            await asyncio.sleep(5.0)

        result = await run_with_timeout(
            slow(),
            budget_s=0.05,
            stage="stt",
            fallback_value="cached_response",
        )
        assert result.timed_out
        assert result.value == "cached_response"
        assert result.degradation_level == DegradationLevel.MODERATE

    async def test_raises_stage_timeout_error_without_fallback(self):
        async def slow():
            await asyncio.sleep(5.0)

        with pytest.raises(StageTimeoutError) as exc_info:
            await run_with_timeout(slow(), budget_s=0.05, stage="llm")

        assert exc_info.value.stage == "llm"
        assert exc_info.value.budget_ms == 50

    async def test_none_fallback_treats_as_no_fallback(self):
        """fallback_value=None should raise StageTimeoutError, not return None."""
        async def slow():
            await asyncio.sleep(5.0)

        with pytest.raises(StageTimeoutError):
            await run_with_timeout(slow(), budget_s=0.05, stage="tts", fallback_value=None)

    async def test_non_timeout_exception_propagates(self):
        async def fail():
            raise ValueError("network error")

        with pytest.raises(ValueError, match="network error"):
            await run_with_timeout(fail(), budget_s=1.0, stage="llm")


# ── with_timeout decorator ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestWithTimeoutDecorator:
    async def test_decorator_returns_timeout_result(self):
        @with_timeout(budget_s=1.0, stage="test")
        async def fast_fn() -> str:
            return "ok"

        result = await fast_fn()
        assert isinstance(result, TimeoutResult)
        assert result.value == "ok"
        assert not result.timed_out

    async def test_decorator_invokes_fallback(self):
        async def fallback_fn() -> str:
            return "fallback"

        @with_timeout(budget_s=0.05, stage="stt", fallback=fallback_fn)
        async def slow_fn() -> str:
            await asyncio.sleep(5.0)
            return "never"

        result = await slow_fn()
        assert result.timed_out
        assert result.value == "fallback"
        assert result.fallback_used == "fallback_fn"

    async def test_decorator_raises_without_fallback(self):
        @with_timeout(budget_s=0.05, stage="llm")
        async def slow_fn():
            await asyncio.sleep(5.0)

        with pytest.raises(StageTimeoutError):
            await slow_fn()

    async def test_decorator_passes_args_to_fallback(self):
        """Fallback must receive the same args as the decorated function."""
        received_args = []

        async def fallback(x: int, y: str) -> str:
            received_args.extend([x, y])
            return "fallback"

        @with_timeout(budget_s=0.05, stage="test", fallback=fallback)
        async def slow_fn(x: int, y: str) -> str:
            await asyncio.sleep(5.0)
            return "never"

        await slow_fn(99, "hello")
        assert received_args == [99, "hello"]


# ── track_degradation ────────────────────────────────────────────────────────


class TestTrackDegradation:
    def test_no_timeouts_returns_none(self):
        results = [
            TimeoutResult(value=1),
            TimeoutResult(value=2),
        ]
        assert track_degradation(results) == DegradationLevel.NONE

    def test_single_moderate_timeout(self):
        results = [
            TimeoutResult(value=1),
            TimeoutResult(
                value="fallback",
                timed_out=True,
                degradation_level=DegradationLevel.MODERATE,
            ),
        ]
        assert track_degradation(results) == DegradationLevel.MODERATE

    def test_three_timeouts_is_severe(self):
        results = [
            TimeoutResult(value="f", timed_out=True, degradation_level=DegradationLevel.MINOR),
            TimeoutResult(value="f", timed_out=True, degradation_level=DegradationLevel.MINOR),
            TimeoutResult(value="f", timed_out=True, degradation_level=DegradationLevel.MINOR),
        ]
        assert track_degradation(results) == DegradationLevel.SEVERE

    def test_empty_list(self):
        assert track_degradation([]) == DegradationLevel.NONE
