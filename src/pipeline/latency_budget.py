"""Latency Budget Tracking for the Real-Time Multimodal Pipeline.

Design goals
============
1. Every pipeline stage declares an *explicit* budget (ms) upfront.
2. ``LatencyBudget`` context manager measures actual wall-clock time per stage.
3. Budget overruns are recorded and can trigger graceful degradation callbacks.
4. A ``PipelineBudget`` aggregator tracks the full turn and emits a JSON report.

Budget breakdown (target: 2 000 ms E2E, p50)
=============================================
Stage                       Budget (ms)   % of total
─────────────────────────── ──────────── ───────────
1.  audio_capture                  20          1%
2.  vad_segmentation               30          1.5%
3.  network_ingress                50          2.5%
4.  audio_preprocessing            20          1%
5.  speech_to_text                400         20%
6.  intent_classification          30          1.5%
7.  vision_processing              80          4%
8.  llm_first_token               500         25%
9.  llm_generation                800         40%
10. tts_synthesis                 200         10%
11. network_egress                 50          2.5%
12. audio_playback_buffer          30          1.5%
─────────────────────────── ──────────── ───────────
Sequential total                2 210
Streaming pipeline              1 200        ─45%

The streaming pipeline gains the ~45% reduction by *overlapping* stages:
  STT partial → LLM (token stream) → TTS first sentence
rather than waiting for each stage to complete before starting the next.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Awaitable

import structlog

logger = structlog.get_logger(__name__)

# ── Stage budgets ─────────────────────────────────────────────────────────────

STAGE_BUDGETS_MS: dict[str, int] = {
    "audio_capture": 20,
    "vad_segmentation": 30,
    "network_ingress": 50,
    "audio_preprocessing": 20,
    "speech_to_text": 400,
    "intent_classification": 30,
    "vision_processing": 80,
    "llm_first_token": 500,
    "llm_generation": 800,
    "tts_synthesis": 200,
    "network_egress": 50,
    "audio_playback_buffer": 30,
}

PIPELINE_TOTAL_BUDGET_MS: int = 2_000


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class StageMeasurement:
    """Result of a single stage timing measurement."""

    stage: str
    budget_ms: int
    actual_ms: float
    over_budget: bool = field(init=False)
    overage_ms: float = field(init=False)

    def __post_init__(self) -> None:
        self.over_budget = self.actual_ms > self.budget_ms
        self.overage_ms = max(0.0, self.actual_ms - self.budget_ms)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "budget_ms": self.budget_ms,
            "actual_ms": round(self.actual_ms, 2),
            "over_budget": self.over_budget,
            "overage_ms": round(self.overage_ms, 2),
        }


@dataclass
class PipelineReport:
    """Aggregated latency report for a complete pipeline turn."""

    stages: list[StageMeasurement]
    total_budget_ms: int
    trace_id: str = ""

    @property
    def total_actual_ms(self) -> float:
        return sum(s.actual_ms for s in self.stages)

    @property
    def critical_path_ms(self) -> float:
        """Largest single-stage latency (the bottleneck)."""
        if not self.stages:
            return 0.0
        return max(s.actual_ms for s in self.stages)

    @property
    def within_budget(self) -> bool:
        return self.total_actual_ms <= self.total_budget_ms

    @property
    def over_budget_stages(self) -> list[StageMeasurement]:
        return [s for s in self.stages if s.over_budget]

    def as_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_budget_ms": self.total_budget_ms,
            "total_actual_ms": round(self.total_actual_ms, 2),
            "within_budget": self.within_budget,
            "critical_path_ms": round(self.critical_path_ms, 2),
            "over_budget_stages": [s.stage for s in self.over_budget_stages],
            "stages": [s.as_dict() for s in self.stages],
        }


# ── Context manager ───────────────────────────────────────────────────────────


@asynccontextmanager
async def measure_stage(
    stage: str,
    budget_ms: int | None = None,
    on_over_budget: Callable[[StageMeasurement], Awaitable[None]] | None = None,
) -> AsyncGenerator[None, None]:
    """Async context manager that measures wall-clock time for a pipeline stage.

    Args:
        stage:          Stage name (must match a key in ``STAGE_BUDGETS_MS``
                        or provide ``budget_ms`` explicitly).
        budget_ms:      Override the default budget for this stage.
        on_over_budget: Optional async callback invoked when the stage exceeds
                        its budget. Receives the ``StageMeasurement`` object.

    Yields:
        Nothing — the caller runs its work inside the ``async with`` block.

    Side-effects:
        Emits a structured log entry (DEBUG level) with timing info.

    Example::

        async with measure_stage("speech_to_text") as _:
            transcript = await stt_model.transcribe(audio)
    """
    resolved_budget = budget_ms or STAGE_BUDGETS_MS.get(stage, 9999)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1_000
        measurement = StageMeasurement(
            stage=stage,
            budget_ms=resolved_budget,
            actual_ms=elapsed_ms,
        )
        log_fn = logger.warning if measurement.over_budget else logger.debug
        log_fn(
            "stage_timed",
            stage=stage,
            actual_ms=round(elapsed_ms, 2),
            budget_ms=resolved_budget,
            over_budget=measurement.over_budget,
        )
        if measurement.over_budget and on_over_budget is not None:
            await on_over_budget(measurement)


class PipelineBudget:
    """Tracks all stage measurements for a single request turn.

    Usage::

        budget = PipelineBudget(trace_id="abc-123")

        async with budget.stage("speech_to_text"):
            transcript = await stt(audio)

        async with budget.stage("llm_first_token"):
            first_token = await llm.stream_first(...)

        report = budget.report()
        print(report.as_dict())
    """

    def __init__(
        self,
        total_budget_ms: int = PIPELINE_TOTAL_BUDGET_MS,
        trace_id: str = "",
    ) -> None:
        self._total_budget_ms = total_budget_ms
        self._trace_id = trace_id
        self._measurements: list[StageMeasurement] = []
        self._turn_start = time.perf_counter()

    @asynccontextmanager
    async def stage(
        self,
        name: str,
        budget_ms: int | None = None,
        on_over_budget: Callable[[StageMeasurement], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[None, None]:
        """Context manager for a single stage within this pipeline budget."""
        resolved_budget = budget_ms or STAGE_BUDGETS_MS.get(name, 9999)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1_000
            m = StageMeasurement(stage=name, budget_ms=resolved_budget, actual_ms=elapsed_ms)
            self._measurements.append(m)
            log_fn = logger.warning if m.over_budget else logger.debug
            log_fn(
                "stage_timed",
                stage=name,
                actual_ms=round(elapsed_ms, 2),
                budget_ms=resolved_budget,
                over_budget=m.over_budget,
                trace_id=self._trace_id,
            )
            if m.over_budget and on_over_budget is not None:
                await on_over_budget(m)

    def remaining_budget_ms(self) -> float:
        """Milliseconds left in the overall pipeline budget."""
        elapsed = (time.perf_counter() - self._turn_start) * 1_000
        return self._total_budget_ms - elapsed

    def is_budget_exhausted(self) -> bool:
        return self.remaining_budget_ms() <= 0

    def report(self) -> PipelineReport:
        return PipelineReport(
            stages=list(self._measurements),
            total_budget_ms=self._total_budget_ms,
            trace_id=self._trace_id,
        )
