"""Unit tests for PipelineBudget and LatencyBudget."""

from __future__ import annotations

import asyncio

import pytest

from src.pipeline.latency_budget import (
    PIPELINE_TOTAL_BUDGET_MS,
    STAGE_BUDGETS_MS,
    PipelineBudget,
    PipelineReport,
    StageMeasurement,
    measure_stage,
)


class TestStageMeasurement:
    def test_within_budget(self):
        m = StageMeasurement(stage="stt", budget_ms=400, actual_ms=280.5)
        assert not m.over_budget
        assert m.overage_ms == 0.0

    def test_over_budget(self):
        m = StageMeasurement(stage="stt", budget_ms=400, actual_ms=650.0)
        assert m.over_budget
        assert m.overage_ms == pytest.approx(250.0)

    def test_as_dict_structure(self):
        m = StageMeasurement(stage="stt", budget_ms=400, actual_ms=300.0)
        d = m.as_dict()
        assert set(d.keys()) == {"stage", "budget_ms", "actual_ms", "over_budget", "overage_ms"}


class TestPipelineReport:
    def _make_report(self, actual_ms: float, budget_ms: int = 2000) -> PipelineReport:
        return PipelineReport(
            stages=[StageMeasurement("stt", 400, actual_ms)],
            total_budget_ms=budget_ms,
        )

    def test_within_budget(self):
        report = self._make_report(300.0)
        assert report.within_budget

    def test_over_budget(self):
        report = self._make_report(2100.0)
        assert not report.within_budget

    def test_total_actual_ms(self):
        report = PipelineReport(
            stages=[
                StageMeasurement("stt", 400, 300.0),
                StageMeasurement("llm", 800, 500.0),
            ],
            total_budget_ms=2000,
        )
        assert report.total_actual_ms == pytest.approx(800.0)

    def test_critical_path_is_max_stage(self):
        report = PipelineReport(
            stages=[
                StageMeasurement("stt", 400, 300.0),
                StageMeasurement("llm", 800, 700.0),
                StageMeasurement("tts", 200, 100.0),
            ],
            total_budget_ms=2000,
        )
        assert report.critical_path_ms == pytest.approx(700.0)

    def test_over_budget_stages_list(self):
        report = PipelineReport(
            stages=[
                StageMeasurement("stt", 400, 300.0),
                StageMeasurement("llm", 800, 1100.0),  # over
            ],
            total_budget_ms=2000,
        )
        assert len(report.over_budget_stages) == 1
        assert report.over_budget_stages[0].stage == "llm"


@pytest.mark.asyncio
class TestPipelineBudget:
    async def test_stage_measurement_recorded(self):
        budget = PipelineBudget(total_budget_ms=5000)
        async with budget.stage("stt", budget_ms=400):
            await asyncio.sleep(0.01)
        report = budget.report()
        assert len(report.stages) == 1
        assert report.stages[0].stage == "stt"
        assert report.stages[0].actual_ms >= 10.0

    async def test_multiple_stages_accumulated(self):
        budget = PipelineBudget(total_budget_ms=5000)
        for name in ("audio_preprocessing", "speech_to_text", "intent_classification"):
            async with budget.stage(name):
                pass
        report = budget.report()
        assert len(report.stages) == 3

    async def test_remaining_budget_decreases(self):
        budget = PipelineBudget(total_budget_ms=5000)
        remaining_before = budget.remaining_budget_ms()
        await asyncio.sleep(0.05)
        remaining_after = budget.remaining_budget_ms()
        assert remaining_after < remaining_before

    async def test_over_budget_callback_invoked(self):
        callback_invoked = []

        async def on_overrun(measurement):
            callback_invoked.append(measurement.stage)

        budget = PipelineBudget(total_budget_ms=5000)
        async with budget.stage("stt", budget_ms=1, on_over_budget=on_overrun):
            await asyncio.sleep(0.05)  # definitely over 1ms

        assert "stt" in callback_invoked

    async def test_trace_id_propagated_to_report(self):
        budget = PipelineBudget(trace_id="test-123")
        async with budget.stage("stt"):
            pass
        assert budget.report().trace_id == "test-123"


@pytest.mark.asyncio
class TestMeasureStage:
    async def test_standalone_context_manager(self):
        executed = []
        async with measure_stage("speech_to_text"):
            executed.append(True)
        assert executed

    async def test_stage_budgets_have_all_expected_keys(self):
        expected = [
            "audio_capture", "vad_segmentation", "network_ingress",
            "audio_preprocessing", "speech_to_text", "intent_classification",
            "vision_processing", "llm_first_token", "llm_generation",
            "tts_synthesis", "network_egress", "audio_playback_buffer",
        ]
        for key in expected:
            assert key in STAGE_BUDGETS_MS, f"Missing stage key: {key}"

    async def test_total_stage_budget_approximately_equals_pipeline_budget(self):
        total = sum(STAGE_BUDGETS_MS.values())
        # Total stage budgets should be within 20% of pipeline total
        assert abs(total - PIPELINE_TOTAL_BUDGET_MS) / PIPELINE_TOTAL_BUDGET_MS < 0.20
