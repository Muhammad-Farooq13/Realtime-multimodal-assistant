"""Integration tests for the full pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.audio.processor import ProcessedAudio
from src.audio.transcription import TranscriptionResult
from src.pipeline.circuit_breaker import CircuitBreakerRegistry
from src.pipeline.orchestrator import PipelineOrchestrator, TurnRequest, TurnResponse
from src.pipeline.timeout_handler import DegradationLevel
from src.vision.processor import VisionContext


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_request(include_image: bool = False) -> TurnRequest:
    audio = b"\x00\x01" * 8000  # 500ms of pseudo-audio
    image = b"\xff\xd8\xff" if include_image else None  # JPEG magic bytes
    return TurnRequest(audio_bytes=audio, image_bytes=image, trace_id="test-turn")


@pytest.fixture
def orchestrator(
    mock_stt_service,
    mock_llm_pipeline,
    mock_tts_service,
    mock_vision_context,
) -> PipelineOrchestrator:
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)

    from src.audio.processor import AudioProcessor

    orch._audio_processor = MagicMock()
    orch._audio_processor.process = AsyncMock(
        return_value=ProcessedAudio(
            pcm_bytes=b"\x00" * 16000,
            sample_rate=16_000,
            duration_ms=500.0,
            has_speech=True,
            silence_trimmed_ms=0.0,
        )
    )
    orch._stt = mock_stt_service
    orch._vision = MagicMock()
    orch._vision.process = AsyncMock(return_value=mock_vision_context)
    orch._llm = mock_llm_pipeline
    orch._tts = mock_tts_service

    from src.llm.intent import IntentClassifier, Intent, IntentCategory

    orch._intent = MagicMock()
    orch._intent.classify = AsyncMock(
        return_value=Intent(category=IntentCategory.QUESTION, confidence=0.9)
    )

    from src.monitoring.metrics import PipelineMetrics

    orch._metrics = MagicMock()
    orch._metrics.record_turn = MagicMock()
    orch._registry = CircuitBreakerRegistry.instance()

    return orch


@pytest.mark.asyncio
class TestOrchestratorRunTurn:
    async def test_returns_turn_response(self, orchestrator):
        request = _make_request()
        response = await orchestrator.run_turn(request)
        assert isinstance(response, TurnResponse)

    async def test_transcript_populated(self, orchestrator):
        response = await orchestrator.run_turn(_make_request())
        assert response.transcript == "What is the weather like today?"

    async def test_response_text_populated(self, orchestrator):
        response = await orchestrator.run_turn(_make_request())
        assert "sunny" in response.response_text

    async def test_latency_report_has_stages(self, orchestrator):
        response = await orchestrator.run_turn(_make_request())
        assert "stages" in response.latency_report
        assert len(response.latency_report["stages"]) > 0

    async def test_trace_id_preserved(self, orchestrator):
        request = _make_request()
        request.trace_id = "my-trace-99"
        response = await orchestrator.run_turn(request)
        assert response.trace_id == "my-trace-99"

    async def test_no_degradation_on_happy_path(self, orchestrator):
        response = await orchestrator.run_turn(_make_request())
        assert response.degradation_level == DegradationLevel.NONE

    async def test_audio_bytes_returned(self, orchestrator):
        response = await orchestrator.run_turn(_make_request())
        assert response.audio_bytes is not None
        assert len(response.audio_bytes) > 0


@pytest.mark.asyncio
class TestOrchestratorDegradation:
    async def test_stt_timeout_degrades_gracefully(self, orchestrator):
        """STT timeout should not raise — response should still be produced."""
        import asyncio

        async def slow_stt(_):
            await asyncio.sleep(10.0)

        orchestrator._stt.transcribe = slow_stt

        # Use very short timeout
        with patch("src.pipeline.orchestrator.settings") as mock_settings:
            mock_settings.stt_timeout_s = 0.05
            mock_settings.llm_timeout_s = 5.0
            mock_settings.tts_timeout_s = 5.0
            mock_settings.vision_timeout_s = 5.0
            mock_settings.pipeline_total_budget_ms = 10000
            mock_settings.circuit_breaker_failure_threshold = 10
            mock_settings.circuit_breaker_recovery_seconds = 60

            response = await orchestrator.run_turn(_make_request())

        # Should have degraded but still returned something
        assert isinstance(response, TurnResponse)

    async def test_circuit_open_on_stt_returns_empty_transcript(self, orchestrator):
        from src.pipeline.circuit_breaker import CircuitOpenError

        orchestrator._stt.transcribe = AsyncMock(side_effect=CircuitOpenError("stt", 10.0))
        response = await orchestrator.run_turn(_make_request())
        assert response.transcript == ""

    async def test_tts_failure_returns_text_only(self, orchestrator):
        orchestrator._tts.synthesize = AsyncMock(return_value=None)
        response = await orchestrator.run_turn(_make_request())
        assert response.response_text  # text still returned
        assert response.audio_bytes is None


@pytest.mark.asyncio
class TestOrchestratorStreamTurn:
    async def test_stream_yields_transcript_event(self, orchestrator):
        events = []
        async for event in orchestrator.stream_turn(_make_request()):
            events.append(event)

        types = [e["type"] for e in events]
        assert "transcript" in types

    async def test_stream_yields_latency_report_last(self, orchestrator):
        events = []
        async for event in orchestrator.stream_turn(_make_request()):
            events.append(event)

        assert events[-1]["type"] == "latency_report"

    async def test_stream_yields_llm_tokens(self, orchestrator):
        events = []
        async for event in orchestrator.stream_turn(_make_request()):
            events.append(event)

        token_events = [e for e in events if e["type"] == "llm_token"]
        assert len(token_events) > 0
