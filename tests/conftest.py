"""Shared pytest fixtures for unit and integration tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.audio.processor import AudioProcessor, ProcessedAudio
from src.audio.transcription import TranscriptionResult
from src.config.settings import Settings
from src.pipeline.circuit_breaker import CircuitBreakerRegistry
from src.vision.processor import VisionContext


# ── Event loop ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


# ── Settings override ─────────────────────────────────────────────────────────


@pytest.fixture
def test_settings(monkeypatch) -> Settings:
    """Return a Settings instance with safe test defaults."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "tiny")
    monkeypatch.setenv("PIPELINE_TOTAL_BUDGET_MS", "5000")
    monkeypatch.setenv("STT_TIMEOUT_MS", "2000")
    monkeypatch.setenv("LLM_TIMEOUT_MS", "3000")
    monkeypatch.setenv("METRICS_ENABLED", "false")
    from src.config.settings import get_settings
    get_settings.cache_clear()
    return get_settings()


# ── Audio fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def silent_audio_bytes() -> bytes:
    """256 frames of PCM16 silence (16kHz, mono)."""
    import numpy as np
    samples = np.zeros(4096, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def speech_audio_bytes() -> bytes:
    """Synthetic 440Hz tone as PCM16 (simulates 'speech' above energy threshold)."""
    import numpy as np
    sample_rate = 16_000
    duration_s = 0.5
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return tone.tobytes()


@pytest.fixture
def processed_audio_with_speech(speech_audio_bytes) -> ProcessedAudio:
    return ProcessedAudio(
        pcm_bytes=speech_audio_bytes,
        sample_rate=16_000,
        duration_ms=500.0,
        has_speech=True,
        silence_trimmed_ms=0.0,
    )


@pytest.fixture
def mock_transcription_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="What is the weather like today?",
        confidence=0.92,
        language="en",
    )


# ── Vision fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Tiny 8x8 white JPEG for fast tests."""
    from PIL import Image
    import io
    img = Image.new("RGB", (8, 8), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_vision_context() -> VisionContext:
    return VisionContext(
        base64_image="abc123",
        mime_type="image/jpeg",
        width=640,
        height=480,
    )


# ── Service mocks ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_stt_service(mock_transcription_result):
    svc = MagicMock()
    svc.transcribe = AsyncMock(return_value=mock_transcription_result)
    svc.transcribe_fast = AsyncMock(
        return_value=TranscriptionResult(text="weather today", confidence=0.75, language="en")
    )
    return svc


@pytest.fixture
def mock_llm_pipeline():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="It is sunny and 22°C today.")

    async def _fake_stream(*args, **kwargs):
        for token in ["It ", "is ", "sunny ", "today."]:
            yield token

    llm.stream = _fake_stream
    return llm


@pytest.fixture
def mock_tts_service():
    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=b"\x00\x01" * 1024)
    return tts


# ── Circuit breaker reset ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
async def reset_circuit_breakers():
    """Reset all circuit breakers before each test to ensure isolation."""
    registry = CircuitBreakerRegistry.instance()
    for breaker in registry._breakers.values():
        await breaker.reset()
    yield
    for breaker in registry._breakers.values():
        await breaker.reset()
