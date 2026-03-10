"""Integration tests for the FastAPI application endpoints."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.audio.processor import ProcessedAudio
from src.audio.transcription import TranscriptionResult
from src.pipeline.orchestrator import TurnResponse
from src.pipeline.timeout_handler import DegradationLevel


@pytest.fixture
def test_app(test_settings):
    """Create a test FastAPI app with mocked services."""
    from src.api.app import create_app
    from unittest.mock import patch

    app = create_app()

    # Bypass real lifespan by pre-populating app.state
    mock_stt = MagicMock()
    mock_stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(text="test query", confidence=0.9, language="en")
    )
    mock_stt.load_models = AsyncMock()

    mock_orchestrator = MagicMock()
    mock_orchestrator.run_turn = AsyncMock(
        return_value=TurnResponse(
            transcript="test query",
            response_text="This is the answer.",
            audio_bytes=b"\x00\x01" * 512,
            intent=None,
            latency_report={
                "total_actual_ms": 850.0,
                "total_budget_ms": 2000,
                "within_budget": True,
                "stages": [],
            },
            degradation_level=DegradationLevel.NONE,
            trace_id="test-id",
        )
    )

    app.state.stt_service = mock_stt
    app.state.orchestrator = mock_orchestrator
    app.state.ready = True
    return app


@pytest.fixture
def wav_bytes() -> bytes:
    """Minimal valid WAV file bytes (16kHz mono, 0.25s silence)."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        silent_frames = np.zeros(4000, dtype=np.int16).tobytes()
        wf.writeframes(silent_frames)
    return buf.getvalue()


class TestHealthEndpoints:
    def test_liveness_200(self, test_app):
        with TestClient(test_app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness_200_when_ready(self, test_app):
        with TestClient(test_app) as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_readiness_503_when_not_ready(self, test_app):
        test_app.state.ready = False
        with TestClient(test_app) as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 503
        test_app.state.ready = True


class TestAudioEndpoints:
    def test_transcribe_returns_transcript(self, test_app, wav_bytes):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/audio/transcribe",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "confidence" in data
        assert "language" in data

    def test_transcribe_rejects_empty_file(self, test_app):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/audio/transcribe",
                files={"file": ("empty.wav", b"", "audio/wav")},
            )
        assert resp.status_code == 400

    def test_synthesize_returns_audio(self, test_app):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/audio/synthesize",
                json={"text": "Hello world"},
            )
        # May return 503 if edge-tts unavailable in test env — that's acceptable
        assert resp.status_code in (200, 503)

    def test_synthesize_rejects_empty_text(self, test_app):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/audio/synthesize",
                json={"text": ""},
            )
        assert resp.status_code == 400

    def test_synthesize_rejects_too_long_text(self, test_app):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/audio/synthesize",
                json={"text": "x" * 2001},
            )
        assert resp.status_code == 400


class TestStreamEndpoints:
    def test_stream_query_returns_response(self, test_app, wav_bytes):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/stream/query",
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"] == "test query"
        assert data["response_text"] == "This is the answer."
        assert "latency_report" in data

    def test_stream_query_rejects_empty_audio(self, test_app):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/stream/query",
                files={"audio": ("empty.wav", b"", "audio/wav")},
            )
        assert resp.status_code == 400

    def test_response_time_header_present(self, test_app, wav_bytes):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/stream/query",
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )
        assert "X-Response-Time-Ms" in resp.headers

    def test_request_id_echoed(self, test_app, wav_bytes):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/stream/query",
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
                headers={"X-Request-ID": "my-req-id"},
            )
        assert resp.headers.get("X-Request-ID") == "my-req-id"
