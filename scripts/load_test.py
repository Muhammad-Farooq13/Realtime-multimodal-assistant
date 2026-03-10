"""Locust load test for the multimodal assistant API.

Usage
=====
# Headless (CI)
locust -f scripts/load_test.py --host http://localhost:8000 \
       --users 20 --spawn-rate 2 --run-time 60s --headless

# Interactive UI
locust -f scripts/load_test.py --host http://localhost:8000

Scenarios
=========
- HealthCheck (10%) — GET /health every 5s
- AudioTranscribe (60%) — POST /api/v1/audio/transcribe with synthetic WAV
- MultimodalQuery (30%) — POST /api/v1/stream/query with audio + image
"""

from __future__ import annotations

import io
import random
import wave

import numpy as np
from locust import HttpUser, between, task


def _make_wav_bytes(duration_s: float = 0.3) -> bytes:
    sample_rate = 16_000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(tone.tobytes())
    return buf.getvalue()


def _make_jpeg_bytes() -> bytes:
    """Minimal 8x8 JPEG for vision multimodal tests."""
    try:
        from PIL import Image

        img = Image.new("RGB", (8, 8), color=(random.randint(0, 255), 128, 64))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        return b""


class VoiceAssistantUser(HttpUser):
    """Simulates a real-time voice assistant client."""

    wait_time = between(1, 3)

    @task(10)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(60)
    def transcribe_audio(self):
        wav = _make_wav_bytes(duration_s=random.uniform(0.2, 1.5))
        with self.client.post(
            "/api/v1/audio/transcribe",
            files={"file": ("audio.wav", wav, "audio/wav")},
            catch_response=True,
            name="/api/v1/audio/transcribe",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "text" not in data:
                    resp.failure("Missing 'text' in response")
            elif resp.status_code == 503:
                resp.success()  # degraded but expected under load
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(30)
    def multimodal_query(self):
        wav = _make_wav_bytes(duration_s=0.5)
        jpeg = _make_jpeg_bytes()

        files = {"audio": ("audio.wav", wav, "audio/wav")}
        if jpeg:
            files["image"] = ("frame.jpg", jpeg, "image/jpeg")

        with self.client.post(
            "/api/v1/stream/query",
            files=files,
            catch_response=True,
            name="/api/v1/stream/query",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                report = data.get("latency_report", {})
                total_ms = report.get("total_actual_ms", 0)
                if total_ms > 3500:
                    resp.failure(f"E2E latency exceeded SLA: {total_ms:.0f}ms > 3500ms")
            elif resp.status_code in (503, 504):
                resp.success()  # degraded service — acceptable
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")
