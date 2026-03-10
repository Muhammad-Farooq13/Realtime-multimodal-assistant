"""Speech-to-Text service using faster-whisper (CTranslate2 backend).

faster-whisper is 2–4× faster than openai-whisper for the same model size
because it uses CTranslate2's optimised int8 quantisation.

Fallback strategy
=================
If the primary model (e.g. ``base``) is too slow (circuit opens or timeout
fires), the orchestrator switches to the ``tiny`` model for that turn.
This is surfaced via the ``TranscriptionService.transcribe_fast()`` method.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from src.audio.processor import ProcessedAudio
from src.config.settings import get_settings

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class TranscriptionResult:
    """Output of the STT stage."""

    text: str
    confidence: float          # Average log-prob converted to [0, 1]
    language: str
    segments: list[dict] | None = None


class TranscriptionService:
    """Wraps faster-whisper for async transcription.

    The underlying WhisperModel is CPU/CUDA-bound, so we run it in a
    thread pool executor to avoid blocking the asyncio event loop.

    One ``TranscriptionService`` instance is created at startup and shared
    (each call gets its own CPU thread via ``run_in_executor``).
    """

    def __init__(self) -> None:
        self._primary_model: WhisperModel | None = None
        self._fallback_model: WhisperModel | None = None
        self._lock = asyncio.Lock()

    async def load_models(self) -> None:
        """Pre-load both primary and fallback models into RAM.

        Called once at application startup (lifespan handler).
        """
        logger.info("loading_stt_models", primary=settings.whisper_model_size)
        loop = asyncio.get_event_loop()
        # Load primary
        self._primary_model = await loop.run_in_executor(
            None, self._load_model, settings.whisper_model_size
        )
        # Load fallback (usually tiny — already in RAM after primary load)
        if settings.whisper_fallback_model_size != settings.whisper_model_size:
            self._fallback_model = await loop.run_in_executor(
                None, self._load_model, settings.whisper_fallback_model_size
            )
        else:
            self._fallback_model = self._primary_model
        logger.info("stt_models_ready")

    async def transcribe(self, audio: ProcessedAudio) -> TranscriptionResult:
        """Transcribe using the primary (higher accuracy) model."""
        return await self._transcribe_with(audio, self._primary_model, "primary")

    async def transcribe_fast(self, audio: ProcessedAudio) -> TranscriptionResult:
        """Transcribe using the fallback (faster, lower accuracy) model.

        Called by the orchestrator when the primary model times out.
        """
        return await self._transcribe_with(audio, self._fallback_model, "fallback")

    # ── Private ───────────────────────────────────────────────────────────────

    async def _transcribe_with(
        self, audio: ProcessedAudio, model: "WhisperModel | None", label: str
    ) -> TranscriptionResult:
        if not audio.has_speech:
            return TranscriptionResult(text="", confidence=1.0, language="en")

        if model is None:
            logger.warning("stt_model_not_loaded", label=label)
            return TranscriptionResult(text="", confidence=0.0, language="en")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_transcription,
            model,
            audio.pcm_bytes,
        )
        logger.debug(
            "transcription_complete",
            label=label,
            text_len=len(result.text),
            confidence=round(result.confidence, 3),
            language=result.language,
        )
        return result

    @staticmethod
    def _run_transcription(model: "WhisperModel", pcm_bytes: bytes) -> TranscriptionResult:
        """Blocking call — runs in thread pool executor."""
        import numpy as np
        from faster_whisper import WhisperModel  # noqa: F401 – already imported

        # faster-whisper expects float32 numpy array or a file path
        audio_np = (
            np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )

        segments, info = model.transcribe(
            audio_np,
            language=None,  # auto-detect
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )

        text_parts: list[str] = []
        avg_logprob_sum = 0.0
        count = 0

        for seg in segments:
            text_parts.append(seg.text)
            avg_logprob_sum += seg.avg_logprob
            count += 1

        avg_logprob = avg_logprob_sum / count if count else -1.0
        # Convert log-probability to pseudo-confidence [0, 1]
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob))

        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            confidence=confidence,
            language=info.language,
        )

    @staticmethod
    def _load_model(model_size: str) -> "WhisperModel":
        from faster_whisper import WhisperModel
        return WhisperModel(
            model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
