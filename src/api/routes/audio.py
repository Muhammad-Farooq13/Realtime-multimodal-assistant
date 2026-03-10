"""Audio REST endpoints — one-shot transcription and synthesis."""

from __future__ import annotations

import base64
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from src.audio.processor import AudioProcessor
from src.audio.tts import TTSService
from src.audio.transcription import TranscriptionService
from src.config.settings import Settings, get_settings

router = APIRouter()


# ── Dependency helpers ────────────────────────────────────────────────────────


def get_stt(request: Request) -> TranscriptionService:
    return request.app.state.stt_service


def get_settings_dep() -> Settings:
    return get_settings()


# ── Request / Response schemas ────────────────────────────────────────────────


class TranscribeResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration_ms: float


class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/transcribe", response_model=TranscribeResponse, summary="One-shot STT")
async def transcribe_audio(
    file: Annotated[UploadFile, File(description="Raw PCM16 or WAV audio file")],
    stt: TranscriptionService = Depends(get_stt),
    settings: Settings = Depends(get_settings_dep),
):
    """Transcribe an uploaded audio file.

    Accepts WAV, MP3, or raw PCM16 (16 kHz mono).
    Returns the transcript with confidence score and detected language.
    """
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    max_bytes = settings.max_audio_duration_seconds * settings.audio_sample_rate * 2
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds {settings.max_audio_duration_seconds}s limit",
        )

    processor = AudioProcessor()
    processed = await processor.process(raw_bytes, settings.audio_sample_rate)
    result = await stt.transcribe(processed)

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        confidence=round(result.confidence, 3),
        duration_ms=round(processed.duration_ms, 1),
    )


@router.post("/synthesize", summary="One-shot TTS", response_class=Response)
async def synthesize_speech(body: SynthesizeRequest):
    """Synthesise speech from text.

    Returns raw audio bytes (MP3 or PCM16 depending on server config).
    Use ``Accept: audio/mpeg`` or ``Accept: audio/pcm`` to request format.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(body.text) > 2000:
        raise HTTPException(status_code=400, detail="text exceeds 2000 character limit")

    tts = TTSService()
    audio = await tts.synthesize(body.text)

    if audio is None:
        raise HTTPException(status_code=503, detail="TTS service temporarily unavailable")

    return Response(content=audio, media_type="audio/mpeg")
