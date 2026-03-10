"""Text-to-Speech service using Microsoft edge-tts.

edge-tts streams synthesised audio directly without any local model,
giving neural voice quality at ~0ms model-load time and ~100–300ms TTFB.

The ``synthesize`` method returns raw PCM16 bytes ready for WebSocket streaming.

Fallback
========
If edge-tts is unavailable (network error, circuit open), the method returns
``None`` so the orchestrator can fall back to text-only response.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from typing import AsyncIterator

import structlog

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class TTSService:
    """Async TTS wrapper around edge-tts.

    All public methods are safe to call concurrently from multiple tasks —
    edge-tts creates a new HTTP connection per call.
    """

    def __init__(self) -> None:
        self._voice = settings.tts_voice
        self._rate = settings.tts_rate
        self._volume = settings.tts_volume

    async def synthesize(self, text: str) -> bytes | None:
        """Synthesise ``text`` and return MP3/PCM bytes.

        Returns ``None`` on any error so the caller can degrade gracefully.
        """
        if not text.strip():
            return None
        try:
            return await self._synthesize_edge_tts(text)
        except Exception as exc:
            logger.warning("tts_edge_failed", error=str(exc))
            return None

    async def stream_sentences(self, text: str) -> AsyncIterator[bytes]:
        """Yield audio chunks sentence-by-sentence for lower perceived latency.

        Splits text on sentence boundaries, synthesises each, and yields
        bytes as soon as each sentence is ready.
        """
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if sentence.strip():
                audio = await self.synthesize(sentence)
                if audio:
                    yield audio

    # ── Private ───────────────────────────────────────────────────────────────

    async def _synthesize_edge_tts(self, text: str) -> bytes:
        """Use edge-tts to produce MP3 audio and decode to PCM bytes."""
        try:
            import edge_tts  # type: ignore
        except ImportError as exc:
            raise RuntimeError("edge-tts not installed") from exc

        communicate = edge_tts.Communicate(
            text,
            self._voice,
            rate=self._rate,
            volume=self._volume,
        )

        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])

        if not mp3_chunks:
            return b""

        mp3_bytes = b"".join(mp3_chunks)
        # Decode MP3 → PCM16 for WebSocket streaming
        return self._mp3_to_pcm16(mp3_bytes)

    @staticmethod
    def _mp3_to_pcm16(mp3_bytes: bytes) -> bytes:
        """Decode MP3 to PCM16 LE mono @ 24kHz.

        Uses pydub/ffmpeg pipeline. Returns raw PCM bytes.
        """
        try:
            from pydub import AudioSegment  # type: ignore

            seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            seg = seg.set_channels(1).set_frame_rate(24_000).set_sample_width(2)
            return seg.raw_data
        except ImportError:
            # pydub not installed — return raw MP3 bytes as fallback
            # The client must handle MP3 decoding in this case
            return mp3_bytes
        except Exception as exc:
            logger.warning("mp3_decode_failed", error=str(exc))
            return mp3_bytes

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on punctuation boundaries."""
        import re
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]
