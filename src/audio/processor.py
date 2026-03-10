"""Audio pre-processing: PCM normalisation, resampling, and Voice Activity Detection.

VAD strategy
============
Uses ``webrtcvad`` (Google's WebRTC VAD C library) for high-accuracy,
low-latency voice/silence detection.

Frame durations supported by WebRTC VAD: 10ms, 20ms, or 30ms.
Default chunk size: 30ms @ 16 kHz → 960 samples per frame.

Processing pipeline
===================
raw bytes → resample → normalise → VAD frame scan → trim silence → ProcessedAudio
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import webrtcvad
    _WEBRTCVAD_AVAILABLE = True
except ImportError:
    _WEBRTCVAD_AVAILABLE = False
    logger.warning("webrtcvad_not_available", note="Falling back to energy-based VAD")


@dataclass
class ProcessedAudio:
    """Audio ready for the STT stage."""

    pcm_bytes: bytes          # 16-bit signed PCM, mono
    sample_rate: int          # Hz (always 16 000 after processing)
    duration_ms: float        # duration of voice-active audio
    has_speech: bool          # False → skip STT
    silence_trimmed_ms: float  # how much silence was removed


class AudioProcessor:
    """Pre-processes raw audio chunks for downstream pipeline stages.

    Thread-safe (stateless between calls). One instance per application.
    """

    TARGET_SAMPLE_RATE = 16_000
    FRAME_DURATION_MS = 30       # 10 | 20 | 30 supported by WebRTC VAD
    VAD_AGGRESSIVENESS = 2       # 0 = least, 3 = most aggressive

    def __init__(self, vad_aggressiveness: int = VAD_AGGRESSIVENESS) -> None:
        self._vad: "webrtcvad.Vad | None" = None
        if _WEBRTCVAD_AVAILABLE:
            self._vad = webrtcvad.Vad(vad_aggressiveness)

    # ── Public API ────────────────────────────────────────────────────────────

    async def process(self, raw_bytes: bytes, sample_rate: int) -> ProcessedAudio:
        """Process raw PCM16 audio bytes.

        Args:
            raw_bytes:   Raw PCM16 LE bytes (may be any sample rate).
            sample_rate: Sample rate of ``raw_bytes``.

        Returns:
            ``ProcessedAudio`` with resampled, silence-trimmed PCM16 bytes.
        """
        # Convert bytes → numpy float32
        audio_np = self._bytes_to_float32(raw_bytes)

        # Resample to target rate if necessary
        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio_np = self._resample(audio_np, sample_rate, self.TARGET_SAMPLE_RATE)

        # Normalise amplitude
        audio_np = self._normalise(audio_np)

        # VAD — detect and trim silence
        pcm16 = self._float32_to_pcm16(audio_np)
        voice_frames, silence_trimmed_ms = self._apply_vad(pcm16)

        has_speech = len(voice_frames) > 0
        final_pcm = b"".join(voice_frames) if voice_frames else pcm16

        duration_ms = (len(final_pcm) / 2) / self.TARGET_SAMPLE_RATE * 1000

        return ProcessedAudio(
            pcm_bytes=final_pcm,
            sample_rate=self.TARGET_SAMPLE_RATE,
            duration_ms=duration_ms,
            has_speech=has_speech,
            silence_trimmed_ms=silence_trimmed_ms,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _bytes_to_float32(raw: bytes) -> np.ndarray:
        """Convert raw PCM16 LE bytes to float32 in [-1.0, 1.0]."""
        if len(raw) % 2 != 0:
            raw = raw[: len(raw) - 1]  # drop odd trailing byte
        samples = np.frombuffer(raw, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    @staticmethod
    def _float32_to_pcm16(audio: np.ndarray) -> bytes:
        """Convert float32 array to PCM16 LE bytes."""
        clipped = np.clip(audio, -1.0, 1.0)
        samples_int16 = (clipped * 32767).astype(np.int16)
        return samples_int16.tobytes()

    @staticmethod
    def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        """Simple linear resampling (low-dependency fallback).

        For production, replace with ``librosa.resample`` or ``soxr``.
        """
        try:
            import librosa  # type: ignore
            return librosa.resample(audio, orig_sr=orig_rate, target_sr=target_rate)
        except ImportError:
            ratio = target_rate / orig_rate
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio)

    @staticmethod
    def _normalise(audio: np.ndarray) -> np.ndarray:
        """Peak normalise to 90% of max amplitude."""
        peak = np.abs(audio).max()
        if peak < 1e-6:
            return audio
        return audio * (0.9 / peak)

    def _apply_vad(self, pcm16: bytes) -> tuple[list[bytes], float]:
        """Split PCM16 bytes into 30ms frames and filter out silence.

        Returns:
            (voice_frames, silence_trimmed_ms)
        """
        frame_bytes = int(self.TARGET_SAMPLE_RATE * self.FRAME_DURATION_MS / 1000) * 2
        frames = [
            pcm16[i : i + frame_bytes]
            for i in range(0, len(pcm16), frame_bytes)
            if len(pcm16[i : i + frame_bytes]) == frame_bytes
        ]

        if not frames:
            return [], 0.0

        silent_frames = 0
        voice_frames: list[bytes] = []

        for frame in frames:
            is_speech = self._is_speech(frame)
            if is_speech:
                voice_frames.append(frame)
            else:
                silent_frames += 1

        silence_trimmed_ms = silent_frames * self.FRAME_DURATION_MS
        return voice_frames, silence_trimmed_ms

    def _is_speech(self, frame: bytes) -> bool:
        """Classify a PCM16 frame as speech or silence."""
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, self.TARGET_SAMPLE_RATE)
            except Exception:
                pass
        # Fallback: energy threshold
        samples = struct.unpack(f"{len(frame) // 2}h", frame)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms > 300  # empirical threshold for 16-bit PCM
