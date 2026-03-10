"""Unit tests for the AudioProcessor (VAD + preprocessing)."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.processor import AudioProcessor, ProcessedAudio


@pytest.mark.asyncio
class TestAudioProcessor:
    async def test_silent_audio_has_no_speech(self, silent_audio_bytes):
        processor = AudioProcessor()
        result = await processor.process(silent_audio_bytes, 16_000)
        assert isinstance(result, ProcessedAudio)
        assert not result.has_speech

    async def test_tone_audio_detected_as_speech(self, speech_audio_bytes):
        processor = AudioProcessor()
        result = await processor.process(speech_audio_bytes, 16_000)
        assert isinstance(result, ProcessedAudio)
        # A 440Hz tone should exceed the energy threshold
        assert result.has_speech

    async def test_output_sample_rate_always_16k(self, speech_audio_bytes):
        processor = AudioProcessor()
        result = await processor.process(speech_audio_bytes, 16_000)
        assert result.sample_rate == 16_000

    async def test_resampling_from_44100(self, speech_audio_bytes):
        """Input at 44.1kHz should be resampled to 16kHz without error."""
        # Re-generate tone at 44100 Hz
        sample_rate = 44_100
        duration_s = 0.3
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        audio_44k = tone.tobytes()

        processor = AudioProcessor()
        result = await processor.process(audio_44k, 44_100)
        assert result.sample_rate == 16_000

    async def test_duration_ms_correct(self, speech_audio_bytes):
        processor = AudioProcessor()
        result = await processor.process(speech_audio_bytes, 16_000)
        # 500ms tone → some voice frames should remain after VAD
        assert result.duration_ms > 0

    async def test_odd_length_bytes_handled(self):
        """Odd-length byte sequences should not raise."""
        processor = AudioProcessor()
        odd_bytes = b"\x00" * 101  # 101 bytes — odd
        result = await processor.process(odd_bytes, 16_000)
        assert isinstance(result, ProcessedAudio)

    def test_bytes_to_float32_round_trip(self):
        proc = AudioProcessor()
        original = np.array([0, 16383, -16384, 32767, -32768], dtype=np.int16)
        raw = original.tobytes()
        floats = proc._bytes_to_float32(raw)
        back = (floats * 32767).astype(np.int16)
        # Values should be approximately preserved
        np.testing.assert_allclose(original, back, atol=2)

    def test_normalise_scales_to_09(self):
        proc = AudioProcessor()
        audio = np.array([0.1, 0.5, -0.8, 0.3], dtype=np.float32)
        normalised = proc._normalise(audio)
        assert np.abs(normalised).max() == pytest.approx(0.9, abs=1e-5)

    def test_normalise_silent_audio_unchanged(self):
        proc = AudioProcessor()
        audio = np.zeros(100, dtype=np.float32)
        result = proc._normalise(audio)
        np.testing.assert_array_equal(result, audio)
