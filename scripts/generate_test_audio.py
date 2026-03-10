#!/usr/bin/env python
"""Generate synthetic test audio clips for use in tests and demos.

Usage
=====
    python scripts/generate_test_audio.py --output data/samples/

Generates
=========
- silence_500ms.wav    — 500ms of silence (tests VAD)
- tone_440hz_1s.wav    — 1s of 440Hz tone (tests energy-based VAD fallback)
- speech_sim_2s.wav    — 2s chirp (simulates spoken sentence energy envelope)
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def save_wav(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())
    print(f"  Saved: {path}  ({len(samples) / sample_rate * 1000:.0f}ms)")


def generate_silence(duration_s: float, sr: int = 16_000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.int16)


def generate_tone(freq: float, duration_s: float, amplitude: int = 16000, sr: int = 16_000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.int16)


def generate_speech_sim(duration_s: float = 2.0, sr: int = 16_000) -> np.ndarray:
    """Chirp with envelope — simulates the energy profile of speech."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    freq = np.linspace(200, 2000, len(t))  # frequency sweep
    chirp = np.sin(2 * np.pi * freq * t / sr)

    # Natural speech envelope: ramp up/down with short pauses
    envelope = np.ones(len(t))
    pause_regions = [(0.3, 0.4), (0.8, 0.95), (1.4, 1.5)]
    for start, end in pause_regions:
        s_idx = int(start * sr)
        e_idx = int(end * sr)
        envelope[s_idx:e_idx] = 0.05

    return (chirp * envelope * 16000).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/samples", help="Output directory")
    args = parser.parse_args()
    out = Path(args.output)

    print(f"\nGenerating test audio clips → {out}/\n")
    save_wav(out / "silence_500ms.wav", generate_silence(0.5))
    save_wav(out / "tone_440hz_1s.wav", generate_tone(440, 1.0))
    save_wav(out / "tone_880hz_500ms.wav", generate_tone(880, 0.5))
    save_wav(out / "speech_sim_2s.wav", generate_speech_sim(2.0))
    print("\nDone.")


if __name__ == "__main__":
    main()
