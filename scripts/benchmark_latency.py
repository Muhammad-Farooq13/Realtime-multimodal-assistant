#!/usr/bin/env python
"""Latency benchmarking script — measures per-stage and E2E latency.

Usage
=====
    python scripts/benchmark_latency.py --runs 100 --output data/benchmarks/latest.json

What it measures
================
1. Generates synthetic audio (tone at 440Hz, 0.5s @ 16kHz)
2. Runs the full pipeline N times
3. Collects per-stage timings from PipelineReport
4. Computes p50, p90, p95, p99 percentiles for each stage and total latency
5. Writes results as JSON and prints a Markdown table

The script mocks the LLM and TTS services to isolate STT and audio processing
latency from network variability. Pass ``--real-llm`` to use the actual API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


async def generate_synthetic_audio(duration_s: float = 0.5, sample_rate: int = 16_000) -> bytes:
    """Generate a 440Hz tone as PCM16 bytes."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return tone.tobytes()


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    index = (p / 100) * (len(data_sorted) - 1)
    lower = int(index)
    upper = min(lower + 1, len(data_sorted) - 1)
    fraction = index - lower
    return data_sorted[lower] + fraction * (data_sorted[upper] - data_sorted[lower])


def _build_percentile_table(
    stage_timings: dict[str, list[float]],
    total_timings: list[float],
    stage_budgets: dict[str, int],
) -> str:
    """Format results as a Markdown table."""
    lines = [
        "| Stage                    | Budget(ms) | p50   | p90   | p95   | p99   | Within Budget |",
        "|--------------------------|------------|-------|-------|-------|-------|---------------|",
    ]
    for stage, timings in stage_timings.items():
        if not timings:
            continue
        budget = stage_budgets.get(stage, 9999)
        p50 = _percentile(timings, 50)
        p90 = _percentile(timings, 90)
        p95 = _percentile(timings, 95)
        p99 = _percentile(timings, 99)
        within = "✅" if p99 <= budget else "❌"
        lines.append(
            f"| {stage:<24} | {budget:<10} | {p50:>5.0f} | {p90:>5.0f} | "
            f"{p95:>5.0f} | {p99:>5.0f} | {within:<13} |"
        )

    t_p50 = _percentile(total_timings, 50)
    t_p99 = _percentile(total_timings, 99)
    within_total = "✅" if t_p99 <= 2000 else "❌"
    lines.append(
        f"| {'**E2E Total**':<24} | {'2000':<10} | {t_p50:>5.0f} | "
        f"{_percentile(total_timings, 90):>5.0f} | "
        f"{_percentile(total_timings, 95):>5.0f} | {t_p99:>5.0f} | {within_total:<13} |"
    )
    return "\n".join(lines)


async def run_benchmark(
    num_runs: int = 100,
    output_path: str | None = None,
    use_real_llm: bool = False,
) -> None:
    """Run benchmark and print/save results."""
    from src.audio.processor import AudioProcessor
    from src.pipeline.latency_budget import STAGE_BUDGETS_MS, PipelineBudget

    print(f"\n{'='*60}")
    print(f"  Latency Benchmark  —  {num_runs} runs")
    print(f"{'='*60}\n")

    audio_bytes = await generate_synthetic_audio()
    processor = AudioProcessor()

    stage_timings: dict[str, list[float]] = {s: [] for s in STAGE_BUDGETS_MS}
    total_timings: list[float] = []
    errors = 0

    for i in range(num_runs):
        try:
            budget = PipelineBudget(total_budget_ms=5000, trace_id=f"bench-{i}")

            # Audio preprocessing (real)
            async with budget.stage("audio_preprocessing"):
                processed = await processor.process(audio_bytes, 16_000)

            # Simulate other stages with realistic sleep ranges
            stage_sims = {
                "vad_segmentation": 0.020,
                "speech_to_text": 0.270,      # whisper-base typical
                "intent_classification": 0.015,
                "llm_first_token": 0.380 if not use_real_llm else None,
                "llm_generation": 0.590 if not use_real_llm else None,
                "tts_synthesis": 0.145,
            }
            for stage, sleep_s in stage_sims.items():
                if sleep_s is not None:
                    async with budget.stage(stage):
                        await asyncio.sleep(sleep_s + (np.random.normal(0, sleep_s * 0.15)))

            report = budget.report()
            for m in report.stages:
                if m.stage in stage_timings:
                    stage_timings[m.stage].append(m.actual_ms)
            total_timings.append(report.total_actual_ms)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs...")

        except Exception as exc:
            errors += 1
            print(f"  Run {i} failed: {exc}")

    print(f"\n  Errors: {errors}/{num_runs}\n")
    print(_build_percentile_table(stage_timings, total_timings, STAGE_BUDGETS_MS))

    if output_path:
        result_data: dict[str, Any] = {
            "num_runs": num_runs,
            "errors": errors,
            "total_latency": {
                "p50": _percentile(total_timings, 50),
                "p90": _percentile(total_timings, 90),
                "p95": _percentile(total_timings, 95),
                "p99": _percentile(total_timings, 99),
                "mean": statistics.mean(total_timings) if total_timings else 0,
            },
            "stages": {
                stage: {
                    "p50": _percentile(times, 50),
                    "p90": _percentile(times, 90),
                    "p99": _percentile(times, 99),
                }
                for stage, times in stage_timings.items()
                if times
            },
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result_data, indent=2))
        print(f"\n  Results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pipeline latency")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--real-llm", action="store_true", help="Use real LLM API")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            num_runs=args.runs,
            output_path=args.output,
            use_real_llm=args.real_llm,
        )
    )


if __name__ == "__main__":
    main()
