"""
demo_bundle.py — Generates data/demo_bundle.pkl for the Streamlit demo app.

Simulates 500 pipeline turns with realistic latency distributions and circuit
breaker behaviour.  No API keys or ML models required.
Run: python demo_bundle.py
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

# ── Stage budget definitions (from src/pipeline/latency_budget.py) ─────────
STAGE_BUDGETS_MS: dict[str, int] = {
    "audio_preprocessing": 20,
    "speech_to_text": 400,
    "intent_classification": 30,
    "llm_first_token": 500,
    "llm_generation": 800,
    "tts_synthesis": 200,
}

TOTAL_BUDGET_MS = 2000

LATENCY_PROFILES = {
    "healthy": {
        "audio_preprocessing": (15, 4),
        "speech_to_text": (300, 60),
        "intent_classification": (22, 5),
        "llm_first_token": (400, 80),
        "llm_generation": (650, 120),
        "tts_synthesis": (150, 30),
    },
    "degraded": {
        "audio_preprocessing": (18, 5),
        "speech_to_text": (480, 90),     # over 400 ms budget
        "intent_classification": (28, 6),
        "llm_first_token": (700, 150),   # over 500 ms budget
        "llm_generation": (900, 200),    # over 800 ms budget
        "tts_synthesis": (180, 40),
    },
}

DEGRADATION_LEVELS = ["NONE", "PARTIAL_DEGRADATION", "FALLBACK_STT", "TEXT_ONLY_RESPONSE"]

INTENTS = [
    "weather_query", "music_control", "navigation", "timer_reminder",
    "general_qa", "smart_home", "calendar", "search",
]

SAMPLE_TRANSCRIPTS = [
    "What's the weather like in London tomorrow?",
    "Play some jazz music please.",
    "Set a timer for 20 minutes.",
    "Navigate to the nearby coffee shop.",
    "What is the capital of Japan?",
    "Turn off the living room lights.",
    "Schedule a meeting with John for tomorrow at 3pm.",
    "Search for Python tutorials on YouTube.",
    "How long does it take to fly to New York?",
    "What's the current Bitcoin price?",
]


def simulate_turns(n: int = 500) -> list[dict]:
    turns = []
    cb_failures = 0           # consecutive failures for circuit breaker sim
    cb_open = False
    cb_recovery_countdown = 0

    for i in range(n):
        # Circuit breaker simulation
        if cb_open:
            cb_recovery_countdown -= 1
            if cb_recovery_countdown <= 0:
                cb_open = False
                cb_failures = 0

        is_degraded = rng.random() < 0.12
        profile_name = "degraded" if (is_degraded and not cb_open) else "healthy"
        profile = LATENCY_PROFILES[profile_name]

        stage_results = {}
        total_actual = 0.0
        any_over_budget = False

        for stage, budget_ms in STAGE_BUDGETS_MS.items():
            mean_ms, std_ms = profile[stage]
            actual = float(rng.normal(mean_ms, std_ms))
            actual = max(5.0, actual)
            over = actual > budget_ms
            if over:
                any_over_budget = True
            stage_results[stage] = {
                "budget_ms": budget_ms,
                "actual_ms": round(actual, 1),
                "over_budget": over,
                "overage_ms": round(max(0.0, actual - budget_ms), 1),
            }
            total_actual += actual

        within_budget = total_actual <= TOTAL_BUDGET_MS

        # Circuit breaker logic
        if profile_name == "degraded":
            cb_failures += 1
        else:
            cb_failures = max(0, cb_failures - 1)

        if cb_failures >= 5 and not cb_open:
            cb_open = True
            cb_recovery_countdown = int(rng.integers(8, 20))

        # Degradation level
        if cb_open:
            deg_level = "TEXT_ONLY_RESPONSE"
        elif any_over_budget:
            deg_level = rng.choice(["PARTIAL_DEGRADATION", "FALLBACK_STT"])
        else:
            deg_level = "NONE"

        turns.append({
            "turn_id": i + 1,
            "transcript": rng.choice(SAMPLE_TRANSCRIPTS),
            "intent": rng.choice(INTENTS),
            "total_actual_ms": round(total_actual, 1),
            "total_budget_ms": TOTAL_BUDGET_MS,
            "within_budget": within_budget,
            "degradation_level": deg_level,
            "circuit_breaker_open": cb_open,
            "stages": stage_results,
        })

    return turns


def main() -> None:
    print("Simulating 500 pipeline turns …")
    turns = simulate_turns(500)

    # Aggregate statistics
    actuals = [t["total_actual_ms"] for t in turns]
    within = sum(1 for t in turns if t["within_budget"])
    deg_counts: dict[str, int] = {}
    for t in turns:
        d = t["degradation_level"]
        deg_counts[d] = deg_counts.get(d, 0) + 1

    intent_counts: dict[str, int] = {}
    for t in turns:
        intent_counts[t["intent"]] = intent_counts.get(t["intent"], 0) + 1

    stage_stats: dict[str, dict] = {}
    for stage in STAGE_BUDGETS_MS:
        stage_actuals = [t["stages"][stage]["actual_ms"] for t in turns]
        over = sum(1 for t in turns if t["stages"][stage]["over_budget"])
        stage_stats[stage] = {
            "budget_ms": STAGE_BUDGETS_MS[stage],
            "p50_ms": round(float(np.percentile(stage_actuals, 50)), 1),
            "p95_ms": round(float(np.percentile(stage_actuals, 95)), 1),
            "p99_ms": round(float(np.percentile(stage_actuals, 99)), 1),
            "over_budget_pct": round(over / len(turns) * 100, 1),
        }

    bundle = {
        "turns": turns,
        "stage_budgets_ms": STAGE_BUDGETS_MS,
        "total_budget_ms": TOTAL_BUDGET_MS,
        "summary": {
            "total_turns": len(turns),
            "within_budget_pct": round(within / len(turns) * 100, 1),
            "p50_e2e_ms": round(float(np.percentile(actuals, 50)), 1),
            "p95_e2e_ms": round(float(np.percentile(actuals, 95)), 1),
            "p99_e2e_ms": round(float(np.percentile(actuals, 99)), 1),
            "mean_e2e_ms": round(float(np.mean(actuals)), 1),
        },
        "degradation_counts": deg_counts,
        "intent_counts": intent_counts,
        "stage_stats": stage_stats,
    }

    out = Path("data/demo_bundle.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nBundle saved → {out}")
    print(f"  Turns: {len(turns)}")
    print(f"  Within budget: {bundle['summary']['within_budget_pct']}%")
    print(f"  P50 E2E: {bundle['summary']['p50_e2e_ms']} ms")
    print(f"  P95 E2E: {bundle['summary']['p95_e2e_ms']} ms")
    print(f"  P99 E2E: {bundle['summary']['p99_e2e_ms']} ms")
    print(f"  Degradation: {deg_counts}")


if __name__ == "__main__":
    main()
