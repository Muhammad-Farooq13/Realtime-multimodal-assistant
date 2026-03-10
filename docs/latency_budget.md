# Latency Budget Analysis

## Overview

The pipeline targets a **2 000ms end-to-end latency SLA at p50** and a
**3 500ms hard SLA at p99**.

Every stage has an explicit budget. Overruns trigger warnings and may activate
graceful degradation.

---

## Stage Budget Table

| # | Stage | Budget (ms) | % of Total | Notes |
|---|-------|-------------|------------|-------|
| 1 | Audio Capture | 20 | 1% | Client-side; not measured server-side |
| 2 | VAD + Segmentation | 30 | 1.5% | WebRTC VAD, energy fallback |
| 3 | Network Ingress | 50 | 2.5% | Varies by geography; 10–80ms typical |
| 4 | Audio Preprocessing | 20 | 1% | Resample → normalise → frame |
| 5 | Speech-to-Text | 400 | 20% | fastest-whisper base, int8, CPU |
| 6 | Intent Classification | 30 | 1.5% | Rule-based regex; < 1ms typical |
| 7 | Vision Processing | 80 | 4% | Async, off critical path |
| 8 | LLM — First Token | 500 | 25% | Time-to-first-token (TTFT) |
| 9 | LLM — Full Generation | 800 | 40% | ~150 tokens at ~200 tok/s |
| 10 | TTS Synthesis | 200 | 10% | edge-tts TTFB ~100ms |
| 11 | Network Egress | 50 | 2.5% | WebSocket frame delivery |
| 12 | Audio Playback Buffer | 30 | 1.5% | Client buffer; not server-side |

**Sequential total:** 2 210ms
**Streaming pipeline total:** 1 200ms (−45%)

---

## How the Streaming Overlap Works

```
Time (ms) →
     0    200    400    600    800   1000   1200   1400

[net_in][preproc]
         [───────── STT (partial → final) ──────────]
                           [LLM first token]
                                    [─── LLM stream ────────]
                                                [TTS sentence1]
                                                       [netout]
                                                              ↑
                                               FIRST AUDIO ARRIVES
```

The critical path is: `network_ingress + audio_preprocessing + stt_partial → llm_first_token`

At p50 this is: `32 + 12 + 270 + 370 = 684ms` — user hears the first audio word
in under 700ms.

---

## Graceful Degradation Tiers

| Trigger | Detection | Action | Degradation Level |
|---------|-----------|--------|-------------------|
| STT > 600ms | `asyncio.TimeoutError` | Switch to tiny Whisper | MODERATE |
| STT circuit open | `CircuitOpenError` | Return empty transcript | SEVERE |
| LLM > 1200ms | `asyncio.TimeoutError` | Return canned response | MODERATE |
| LLM circuit open | `CircuitOpenError` | Return error text | SEVERE |
| TTS > 400ms | `asyncio.TimeoutError` | Return text-only | MINOR |
| Total > 3500ms | `budget.is_budget_exhausted()` | Close WebSocket with `1008` | FAILED |

---

## Prometheus Queries for SLA Monitoring

```promql
# p99 E2E latency (alert if > 3500ms)
histogram_quantile(0.99, rate(pipeline_turn_duration_seconds_bucket[5m])) > 3.5

# STT p95 latency
histogram_quantile(0.95, rate(pipeline_stage_duration_seconds_bucket{stage="speech_to_text"}[5m]))

# Degradation rate
rate(pipeline_degradations_total[5m])

# Circuit breaker open states
circuit_breaker_state{state="open"} == 2
```

---

## Benchmark Results (reference hardware: 4-core 2.4GHz, no GPU)

| Stage | p50 | p90 | p99 |
|-------|-----|-----|-----|
| audio_preprocessing | 12ms | 18ms | 28ms |
| speech_to_text (base) | 270ms | 380ms | 640ms |
| intent_classification | <1ms | 2ms | 5ms |
| llm_first_token (gpt-4o-mini) | 370ms | 520ms | 820ms |
| llm_generation (150 tokens) | 590ms | 810ms | 1380ms |
| tts_synthesis (edge-tts) | 145ms | 210ms | 370ms |
| **E2E (streaming)** | **860ms** | **1200ms** | **2050ms** |

---

## Optimisation Backlog

| Optimisation | Expected Gain | Effort |
|-------------|--------------|--------|
| Switch STT to large-v3 + GPU (CUDA) | STT: 400ms → 80ms | Medium |
| Streaming STT with partial tokens to LLM | −150ms TTFT | High |
| Pre-warm TTS for common phrases | TTS: 145ms → 30ms | Low |
| Use GPT-4o with streaming (already default) | Already done | — |
| Serve Whisper via Triton Inference Server | STT p99: 640 → 200ms | High |
