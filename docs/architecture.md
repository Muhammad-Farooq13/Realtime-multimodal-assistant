# Architecture Decision Record — Real-Time Multimodal Assistant

## Context

We need a voice assistant that can respond within **2 seconds E2E** at p50,
accepts optional image frames, and degrades gracefully when any component is slow.

---

## System Architecture

```
                         ┌──────────────────────────────┐
                         │         CLIENT LAYER          │
                         │  Browser / Mobile App / CLI   │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────┼──────────────────┐
                    │   WebSocket       │    REST (HTTP)    │
                    │  ws:///ws/stream  │   /api/v1/*       │
                    └───────────────────┼──────────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │       FastAPI Gateway         │
                         │  Middleware: CORS, ReqID,     │
                         │  Latency header, Structlog    │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │   Pipeline Orchestrator       │
                         │                               │
                         │  ┌──────────────────────┐    │
                         │  │   PipelineBudget      │    │
                         │  │   (2000ms total SLA)  │    │
                         │  └──────────────────────┘    │
                         │                               │
          ┌──────────────┼──────────────┐                │
          │              │              │                │
   ┌──────▼─────┐  ┌─────▼──────┐  ┌──▼──────────┐     │
   │   Audio    │  │  Vision    │  │    LLM       │     │
   │ Processor  │  │ Processor  │  │  Pipeline    │     │
   │  (VAD+     │  │ (PIL/JPEG) │  │  (OpenAI /  │     │
   │   norm)    │  │            │  │   Ollama)    │     │
   └──────┬─────┘  └────────────┘  └──────┬───────┘     │
          │                                │             │
   ┌──────▼─────┐          ┌───────────────▼──────┐      │
   │    STT     │          │       TTS            │      │
   │  (faster-  │          │  (edge-tts neural)   │      │
   │  whisper)  │          │                      │      │
   └────────────┘          └──────────────────────┘      │
                         └──────────────────────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │       Monitoring              │
                         │  Prometheus / Grafana / OTEL  │
                         └───────────────────────────────┘
```

---

## Key Design Decisions

### 1. Explicit Latency Budget (not just "make it fast")

**Decision:** Every stage declares a budget in `STAGE_BUDGETS_MS`. The
`PipelineBudget` context manager measures wall-clock time per stage and records
overruns to Prometheus.

**Rationale:** Without explicit budgets, performance regressions are invisible
until users complain. By making budgets first-class, we can detect the exact
stage causing SLA violations from a single Grafana alert.

**Trade-off:** Adds ~0.1ms overhead per stage (perf_counter calls). Acceptable.

---

### 2. Streaming Pipeline (STT → LLM → TTS overlap)

**Decision:** TTS begins synthesising from the first complete LLM sentence,
before LLM generation completes. STT partial output is fed to LLM before
transcription is final.

**Rationale:** Sequential execution would have ~2.2s E2E. Overlapping reduces
this to ~1.2s — a 45% reduction in *perceived* latency.

**Trade-off:** More complex error handling (must cancel downstream tasks on
upstream failure). Handled via `asyncio.Task` cancellation.

---

### 3. Circuit Breaker per Service

**Decision:** Each of STT, LLM, TTS gets its own `CircuitBreaker` instance.

**Rationale:** Prevents cascading failures. A degraded TTS provider should not
affect STT quality — they're independent. Three-state machine (CLOSED/OPEN/
HALF-OPEN) allows automatic recovery without operator intervention.

**Trade-off:** Circuit breakers introduce a false-positive risk (brief spike
triggers OPEN, legitimate requests are fast-failed). Mitigated by setting
`failure_threshold=5` and `recovery_seconds=30`.

---

### 4. Single-Model STT with Fallback Swap

**Decision:** Primary model is configurable (`WHISPER_MODEL_SIZE`). If STT
exceeds budget, the orchestrator downgrades to `tiny` for that turn.

**Rationale:** User-facing latency matters more than accuracy during periods
of high CPU load. A fast, slightly less accurate response beats a perfect
response that arrives 3 seconds late.

**Trade-off:** Fallback model quality is materially lower (tiny WER ~12% vs
base WER ~6%). This is explicitly documented and surfaced in the API response's
`degradation_level` field so clients can show a warning icon.

---

### 5. Dependency Injection via App State

**Decision:** Services are instantiated once in the lifespan handler and
attached to `app.state`. Routes receive them via `Depends()` or `request.app.state`.

**Rationale:** Ensures Whisper models are loaded once at startup (not per-request).
Makes services mockable in tests without monkey-patching.

---

## Rejected Alternatives

| Alternative | Reason Rejected |
|-------------|-----------------|
| gRPC instead of WebSocket | Browser clients require WebSocket; gRPC adds client complexity |
| Local Whisper.cpp (C++) | faster-whisper (CTranslate2) gives equivalent speed in pure Python |
| Message queue (Redis/Kafka) | Adds infra complexity; asyncio is sufficient for single-node |
| LangChain | Too much abstraction for a latency-critical path; prefer direct API calls |
