"""
Streamlit demo — Real-Time Multimodal Assistant
Run: streamlit run streamlit_app.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Realtime Multimodal Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_PATH = Path("data/demo_bundle.pkl")


@st.cache_resource(show_spinner="Loading demo bundle …")
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        return {}
    with open(BUNDLE_PATH, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
turns: list[dict] = bundle.get("turns", [])
stage_budgets: dict[str, int] = bundle.get("stage_budgets_ms", {})
total_budget: int = bundle.get("total_budget_ms", 2000)
summary: dict = bundle.get("summary", {})
deg_counts: dict = bundle.get("degradation_counts", {})
intent_counts: dict = bundle.get("intent_counts", {})
stage_stats: dict = bundle.get("stage_stats", {})

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/microphone.png", width=72)
    st.title("🎙️ Multimodal Assistant")
    st.caption("Real-time voice + vision pipeline demo")
    st.divider()

    if summary:
        st.metric("Simulated Turns", f"{summary.get('total_turns', 0):,}")
        st.metric("Within Budget", f"{summary.get('within_budget_pct', 0):.1f}%")
        st.metric("P50 Latency", f"{summary.get('p50_e2e_ms', 0):.0f} ms")
        st.metric("P95 Latency", f"{summary.get('p95_e2e_ms', 0):.0f} ms")
    st.divider()
    st.metric("E2E Budget", f"{total_budget:,} ms")
    st.divider()
    st.caption(
        "Models: Whisper (faster-whisper) · GPT-4o-mini · edge-tts\n"
        "API: FastAPI + WebSocket · Prometheus + OpenTelemetry"
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏗️ Architecture",
    "⏱️ Latency Budget",
    "🔌 Circuit Breaker",
    "⚙️ API & Config",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("System Architecture")
    st.markdown("""
A **real-time multimodal voice assistant** pipeline with explicit latency budget
enforcement, circuit breakers, and graceful degradation.  The entire end-to-end
response targets **≤ 2 000 ms** at p50.
    """)

    if not summary:
        st.warning("Demo bundle not found — run `python demo_bundle.py` first.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("E2E Budget", f"{total_budget:,} ms")
    c2.metric("Within Budget (p50)", f"{summary['within_budget_pct']}%")
    c3.metric("P50 E2E", f"{summary['p50_e2e_ms']:.0f} ms")
    c4.metric("P99 E2E", f"{summary['p99_e2e_ms']:.0f} ms")

    st.divider()

    st.subheader("Pipeline Flow")
    st.markdown("""
```
User speaks         → 🎙️  Audio Capture & VAD
                    ↓
Raw PCM16 audio     → 🔊  Audio Preprocessing  (20 ms budget)
                    ↓
Cleaned audio       → 📝  Speech-to-Text / Whisper  (400 ms budget)
                    ↓
Transcript text  ┬──→ 🧠  Intent Classification  (30 ms budget)
                 │
                 └──→ 🖼️  Vision Processing [optional]  (80 ms budget)
                    ↓
Text + intent       → 🤖  LLM Generation / GPT-4o-mini  (500 ms TTFT + 800 ms gen)
                    ↓
Response text       → 🔈  TTS Synthesis / edge-tts  (200 ms budget)
                    ↓
Audio stream        → 📡  WebSocket → Client
```
    """)

    st.subheader("Streaming Overlap Optimisation")
    st.info(
        "**45 % latency reduction** via pipeline overlapping: "
        "STT partial tokens → LLM generation starts → TTS first sentence begins "
        "before LLM completes. Sequential total would be ~2 210 ms; streaming "
        "pipeline achieves ~1 200 ms p50."
    )

    overlap_data = {
        "Stage": [
            "Audio Preprocessing", "Speech-to-Text", "Intent Classification",
            "LLM First Token", "LLM Generation", "TTS Synthesis",
        ],
        "Sequential Start (ms)": [0, 20, 420, 450, 950, 1750],
        "Streaming Start (ms)": [0, 20, 420, 250, 450, 600],
        "Duration (ms)": [20, 400, 30, 500, 800, 200],
    }
    df_overlap = pd.DataFrame(overlap_data)

    fig_gantt = go.Figure()
    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c", "#1abc9c"]
    for i, (_, row) in enumerate(df_overlap.iterrows()):
        for mode, col in [("Streaming", "Streaming Start (ms)"), ("Sequential", "Sequential Start (ms)")]:
            fig_gantt.add_trace(go.Bar(
                name=f"{row['Stage']} ({mode})",
                x=[row["Duration (ms)"]],
                y=[f"{row['Stage']} ({mode})"],
                base=[row[col]],
                orientation="h",
                marker_color=colors[i] if mode == "Streaming" else f"rgba({','.join(['150']*3)},0.4)",
                showlegend=False,
            ))
    fig_gantt.update_layout(
        barmode="overlay",
        xaxis_title="Time (ms)",
        xaxis=dict(range=[0, 2500]),
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, l=220),
    )
    fig_gantt.add_vline(x=2000, line_color="red", line_dash="dash",
                        annotation_text="2 000 ms budget")
    st.plotly_chart(fig_gantt, use_container_width=True)
    st.caption("Blue bars = streaming pipeline. Grey bars = sequential baseline.")

    st.divider()
    st.subheader("Degradation Strategy")
    col_l, col_m, col_r = st.columns(3)
    with col_l:
        st.info("**FALLBACK_STT**\n\nPrimary Whisper (base) times out → switch to tiny model (2× faster, ~5% WER increase).")
    with col_m:
        st.warning("**PARTIAL_DEGRADATION**\n\nLLM budget exceeded → skip vision context, reduce max_tokens, use previous response cache.")
    with col_r:
        st.error("**TEXT_ONLY_RESPONSE**\n\nTTS circuit breaker OPEN → return text response without audio synthesis.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Latency Budget
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("⏱️ Latency Budget Simulator")
    st.markdown(
        "Explore the **simulated latency distribution** across a 500-turn session. "
        "Each bar represents actual latency vs the budget for that stage."
    )

    if not turns:
        st.warning("Demo bundle not found — run `python demo_bundle.py` first.")
        st.stop()

    # Stage budget bars
    st.subheader("Stage P50 / P95 vs Budget")
    stage_rows = []
    for stage, stats in stage_stats.items():
        stage_rows.append({
            "Stage": stage.replace("_", " ").title(),
            "Budget (ms)": stats["budget_ms"],
            "P50 (ms)": stats["p50_ms"],
            "P95 (ms)": stats["p95_ms"],
            "P99 (ms)": stats["p99_ms"],
            "Over Budget %": stats["over_budget_pct"],
        })
    df_stages = pd.DataFrame(stage_rows)

    fig_stages = go.Figure()
    fig_stages.add_trace(go.Bar(
        name="P50", x=df_stages["Stage"], y=df_stages["P50 (ms)"],
        marker_color="#2ecc71",
    ))
    fig_stages.add_trace(go.Bar(
        name="P95", x=df_stages["Stage"], y=df_stages["P95 (ms)"],
        marker_color="#e67e22",
    ))
    fig_stages.add_trace(go.Bar(
        name="P99", x=df_stages["Stage"], y=df_stages["P99 (ms)"],
        marker_color="#e74c3c",
        opacity=0.5,
    ))
    # Budget line overlay
    for _, row in df_stages.iterrows():
        fig_stages.add_shape(
            type="line",
            x0=row["Stage"], x1=row["Stage"],
            y0=0, y1=row["Budget (ms)"],
            line=dict(color="white", width=3, dash="dot"),
        )
    fig_stages.update_layout(
        barmode="group",
        yaxis_title="Latency (ms)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_stages, use_container_width=True)
    st.caption("Dotted white line = per-stage budget. Bars above budget indicate degradation risk.")

    st.divider()

    # Stage table
    st.subheader("Stage Detail Table")
    st.dataframe(df_stages.style.background_gradient(subset=["Over Budget %"], cmap="RdYlGn_r"),
                 use_container_width=True, hide_index=True)

    st.divider()

    # E2E latency histogram
    st.subheader("End-to-End Latency Distribution (500 turns)")
    e2e_actuals = [t["total_actual_ms"] for t in turns]
    fig_hist = px.histogram(
        x=e2e_actuals,
        nbins=50,
        color_discrete_sequence=["#3498db"],
        labels={"x": "E2E Latency (ms)"},
    )
    fig_hist.add_vline(x=total_budget, line_color="red", line_dash="dash",
                       annotation_text=f"Budget: {total_budget} ms")
    fig_hist.add_vline(x=summary["p50_e2e_ms"], line_color="#2ecc71", line_dash="dot",
                       annotation_text=f"P50: {summary['p50_e2e_ms']} ms")
    fig_hist.add_vline(x=summary["p95_e2e_ms"], line_color="#e67e22", line_dash="dot",
                       annotation_text=f"P95: {summary['p95_e2e_ms']} ms")
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20),
        yaxis_title="Turn Count",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Budget slider simulator
    st.divider()
    st.subheader("Interactive Budget Simulator")
    st.markdown("Adjust the total E2E budget to see how many turns would fall within it.")
    sim_budget = st.slider("E2E Budget (ms)", min_value=500, max_value=4000,
                           value=total_budget, step=50)
    within_sim = sum(1 for v in e2e_actuals if v <= sim_budget)
    pct_sim = within_sim / len(e2e_actuals) * 100

    s1, s2, s3 = st.columns(3)
    s1.metric("Budget", f"{sim_budget:,} ms")
    s2.metric("Turns Within Budget", f"{within_sim} / {len(e2e_actuals)}")
    s3.metric("Success Rate", f"{pct_sim:.1f}%")

    fig_sim = go.Figure(go.Histogram(x=e2e_actuals, nbinsx=60, marker_color="#3498db", name="All turns"))
    fig_sim.add_vrect(x0=0, x1=sim_budget, fillcolor="rgba(46,204,113,0.1)",
                      annotation_text="Within budget", annotation_position="top left")
    fig_sim.add_vline(x=sim_budget, line_color="#2ecc71", line_dash="dash",
                      annotation_text=f"Selected: {sim_budget} ms")
    fig_sim.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20),
        xaxis_title="E2E Latency (ms)",
        yaxis_title="Turn Count",
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Circuit Breaker
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔌 Circuit Breaker Demo")
    st.markdown("""
Each service (STT, LLM, TTS) is protected by a **circuit breaker** that opens
after 5 consecutive failures and recovers after 2 consecutive successes following
a 30-second half-open period.
    """)

    if not turns:
        st.warning("Demo bundle not found — run `python demo_bundle.py` first.")
        st.stop()

    # Degradation over time
    df_turns = pd.DataFrame(turns)

    st.subheader("Degradation Level Over 500 Turns")
    deg_color_map = {
        "NONE": "#2ecc71",
        "PARTIAL_DEGRADATION": "#f39c12",
        "FALLBACK_STT": "#e67e22",
        "TEXT_ONLY_RESPONSE": "#e74c3c",
    }
    df_turns["Degradation Color"] = df_turns["degradation_level"].map(deg_color_map)

    fig_deg_time = px.scatter(
        df_turns,
        x="turn_id",
        y="total_actual_ms",
        color="degradation_level",
        color_discrete_map=deg_color_map,
        size_max=6,
        opacity=0.7,
        labels={"turn_id": "Turn #", "total_actual_ms": "E2E Latency (ms)",
                "degradation_level": "Degradation Level"},
    )
    fig_deg_time.add_hline(y=total_budget, line_color="white", line_dash="dash",
                           annotation_text=f"Budget: {total_budget} ms")
    fig_deg_time.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_deg_time, use_container_width=True)

    st.divider()

    col_deg, col_intent = st.columns(2)

    with col_deg:
        st.subheader("Degradation Distribution")
        if deg_counts:
            # Normalise numpy str keys
            clean_deg = {str(k): int(v) for k, v in deg_counts.items()}
            fig_pie = px.pie(
                names=list(clean_deg.keys()),
                values=list(clean_deg.values()),
                color=list(clean_deg.keys()),
                color_discrete_map=deg_color_map,
                hole=0.4,
            )
            fig_pie.update_layout(margin=dict(t=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_intent:
        st.subheader("Intent Distribution")
        if intent_counts:
            clean_intent = {str(k): int(v) for k, v in intent_counts.items()}
            fig_intent = px.bar(
                x=list(clean_intent.keys()),
                y=list(clean_intent.values()),
                color_discrete_sequence=["#3498db"],
                labels={"x": "Intent", "y": "Count"},
            )
            fig_intent.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_intent, use_container_width=True)

    st.divider()

    # Circuit breaker state machine
    st.subheader("Circuit Breaker State Machine")
    st.markdown("""
```
           ┌──────────────────────────────────┐
           │                                   │
    ┌──────▼──────┐  5 failures  ┌─────────────┴──────┐
    │   CLOSED    │  ──────────► │       OPEN          │
    │  (normal)   │              │  (reject all calls) │
    └─────────────┘              └──────────┬──────────┘
           ▲                                │
           │                        30s timeout
           │                                │
           │ 2 successes          ┌─────────▼──────────┐
           └──────────────────────►     HALF-OPEN       │
                                  │ (probe with 1 call) │
                                  └────────────────────┘
```
    """)

    cb_config = {
        "Failure threshold (→ OPEN)": 5,
        "Success threshold (→ CLOSED)": 2,
        "Recovery window": "30 seconds",
        "Scope": "Per service: STT / LLM / TTS (independent breakers)",
    }
    for k, v in cb_config.items():
        st.markdown(f"- **{k}**: `{v}`")

    st.divider()

    # Interactive CB simulation
    st.subheader("Circuit Breaker Event Timeline (first 100 turns)")
    df_cb = df_turns.head(100).copy()
    cb_timeline = []
    failures = 0
    cb_state = "CLOSED"
    recovery_cd = 0

    for _, row in df_cb.iterrows():
        if cb_state == "OPEN":
            recovery_cd -= 1
            if recovery_cd <= 0:
                cb_state = "HALF-OPEN"
        if row["degradation_level"] in ["PARTIAL_DEGRADATION", "FALLBACK_STT", "TEXT_ONLY_RESPONSE"]:
            failures += 1
            if failures >= 5 and cb_state == "CLOSED":
                cb_state = "OPEN"
                recovery_cd = 15
        else:
            if cb_state == "HALF-OPEN":
                cb_state = "CLOSED"
                failures = 0
            else:
                failures = max(0, failures - 1)
        cb_timeline.append(cb_state)

    df_cb["cb_state"] = cb_timeline
    cb_state_colors = {"CLOSED": "#2ecc71", "OPEN": "#e74c3c", "HALF-OPEN": "#f39c12"}

    fig_cb = go.Figure()
    for state, color in cb_state_colors.items():
        mask = df_cb["cb_state"] == state
        fig_cb.add_trace(go.Bar(
            x=df_cb[mask]["turn_id"],
            y=[1] * mask.sum(),
            name=state,
            marker_color=color,
            width=0.8,
        ))
    fig_cb.update_layout(
        barmode="stack",
        xaxis_title="Turn #",
        yaxis=dict(visible=False),
        height=200,
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20),
        legend=dict(orientation="h", y=1.2),
    )
    st.plotly_chart(fig_cb, use_container_width=True)
    st.caption("Green = CLOSED (healthy), Red = OPEN (rejecting calls), Orange = HALF-OPEN (probing).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — API & Config
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("⚙️ API Reference & Configuration")

    st.subheader("REST Endpoints")
    endpoints = [
        ("GET", "/health", "Liveness probe — always 200 if process alive"),
        ("GET", "/health/ready", "Readiness probe — 200 only when Whisper loaded"),
        ("POST", "/api/v1/audio/transcribe", "Upload WAV → returns transcript JSON"),
        ("POST", "/api/v1/audio/synthesize", "Text → WAV audio bytes"),
        ("POST", "/api/v1/stream/query", "Audio → full turn response (transcript + text + audio)"),
        ("GET", "/metrics", "Prometheus scrape endpoint"),
        ("GET", "/docs", "FastAPI Swagger UI"),
    ]
    df_ep = pd.DataFrame(endpoints, columns=["Method", "Path", "Description"])
    st.dataframe(df_ep, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("WebSocket API")
    st.markdown("""
**Endpoint**: `ws://host:8000/ws/stream`

#### Event stream (server → client)
| Type | Payload | Notes |
|---|---|---|
| `transcript` | `{text, is_final}` | STT result (may be partial) |
| `llm_token` | `{token}` | Individual LLM token for streaming UI |
| `tts_chunk` | `{audio_b64, seq}` | Base64-encoded PCM16 audio chunk |
| `latency_report` | full JSON | Final event — per-stage timing |

#### Sending audio (client → server)
Binary WebSocket frames — PCM16 16kHz mono, in 30ms chunks.
    """)

    st.divider()

    st.subheader("Sample Requests")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Transcribe audio:**")
        st.code("""
curl -X POST http://localhost:8000/api/v1/audio/transcribe \\
  -F "file=@speech.wav" \\
  -H "X-Request-ID: my-request"
# Response:
{
  "text": "What is the weather like today?",
  "confidence": 0.94,
  "language": "en"
}
        """, language="bash")

        st.markdown("**Process a full turn:**")
        st.code("""
curl -X POST http://localhost:8000/api/v1/stream/query \\
  -F "audio=@speech.wav" \\
  -H "X-Request-ID: turn-001"
# Response includes X-Response-Time-Ms header
{
  "transcript": "What is the weather?",
  "response_text": "It's 22°C and sunny today.",
  "audio_bytes": "<base64>",
  "intent": {"name": "weather_query", "confidence": 0.91},
  "latency_report": {
    "total_actual_ms": 847.3,
    "total_budget_ms": 2000,
    "within_budget": true,
    "stages": [...]
  },
  "degradation_level": "NONE",
  "trace_id": "a3f9c12b"
}
        """, language="bash")

    with col_r:
        st.markdown("**Health check:**")
        st.code("""
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0"}

curl http://localhost:8000/health/ready
# {"status": "ready"} (200)  or
# {"status": "not_ready"} (503)
        """, language="bash")

        st.markdown("**Python client:**")
        st.code("""
import asyncio, websockets, json, base64

async def run():
    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as ws:
        # Send raw PCM16 chunks
        with open("speech.wav", "rb") as f:
            chunk = f.read(480)  # 30ms @ 16kHz
            while chunk:
                await ws.send(chunk)
                chunk = f.read(480)

        # Receive streaming events
        async for msg in ws:
            event = json.loads(msg)
            if event["type"] == "tts_chunk":
                audio = base64.b64decode(event["audio_b64"])
                # play audio chunk...
            elif event["type"] == "latency_report":
                print(event)
                break

asyncio.run(run())
        """, language="python")

    st.divider()

    st.subheader("Configuration Reference")
    config_rows = [
        ("OPENAI_API_KEY", "", "OpenAI API key (required)", "Security"),
        ("LLM_MODEL", "gpt-4o-mini", "OpenAI model name", "LLM"),
        ("LLM_MAX_TOKENS", "512", "Max response tokens", "LLM"),
        ("LLM_TEMPERATURE", "0.7", "Sampling temperature", "LLM"),
        ("WHISPER_MODEL_SIZE", "base", "tiny / base / small / medium / large-v3", "STT"),
        ("WHISPER_DEVICE", "cpu", "cpu / cuda / auto", "STT"),
        ("WHISPER_COMPUTE_TYPE", "int8", "int8 / float16 / float32", "STT"),
        ("PIPELINE_TOTAL_BUDGET_MS", "2000", "Hard E2E latency budget", "Budget"),
        ("STT_TIMEOUT_MS", "600", "Whisper per-call timeout", "Budget"),
        ("LLM_TIMEOUT_MS", "1200", "LLM first-token timeout", "Budget"),
        ("TTS_TIMEOUT_MS", "400", "TTS synthesis timeout", "Budget"),
        ("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5", "Failures before OPEN", "Reliability"),
        ("CIRCUIT_BREAKER_RECOVERY_SECONDS", "30", "Recovery window", "Reliability"),
        ("METRICS_ENABLED", "true", "Enable /metrics Prometheus endpoint", "Monitoring"),
        ("TRACING_ENABLED", "false", "Enable OpenTelemetry tracing", "Monitoring"),
        ("OTLP_ENDPOINT", "http://localhost:4317", "OTLP collector endpoint", "Monitoring"),
        ("LOG_FORMAT", "json", "json / console structured logging", "Logging"),
        ("CORS_ORIGINS", '["http://localhost:3000"]', "Allowed CORS origins (JSON list)", "API"),
        ("API_PORT", "8000", "FastAPI listen port", "API"),
    ]
    df_config = pd.DataFrame(config_rows, columns=["Variable", "Default", "Description", "Category"])
    st.dataframe(df_config, use_container_width=True, hide_index=True)

    st.divider()

    col_docker, col_compose = st.columns(2)

    with col_docker:
        st.subheader("Docker")
        st.code("""
# Build
docker build -t multimodal-assistant:latest .

# Run
docker run -d \\
  -p 8000:8000 \\
  -e OPENAI_API_KEY=$OPENAI_API_KEY \\
  -e WHISPER_MODEL_SIZE=base \\
  multimodal-assistant:latest
        """, language="bash")

    with col_compose:
        st.subheader("Docker Compose (with monitoring)")
        st.code("""
# Start full stack
docker compose up -d

# Includes:
#  - FastAPI app  (port 8000)
#  - Prometheus   (port 9090)
#  - Grafana      (port 3000)

# Tail logs
docker compose logs -f app
        """, language="bash")

    st.divider()
    st.subheader("Project Stack")
    stack_rows = [
        ("FastAPI 0.110+", "Async REST + WebSocket API"),
        ("faster-whisper", "CTranslate2-backed Whisper (2-4× faster than openai-whisper)"),
        ("OpenAI GPT-4o-mini", "LLM generation with streaming"),
        ("edge-tts", "Microsoft Neural TTS (free, no API key)"),
        ("Pydantic v2", "Settings validation + data models"),
        ("structlog", "Structured JSON logging"),
        ("Prometheus + OTel", "Metrics + distributed tracing"),
        ("pytest-asyncio", "Async test suite (76 tests)"),
    ]
    df_stack = pd.DataFrame(stack_rows, columns=["Component", "Role"])
    st.dataframe(df_stack, use_container_width=True, hide_index=True)

    st.caption("Built with FastAPI · faster-whisper · edge-tts · Streamlit · Plotly · GitHub Actions")
