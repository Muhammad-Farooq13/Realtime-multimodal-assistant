"""WebSocket handler — real-time bidirectional streaming pipeline.

Protocol summary
================
Client sends JSON messages (see README for full schema).
Server yields JSON events as pipeline stages complete.

Connection lifecycle
====================
1. Client connects → server sends ``{"type": "ready"}``
2. Client sends ``audio_chunk`` messages (30ms PCM16 chunks, base64-encoded)
3. Client sends optional ``image_frame`` messages
4. Client sends ``end_of_speech`` → server begins pipeline execution
5. Server streams: ``transcript`` → ``llm_token``s → ``tts_chunk``s → ``latency_report``
6. Client may send ``interrupt`` at any time to cancel the current response
7. Process repeats for next utterance; connection stays alive

Timeout handling
================
If no message is received for 30 seconds, the server closes with code 1001.
Each pipeline turn runs under the global budget enforced by ``PipelineOrchestrator``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from src.pipeline.orchestrator import PipelineOrchestrator, TurnRequest

logger = structlog.get_logger(__name__)
router = APIRouter()

IDLE_TIMEOUT_SECONDS = 30.0


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, request: Request):
    """Main WebSocket endpoint for real-time multimodal streaming."""
    await websocket.accept()
    orchestrator: PipelineOrchestrator = request.app.state.orchestrator
    connection_id = str(uuid.uuid4())[:8]
    log = logger.bind(connection_id=connection_id)

    await _send(websocket, {"type": "ready", "connection_id": connection_id})
    log.info("ws_connected")

    # Per-connection buffers
    audio_chunks: list[bytes] = []
    image_bytes: bytes | None = None
    current_task: asyncio.Task | None = None

    try:
        while True:
            # Wait for next message with idle timeout
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=IDLE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                log.info("ws_idle_timeout")
                await websocket.close(code=1001, reason="Idle timeout")
                return

            message = _parse_message(raw)
            if message is None:
                await _send(websocket, {"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = message.get("type")

            if msg_type == "audio_chunk":
                chunk = _decode_base64(message.get("data", ""))
                if chunk:
                    audio_chunks.append(chunk)

            elif msg_type == "image_frame":
                image_bytes = _decode_base64(message.get("data", ""))

            elif msg_type == "end_of_speech":
                if not audio_chunks:
                    await _send(websocket, {"type": "error", "message": "No audio received"})
                    continue

                # Cancel any ongoing response (barge-in)
                if current_task and not current_task.done():
                    current_task.cancel()

                combined_audio = b"".join(audio_chunks)
                audio_chunks.clear()
                captured_image = image_bytes
                image_bytes = None

                turn_request = TurnRequest(
                    audio_bytes=combined_audio,
                    image_bytes=captured_image,
                    trace_id=str(uuid.uuid4())[:8],
                )

                current_task = asyncio.create_task(
                    _stream_turn(websocket, orchestrator, turn_request, log)
                )

            elif msg_type == "interrupt":
                if current_task and not current_task.done():
                    current_task.cancel()
                    await _send(websocket, {"type": "interrupted"})
                audio_chunks.clear()

            else:
                await _send(
                    websocket,
                    {"type": "error", "message": f"Unknown message type: {msg_type}"},
                )

    except WebSocketDisconnect:
        log.info("ws_disconnected")
    except Exception as exc:
        log.error("ws_error", error=str(exc))
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass
    finally:
        if current_task and not current_task.done():
            current_task.cancel()


async def _stream_turn(
    websocket: WebSocket,
    orchestrator: PipelineOrchestrator,
    request: TurnRequest,
    log,
) -> None:
    """Run pipeline and stream events to the WebSocket client."""
    try:
        async for event in orchestrator.stream_turn(request):
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            await _send(websocket, event)
    except asyncio.CancelledError:
        log.info("turn_cancelled", trace_id=request.trace_id)
    except Exception as exc:
        log.error("turn_error", error=str(exc), trace_id=request.trace_id)
        await _send(
            websocket,
            {"type": "error", "message": "Pipeline error", "trace_id": request.trace_id},
        )


async def _send(websocket: WebSocket, data: dict[str, Any]) -> None:
    """Send a JSON message, ignoring errors on closed connections."""
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps(data))
    except Exception:
        pass


def _parse_message(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _decode_base64(encoded: str) -> bytes | None:
    try:
        return base64.b64decode(encoded)
    except Exception:
        return None
