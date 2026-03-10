"""Stream REST endpoint — one-shot multimodal query (audio + optional image)."""

from __future__ import annotations

import base64
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.config.settings import Settings, get_settings
from src.pipeline.orchestrator import PipelineOrchestrator, TurnRequest

router = APIRouter()


def get_orchestrator(request: Request) -> PipelineOrchestrator:
    return request.app.state.orchestrator


class QueryResponse(BaseModel):
    transcript: str
    response_text: str
    audio_base64: str | None
    intent: str | None
    latency_report: dict
    degradation_level: str
    trace_id: str


@router.post("/query", response_model=QueryResponse, summary="One-shot multimodal query")
async def multimodal_query(
    audio: Annotated[UploadFile, File(description="PCM16/WAV audio of user utterance")],
    image: Annotated[UploadFile | None, File(description="Optional image frame")] = None,
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_settings),
):
    """Process a complete multimodal turn (audio + optional image).

    This is the synchronous (non-streaming) REST equivalent of the WebSocket
    ``/ws/stream`` endpoint. Useful for batch processing and simple clients.

    Returns the transcript, LLM response text, synthesised audio (base64),
    and a full latency report broken down by pipeline stage.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    image_bytes: bytes | None = None
    if image is not None:
        image_bytes = await image.read()

    request = TurnRequest(
        audio_bytes=audio_bytes,
        sample_rate=settings.audio_sample_rate,
        image_bytes=image_bytes,
    )

    response = await orchestrator.run_turn(request)

    audio_b64: str | None = None
    if response.audio_bytes:
        audio_b64 = base64.b64encode(response.audio_bytes).decode()

    return QueryResponse(
        transcript=response.transcript,
        response_text=response.response_text,
        audio_base64=audio_b64,
        intent=response.intent.category.value if response.intent else None,
        latency_report=response.latency_report,
        degradation_level=response.degradation_level.value,
        trace_id=response.trace_id,
    )
