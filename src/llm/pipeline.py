"""LLM pipeline — streaming generation via the OpenAI-compatible API.

Supports both:
- OpenAI hosted models (GPT-4o, GPT-4o-mini)
- Local models via Ollama (``LLM_BASE_URL=http://localhost:11434/v1``)

Streaming optimisation
======================
The ``stream()`` method is an async generator that yields tokens as soon as
they arrive from the API. The orchestrator can begin TTS on the first
complete sentence before the full response is generated, reducing perceived
latency by ~400ms.

LLM first-token latency is measured separately from full-generation latency
(see ``STAGE_BUDGETS_MS["llm_first_token"]`` vs ``"llm_generation"]``).
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog

from src.config.settings import get_settings
from src.vision.processor import VisionContext

logger = structlog.get_logger(__name__)
settings = get_settings()


_SYSTEM_PROMPT = """You are a helpful, concise voice assistant.
Respond in 1–3 short sentences unless the user explicitly asks for more detail.
Avoid markdown formatting — your responses will be read aloud."""


class LLMPipeline:
    """Wraps the OpenAI-compatible API for text generation.

    One instance is shared for the application lifetime (client is reused).
    """

    def __init__(self) -> None:
        self._client: "openai.AsyncOpenAI | None" = None

    def _get_client(self) -> "openai.AsyncOpenAI":
        if self._client is None:
            import openai  # lazy import so tests can mock easily
            self._client = openai.AsyncOpenAI(
                api_key=settings.openai_api_key or "ollama",  # ollama doesn't check key
                base_url=settings.llm_base_url,
            )
        return self._client

    async def generate(
        self,
        user_text: str,
        vision_context: VisionContext | None = None,
        history: list[dict] | None = None,
    ) -> str:
        """Generate a complete response (non-streaming).

        Used by the one-shot REST endpoint.
        """
        messages = self._build_messages(user_text, vision_context, history or [])
        client = self._get_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        text = response.choices[0].message.content or ""
        logger.debug("llm_generate_complete", tokens=response.usage.total_tokens if response.usage else 0)
        return text.strip()

    async def stream(
        self,
        user_text: str,
        vision_context: VisionContext | None = None,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Yield tokens as they arrive from the API (streaming mode).

        Used by the WebSocket handler for real-time delivery.

        Yields:
            Individual string tokens (may be subword pieces or full words
            depending on the model's tokeniser chunking).
        """
        messages = self._build_messages(user_text, vision_context, history or [])
        client = self._get_client()

        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        user_text: str,
        vision_context: VisionContext | None,
        history: list[dict],
    ) -> list[dict]:
        """Build the messages array for the Chat Completions API.

        Injects multimodal vision content into the latest user message
        when a ``VisionContext`` is provided.
        """
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Inject conversation history (last 6 turns to stay within context)
        for turn in history[-6:]:
            messages.append(turn)

        # Build user content (text + optional image)
        if vision_context:
            user_content: list[dict] = [
                {"type": "text", "text": user_text},
                vision_context.as_openai_content(),
            ]
        else:
            user_content = [{"type": "text", "text": user_text}]  # type: ignore[assignment]

        messages.append({"role": "user", "content": user_content})
        return messages
