"""Vision processor — handles image frame analysis for multimodal context.

For GPT-4o / Claude 3 vision models, we encode the image as a base64 data URI.
For text-only models, we run a local caption extraction and inject the caption
text into the LLM prompt instead.

Processing pipeline
===================
raw JPEG/PNG bytes
 → validate & resize (max 1024×1024, preserve aspect)
 → convert to RGB
 → base64 encode
 → VisionContext (holds both original bytes and base64 string)
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class VisionContext:
    """Everything the LLM stage needs about an attached image.

    Attributes:
        base64_image:  Base64-encoded JPEG (for vision LLM APIs).
        mime_type:     MIME type string (image/jpeg, image/png).
        width:         Image width after resizing.
        height:        Image height after resizing.
        caption:       Optional caption (for text-only LLM fallback).
    """

    base64_image: str
    mime_type: Literal["image/jpeg", "image/png"] = "image/jpeg"
    width: int = 0
    height: int = 0
    caption: str = ""

    def as_openai_content(self) -> dict:
        """Format as an OpenAI chat vision content item."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.mime_type};base64,{self.base64_image}",
                "detail": "auto",
            },
        }


class VisionProcessor:
    """Validates, resizes, and encodes image frames for multimodal LLM input.

    Pipeline stages this processor targets: ``vision_processing`` (80ms budget).
    """

    MAX_DIMENSION = 1024   # pixels — cap both width and height
    JPEG_QUALITY = 85

    async def process(self, image_bytes: bytes) -> VisionContext | None:
        """Process raw image bytes into a ``VisionContext``.

        Returns ``None`` if the bytes are not a valid image.
        """
        try:
            return self._process_sync(image_bytes)
        except Exception as exc:
            logger.warning("vision_processing_failed", error=str(exc))
            return None

    # ── Private ───────────────────────────────────────────────────────────────

    def _process_sync(self, image_bytes: bytes) -> VisionContext:
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")

        # Resize if needed
        w, h = img.size
        if w > self.MAX_DIMENSION or h > self.MAX_DIMENSION:
            scale = self.MAX_DIMENSION / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        final_w, final_h = img.size

        # Encode as JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.JPEG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return VisionContext(
            base64_image=b64,
            mime_type="image/jpeg",
            width=final_w,
            height=final_h,
        )
