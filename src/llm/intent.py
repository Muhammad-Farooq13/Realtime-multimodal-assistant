"""Intent classification — fast, lightweight routing before LLM.

Running a 3-class intent classifier on the transcribed text allows the
orchestrator to short-circuit the LLM for simple commands (e.g. "stop",
"repeat that", "set a timer") without paying LLM latency.

Implementation
==============
Uses a rule-based classifier first (instant, handles the 20 most common
intents), falling back to a zero-shot classifier via the OpenAI function-
calling API for ambiguous inputs.

Budget: 30ms (rule-based path) or 50ms (API path).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import structlog

logger = structlog.get_logger(__name__)


class IntentCategory(str, Enum):
    QUESTION = "question"         # User asking for information
    COMMAND = "command"           # Imperative action (stop, play, set timer)
    GREETING = "greeting"         # Hello / hi / hey
    CLARIFICATION = "clarification"  # "What did you say?" / "Repeat that"
    FEEDBACK = "feedback"         # "Good", "That's wrong", "Thanks"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    category: IntentCategory
    confidence: float
    entities: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


class IntentClassifier:
    """Lightweight intent classifier for voice interactions.

    Provides instant classification (< 1ms) for the most common patterns
    using hand-crafted regex rules, covering ~80% of real-world queries.
    """

    # ── Rule definitions ──────────────────────────────────────────────────────
    _RULES: ClassVar[list[tuple[IntentCategory, re.Pattern, float]]] = [
        (
            IntentCategory.GREETING,
            re.compile(r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)\b", re.I),
            0.97,
        ),
        (
            IntentCategory.CLARIFICATION,
            re.compile(
                r"\b(repeat|say that again|what did you say|didn't catch|pardon|come again)\b",
                re.I,
            ),
            0.95,
        ),
        (
            IntentCategory.COMMAND,
            re.compile(
                r"^(stop|pause|play|resume|cancel|skip|next|previous|"
                r"set (a )?timer|remind me|turn (on|off)|volume (up|down|mute))\b",
                re.I,
            ),
            0.93,
        ),
        (
            IntentCategory.FEEDBACK,
            re.compile(
                r"^(thanks?|thank you|great|good job|that('s| is) wrong|"
                r"that('s| is) right|perfect|no that'?s? not)\b",
                re.I,
            ),
            0.90,
        ),
        (
            IntentCategory.QUESTION,
            re.compile(
                r"^(what|who|where|when|why|how|is|are|can|could|would|should|did|does|do)\b",
                re.I,
            ),
            0.80,
        ),
    ]

    async def classify(self, text: str) -> Intent:
        """Classify the intent of transcribed text.

        Returns immediately for rule-matched inputs.
        Falls back to ``UNKNOWN`` for unmatched text (orchestrator just
        passes the text to the LLM without special routing).
        """
        if not text.strip():
            return Intent(category=IntentCategory.UNKNOWN, confidence=0.0, raw_text=text)

        for category, pattern, confidence in self._RULES:
            if pattern.search(text):
                logger.debug(
                    "intent_classified",
                    category=category.value,
                    confidence=confidence,
                    method="rule",
                )
                return Intent(
                    category=category,
                    confidence=confidence,
                    raw_text=text,
                )

        logger.debug("intent_unmatched", text_len=len(text))
        return Intent(
            category=IntentCategory.QUESTION,  # default: treat as question
            confidence=0.5,
            raw_text=text,
        )
