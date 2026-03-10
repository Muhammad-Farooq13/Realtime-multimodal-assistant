"""Application configuration via Pydantic-Settings.

All settings can be overridden via environment variables or a .env file.
Validation happens at startup — misconfiguration is caught immediately.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised, validated configuration for the entire application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_max_tokens: int = Field(default=512, ge=1, le=4096)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # ── STT ──────────────────────────────────────────────────────────────────
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = (
        Field(default="base")
    )
    whisper_device: Literal["cpu", "cuda", "auto"] = Field(default="cpu")
    whisper_compute_type: Literal["int8", "float16", "float32"] = Field(default="int8")
    whisper_fallback_model_size: Literal["tiny", "base"] = Field(default="tiny")

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_voice: str = Field(default="en-US-AriaNeural")
    tts_rate: str = Field(default="+0%")
    tts_volume: str = Field(default="+0%")

    # ── Latency budgets (ms) ──────────────────────────────────────────────────
    pipeline_total_budget_ms: int = Field(default=2000, ge=500)
    stt_timeout_ms: int = Field(default=600, ge=100)
    llm_timeout_ms: int = Field(default=1200, ge=200)
    tts_timeout_ms: int = Field(default=400, ge=100)
    vision_timeout_ms: int = Field(default=150, ge=50)

    # ── Circuit Breaker ───────────────────────────────────────────────────────
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    circuit_breaker_success_threshold: int = Field(default=2, ge=1)
    circuit_breaker_recovery_seconds: int = Field(default=30, ge=5)

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    # ── Monitoring ────────────────────────────────────────────────────────────
    metrics_enabled: bool = Field(default=True)
    tracing_enabled: bool = Field(default=False)
    otlp_endpoint: str = Field(default="http://localhost:4317")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(default="json")

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_sample_rate: int = Field(default=16000)
    audio_chunk_duration_ms: int = Field(default=30)
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    max_audio_duration_seconds: int = Field(default=30, ge=1, le=120)

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def stt_timeout_s(self) -> float:
        return self.stt_timeout_ms / 1000.0

    @property
    def llm_timeout_s(self) -> float:
        return self.llm_timeout_ms / 1000.0

    @property
    def tts_timeout_s(self) -> float:
        return self.tts_timeout_ms / 1000.0

    @property
    def vision_timeout_s(self) -> float:
        return self.vision_timeout_ms / 1000.0

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
