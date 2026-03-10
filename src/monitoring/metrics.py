"""Prometheus metrics registry for the multimodal pipeline.

Metrics exported
================
Histogram:
  pipeline_stage_duration_seconds{stage, status}
    — Per-stage latency, labelled as "ok" or "over_budget"

Histogram:
  pipeline_turn_duration_seconds{degradation_level}
    — Full turn E2E latency

Counter:
  pipeline_stage_timeouts_total{stage}
    — Count of stage timeouts

Counter:
  pipeline_degradations_total{level}
    — Count of degradation events by severity

Gauge:
  circuit_breaker_state{service}
    — 0=closed, 1=half_open, 2=open

Scrape endpoint: GET /metrics  (Prometheus text format)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from src.pipeline.latency_budget import PipelineReport
from src.pipeline.timeout_handler import DegradationLevel


class PipelineMetrics:
    """Singleton Prometheus metrics registry.

    Usage::

        metrics = PipelineMetrics.instance()
        metrics.record_turn(report, DegradationLevel.NONE)
    """

    _instance: PipelineMetrics | None = None

    def __init__(self) -> None:
        self.stage_duration = Histogram(
            "pipeline_stage_duration_seconds",
            "Per-stage latency in seconds",
            labelnames=["stage", "status"],
            buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        )
        self.turn_duration = Histogram(
            "pipeline_turn_duration_seconds",
            "End-to-end turn latency in seconds",
            labelnames=["degradation_level"],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
        )
        self.stage_timeouts = Counter(
            "pipeline_stage_timeouts_total",
            "Number of stage timeouts",
            labelnames=["stage"],
        )
        self.degradations = Counter(
            "pipeline_degradations_total",
            "Number of pipeline degradation events",
            labelnames=["level"],
        )
        self.circuit_state = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state: 0=closed 1=half_open 2=open",
            labelnames=["service"],
        )

    @classmethod
    def instance(cls) -> PipelineMetrics:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_turn(
        self, report: PipelineReport, degradation: DegradationLevel
    ) -> None:
        """Record all metrics for a completed pipeline turn."""
        _STATE_MAP = {"closed": 0, "half_open": 1, "open": 2}

        for stage_m in report.stages:
            status = "over_budget" if stage_m.over_budget else "ok"
            self.stage_duration.labels(stage=stage_m.stage, status=status).observe(
                stage_m.actual_ms / 1000.0
            )
            if stage_m.over_budget:
                self.stage_timeouts.labels(stage=stage_m.stage).inc()

        self.turn_duration.labels(degradation_level=degradation.value).observe(
            report.total_actual_ms / 1000.0
        )

        if degradation != DegradationLevel.NONE:
            self.degradations.labels(level=degradation.value).inc()

    def update_circuit_states(self) -> None:
        """Sync circuit breaker gauges from the registry."""
        from src.pipeline.circuit_breaker import CircuitBreakerRegistry

        _STATE_MAP = {"closed": 0, "half_open": 1, "open": 2}
        for status in CircuitBreakerRegistry.instance().all_statuses():
            self.circuit_state.labels(service=status["service"]).set(
                _STATE_MAP.get(status["state"], 0)
            )
