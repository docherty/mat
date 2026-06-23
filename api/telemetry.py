"""In-process request telemetry for mat-serve (metrics + recent routing decisions)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from threading import Lock

_MAX_RECENT = 50


@dataclass
class RequestRecord:
    ts: float
    tier: str
    connector_id: str
    model_id: str | None
    stages: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    passed: bool
    stream: bool = False


@dataclass
class MetricsSnapshot:
    requests_total: int = 0
    requests_failed: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    cost_usd_total: float = 0.0
    latency_ms_sum: float = 0.0
    latency_ms_max: float = 0.0
    started_at: float = field(default_factory=time.time)


class Telemetry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics = MetricsSnapshot()
        self._recent: deque[RequestRecord] = deque(maxlen=_MAX_RECENT)

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            m = self._metrics
            m.requests_total += 1
            if not rec.passed:
                m.requests_failed += 1
            m.input_tokens_total += rec.input_tokens
            m.output_tokens_total += rec.output_tokens
            m.cost_usd_total += rec.cost_usd
            m.latency_ms_sum += rec.latency_ms
            m.latency_ms_max = max(m.latency_ms_max, rec.latency_ms)
            self._recent.appendleft(rec)

    def snapshot(self) -> dict:
        with self._lock:
            m = self._metrics
            n = m.requests_total or 1
            uptime = time.time() - m.started_at
            return {
                "uptime_sec": round(uptime, 1),
                "requests_total": m.requests_total,
                "requests_failed": m.requests_failed,
                "input_tokens_total": m.input_tokens_total,
                "output_tokens_total": m.output_tokens_total,
                "cost_usd_total": round(m.cost_usd_total, 6),
                "latency_ms_avg": round(m.latency_ms_sum / n, 1),
                "latency_ms_max": round(m.latency_ms_max, 1),
            }

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = list(self._recent)[:limit]
        return [asdict(r) for r in rows]


# Shared singleton for mat-serve process.
telemetry = Telemetry()
