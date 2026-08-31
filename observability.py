"""Privacy-safe, bounded observability for the local VoltForge AI service.

This module deliberately records operational facts rather than request data.
It never accepts prompts, project contents, memory text, model text, or user
identifiers as metric values.  Labels are allowlisted and all latency samples
are held in small bounded windows so telemetry cannot become an unbounded data
store or a second workload.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import logging
import os
import re
import threading
import time
import tracemalloc
from typing import Any


logger = logging.getLogger("voltforge-ai.observability")

SCHEMA_VERSION = "vfai031-observability-v1"
_MAX_LATENCY_SAMPLES = 256
_MAX_LABEL_LENGTH = 80
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,159}$")
_SAFE_CODE_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{0,63}$")
_SAFE_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_EVENT_TYPES = {"start", "tool", "citation", "uncertainty", "delta", "proposal", "complete", "error"}
_MODES = {"neural", "deterministic-fallback", "unavailable", "unknown"}
_OUTCOMES = {"success", "failure", "cancelled", "timeout", "disconnected", "rejected", "validation-error"}
_STAGES = {"grounding", "generation", "memory", "response", "request-dispatch"}
_SEARCH_STATUSES = {"not-requested", "disabled", "complete", "no-results", "degraded", "failed", "unknown"}
_SAFE_CODE_PREFIXES = (
    "AI_", "API_", "CLIENT_", "INTERNET_", "LOCAL_", "MEMORY_", "MODEL_",
    "PROVIDER_", "QG_", "REQUEST_", "UNAPPROVED_", "UNSAFE_",
)
_KNOWN_ROUTE_SUFFIXES = {
    "/", "/health", "/system/health", "/contract", "/chat", "/chat/stream",
    "/chat/requests/{request_id}", "/memory", "/memory/preferences", "/memory/entries",
    "/validate-circuit", "/engineering/check", "/retrieval/search", "/internet/search",
    "/suggest-wiring", "/review-code", "/schematic-to-code", "/generate-code",
    "/datasheet/search", "/feedback", "/database/health", "/circuit/drc-check",
    "/circuit/export-gerber", "/circuit/bom-sourcing", "/circuit/emi-rules",
    "/circuit/synthesize", "/circuit/thermal-analysis", "/circuit/auto-layout",
    "/circuit/export", "/simulation/stream", "/datasheet/{part_number}", "/metrics",
}


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def safe_request_id(value: object) -> str:
    candidate = value if isinstance(value, str) else ""
    return candidate if _REQUEST_ID_RE.fullmatch(candidate) else "request-unknown"


def safe_code(value: object, *, fallback: str = "other") -> str:
    candidate = value if isinstance(value, str) else ""
    if not _SAFE_CODE_RE.fullmatch(candidate) or not candidate.startswith(_SAFE_CODE_PREFIXES):
        return fallback
    return candidate[:_MAX_LABEL_LENGTH]


def safe_artifact_id(value: object) -> str:
    candidate = value if isinstance(value, str) else ""
    if not _SAFE_ARTIFACT_RE.fullmatch(candidate) or not (
        candidate.startswith("vfdlm-")
        or candidate.startswith("deterministic-")
        or candidate.startswith("gateway-")
    ):
        return "none"
    return candidate[:_MAX_LABEL_LENGTH]


def route_label(value: object) -> str:
    """Return a bounded route template without retaining dynamic path data."""
    candidate = value if isinstance(value, str) else ""
    if not candidate:
        return "unknown"
    candidate = candidate.split("?", 1)[0]
    base = "/voltForge-ai/api/v1/model"
    if candidate.startswith(base):
        suffix = candidate[len(base):] or "/"
        if suffix not in _KNOWN_ROUTE_SUFFIXES:
            return "unknown"
        return suffix
    if candidate == "/":
        return "/"
    return "unknown"


def _outcome_for_status(status_code: int | None) -> str:
    if status_code is None:
        return "failure"
    if status_code in {408, 504}:
        return "timeout"
    if status_code in {409}:
        return "cancelled"
    if status_code == 422:
        return "validation-error"
    if status_code == 429:
        return "rejected"
    return "success" if status_code < 400 else "failure"


def _bounded_ms(value: float | int | None) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        number = 0.0
    return round(max(0.0, min(number, 600_000.0)), 3)


class _LatencyWindow:
    def __init__(self, maximum: int = _MAX_LATENCY_SAMPLES) -> None:
        self.values: deque[float] = deque(maxlen=maximum)

    def add(self, value: float) -> None:
        self.values.append(_bounded_ms(value))

    def snapshot(self) -> dict[str, Any]:
        if not self.values:
            return {"count": 0, "sumMs": 0.0, "p50Ms": None, "p95Ms": None, "maxMs": None}
        ordered = sorted(self.values)

        def percentile(fraction: float) -> float:
            index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
            return ordered[index]

        return {
            "count": len(ordered),
            "sumMs": round(sum(ordered), 3),
            "p50Ms": percentile(0.50),
            "p95Ms": percentile(0.95),
            "maxMs": ordered[-1],
        }


class _RouteStats:
    def __init__(self) -> None:
        self.count = 0
        self.outcomes: Counter[str] = Counter()
        self.latency = _LatencyWindow()

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "outcomes": dict(sorted(self.outcomes.items())),
            "latencyMs": self.latency.snapshot(),
        }


@dataclass
class RequestObservation:
    request_id: str
    route: str
    started_at: float
    finished: bool = False


@dataclass
class StreamObservation:
    request_id: str
    started_at: float
    first_token_at: float | None = None
    event_count: int = 0
    event_bytes: int = 0
    delta_count: int = 0
    output_characters: int = 0
    finished: bool = False


class ObservabilityRegistry:
    """Thread-safe counters and bounded latency windows for one process."""

    def __init__(
        self,
        *,
        latency_alert_ms: int | None = None,
        active_request_alert: int | None = None,
        fallback_alert_count: int | None = None,
        retrieval_error_alert_count: int | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.started_at = time.monotonic()
        self.latency_alert_ms = latency_alert_ms or _bounded_env_int(
            "VOLTFORGE_AI_OBSERVABILITY_LATENCY_ALERT_MS", 10_000, 100, 600_000
        )
        self.active_request_alert = active_request_alert or _bounded_env_int(
            "VOLTFORGE_AI_OBSERVABILITY_ACTIVE_REQUEST_ALERT", 32, 1, 10_000
        )
        self.fallback_alert_count = fallback_alert_count or _bounded_env_int(
            "VOLTFORGE_AI_OBSERVABILITY_FALLBACK_ALERT_COUNT", 5, 1, 100_000
        )
        self.retrieval_error_alert_count = retrieval_error_alert_count or _bounded_env_int(
            "VOLTFORGE_AI_OBSERVABILITY_RETRIEVAL_ERROR_ALERT_COUNT", 5, 1, 100_000
        )
        self._requests_started = 0
        self._requests_completed = 0
        self._active_requests = 0
        self._peak_active_requests = 0
        self._request_outcomes: Counter[str] = Counter()
        self._routes: dict[str, _RouteStats] = {}
        self._request_latency = _LatencyWindow()
        self._queue_latency = _LatencyWindow()
        self._stages: dict[str, _LatencyWindow] = {}
        self._generation_paths: Counter[str] = Counter()
        self._generation_modes: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._artifact_versions: Counter[str] = Counter()
        self._neural_attempts = 0
        self._tool_grounded_responses = 0
        self._model_load_attempts = 0
        self._model_load_successes = 0
        self._model_load_failures: Counter[str] = Counter()
        self._retrieval_attempts = 0
        self._retrieval_statuses: Counter[str] = Counter()
        self._retrieval_failures: Counter[str] = Counter()
        self._retrieval_cache_hits = 0
        self._retrieval_latency = _LatencyWindow()
        self._stream_started = 0
        self._stream_completed = 0
        self._stream_active = 0
        self._stream_outcomes: Counter[str] = Counter()
        self._stream_latency = _LatencyWindow()
        self._first_token_latency = _LatencyWindow()
        self._stream_events = 0
        self._stream_bytes = 0
        self._stream_deltas = 0
        self._stream_output_characters = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._input_token_observations = 0
        self._output_token_observations = 0
        self._resource_samples = 0
        self._latest_python_heap = 0
        self._peak_python_heap = 0
        self._alert_state: dict[str, bool] = {}
        if not tracemalloc.is_tracing():
            tracemalloc.start(1)

    def begin_request(self, request_id: object, route: object) -> RequestObservation:
        observation = RequestObservation(
            request_id=safe_request_id(request_id),
            route=route_label(route),
            started_at=time.monotonic(),
        )
        with self._lock:
            self._requests_started += 1
            self._active_requests += 1
            # The Python service uses non-blocking admission. Recording zero
            # explicitly makes that queue policy observable without inventing
            # a wait measurement.
            self._queue_latency.add(0.0)
            self._peak_active_requests = max(self._peak_active_requests, self._active_requests)
            self._sample_resources_locked()
        return observation

    def finish_request(
        self,
        observation: RequestObservation,
        *,
        status_code: int | None = None,
        outcome: str | None = None,
        route: object | None = None,
        duration_ms: float | None = None,
    ) -> None:
        if observation.finished:
            return
        observation.finished = True
        elapsed = _bounded_ms(duration_ms if duration_ms is not None else (time.monotonic() - observation.started_at) * 1000)
        resolved_outcome = outcome if outcome in _OUTCOMES else _outcome_for_status(status_code)
        resolved_route = route_label(route if route is not None else observation.route)
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._requests_completed += 1
            self._request_outcomes[resolved_outcome] += 1
            self._request_latency.add(elapsed)
            bucket = self._routes.setdefault(resolved_route, _RouteStats())
            bucket.count += 1
            bucket.outcomes[resolved_outcome] += 1
            bucket.latency.add(elapsed)
            self._sample_resources_locked()
            self._emit_alert_transitions_locked()
        logger.info(
            "AI request completed requestId=%s route=%s outcome=%s status=%s durationMs=%s",
            observation.request_id,
            resolved_route,
            resolved_outcome,
            status_code if isinstance(status_code, int) else "unknown",
            elapsed,
        )

    def record_stage(self, stage: object, duration_ms: float) -> None:
        name = stage if isinstance(stage, str) and stage in _STAGES else "other"
        with self._lock:
            self._stages.setdefault(name, _LatencyWindow()).add(duration_ms)

    def start_stream(self, request_id: object) -> StreamObservation:
        observation = StreamObservation(safe_request_id(request_id), time.monotonic())
        with self._lock:
            self._stream_started += 1
            self._stream_active += 1
            self._sample_resources_locked()
        return observation

    def observe_stream_event(self, observation: StreamObservation, event_type: object, event_bytes: int) -> None:
        if observation.finished:
            return
        name = event_type if isinstance(event_type, str) and event_type in _EVENT_TYPES else "other"
        size = max(0, min(int(event_bytes), 262_144))
        now = time.monotonic()
        observation.event_count += 1
        observation.event_bytes += size
        if name == "delta":
            observation.delta_count += 1
            if observation.first_token_at is None:
                observation.first_token_at = now
        with self._lock:
            self._stream_events += 1
            self._stream_bytes += size
            if name == "delta":
                self._stream_deltas += 1

    def finish_stream(
        self,
        observation: StreamObservation,
        *,
        outcome: str,
        output_characters: int = 0,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        if observation.finished:
            return
        observation.finished = True
        resolved_outcome = outcome if outcome in _OUTCOMES else "failure"
        elapsed = _bounded_ms((time.monotonic() - observation.started_at) * 1000)
        with self._lock:
            self._stream_active = max(0, self._stream_active - 1)
            self._stream_completed += 1
            self._stream_outcomes[resolved_outcome] += 1
            self._stream_latency.add(elapsed)
            if observation.first_token_at is not None:
                self._first_token_latency.add((observation.first_token_at - observation.started_at) * 1000)
            bounded_characters = max(0, min(int(output_characters), 24_000))
            self._stream_output_characters += bounded_characters
            if isinstance(input_tokens, int) and 0 <= input_tokens <= 200_000:
                self._input_tokens += input_tokens
                self._input_token_observations += 1
            if isinstance(output_tokens, int) and 0 <= output_tokens <= 200_000:
                self._output_tokens += output_tokens
                self._output_token_observations += 1
            self._sample_resources_locked()
            self._emit_alert_transitions_locked()

    def record_generation(
        self,
        *,
        mode: object,
        fallback_used: bool,
        fallback_reason: object,
        neural_attempted: bool,
        tool_grounded: bool,
        artifact_id: object = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        resolved_mode = mode if isinstance(mode, str) and mode in _MODES else "unknown"
        if resolved_mode == "neural":
            path = "neural"
        elif tool_grounded:
            path = "tool-grounded"
        elif fallback_used:
            path = "deterministic-fallback"
        else:
            path = "unknown"
        artifact = safe_artifact_id(artifact_id)
        reason = safe_code(fallback_reason, fallback="none") if fallback_used else "none"
        with self._lock:
            self._generation_modes[resolved_mode] += 1
            self._generation_paths[path] += 1
            self._artifact_versions[artifact] += 1
            if fallback_used:
                self._fallback_reasons[reason] += 1
            if neural_attempted:
                self._neural_attempts += 1
            if tool_grounded:
                self._tool_grounded_responses += 1
            if isinstance(input_tokens, int) and 0 <= input_tokens <= 200_000:
                self._input_tokens += input_tokens
                self._input_token_observations += 1
            if isinstance(output_tokens, int) and 0 <= output_tokens <= 200_000:
                self._output_tokens += output_tokens
                self._output_token_observations += 1
            self._emit_alert_transitions_locked()

    def record_model_load_success(self, artifact_id: object = None) -> None:
        with self._lock:
            self._model_load_attempts += 1
            self._model_load_successes += 1
            self._artifact_versions[safe_artifact_id(artifact_id)] += 1

    def record_model_load_failure(self, code: object) -> None:
        reason = safe_code(code)
        with self._lock:
            self._model_load_attempts += 1
            self._model_load_failures[reason] += 1
            self._emit_alert_transitions_locked()

    def record_retrieval(
        self,
        *,
        status: object,
        cache_hit: bool = False,
        failure_code: object = None,
        duration_ms: float | None = None,
    ) -> None:
        resolved_status = status if isinstance(status, str) and status in _SEARCH_STATUSES else "unknown"
        with self._lock:
            self._retrieval_attempts += 1
            self._retrieval_statuses[resolved_status] += 1
            if duration_ms is not None:
                self._retrieval_latency.add(duration_ms)
            if cache_hit:
                self._retrieval_cache_hits += 1
            if resolved_status in {"degraded", "failed"}:
                self._retrieval_failures[safe_code(failure_code)] += 1
                self._emit_alert_transitions_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def health(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot_locked()
            return {
                "schemaVersion": SCHEMA_VERSION,
                "ready": True,
                "activeRequests": self._active_requests,
                "activeStreams": self._stream_active,
                "alerts": snapshot["alerts"],
                "rawPromptStored": False,
                "rawProjectContentStored": False,
                "rawModelOutputStored": False,
            }

    def _snapshot_locked(self) -> dict[str, Any]:
        uptime_seconds = max(0.001, time.monotonic() - self.started_at)
        alerts = self._alerts_locked()
        return {
            "schemaVersion": SCHEMA_VERSION,
            "uptimeSeconds": round(uptime_seconds, 3),
            "throughputPerMinute": round(self._requests_completed / uptime_seconds * 60, 3),
            "requests": {
                "started": self._requests_started,
                "completed": self._requests_completed,
                "active": self._active_requests,
                "peakActive": self._peak_active_requests,
                "outcomes": dict(sorted(self._request_outcomes.items())),
                "latencyMs": self._request_latency.snapshot(),
                "queueTimeMs": self._queue_latency.snapshot(),
                "byRoute": {key: value.snapshot() for key, value in sorted(self._routes.items())},
            },
            "stages": {
                key: value.snapshot() for key, value in sorted(self._stages.items())
            },
            "generation": {
                "paths": dict(sorted(self._generation_paths.items())),
                "modes": dict(sorted(self._generation_modes.items())),
                "fallbackReasons": dict(sorted(self._fallback_reasons.items())),
                "artifactVersions": dict(sorted(self._artifact_versions.items())),
                "neuralAttempts": self._neural_attempts,
                "toolGroundedResponses": self._tool_grounded_responses,
            },
            "streams": {
                "started": self._stream_started,
                "completed": self._stream_completed,
                "active": self._stream_active,
                "outcomes": dict(sorted(self._stream_outcomes.items())),
                "latencyMs": self._stream_latency.snapshot(),
                "firstTokenLatencyMs": self._first_token_latency.snapshot(),
                "eventCount": self._stream_events,
                "eventBytes": self._stream_bytes,
                "deltaCount": self._stream_deltas,
                "outputCharacters": self._stream_output_characters,
            },
            "tokens": {
                "inputObserved": self._input_token_observations,
                "inputCount": self._input_tokens,
                "outputObserved": self._output_token_observations,
                "outputCount": self._output_tokens,
                "estimated": False,
            },
            "retrieval": {
                "attempts": self._retrieval_attempts,
                "statuses": dict(sorted(self._retrieval_statuses.items())),
                "cacheHits": self._retrieval_cache_hits,
                "failures": dict(sorted(self._retrieval_failures.items())),
                "latencyMs": self._retrieval_latency.snapshot(),
            },
            "model": {
                "loadAttempts": self._model_load_attempts,
                "loadSuccesses": self._model_load_successes,
                "loadFailures": dict(sorted(self._model_load_failures.items())),
            },
            "resources": {
                "sampleCount": self._resource_samples,
                "latestPythonHeapBytes": self._latest_python_heap,
                "peakPythonHeapBytes": self._peak_python_heap,
                "measurement": "python-traced-heap",
            },
            "alerts": alerts,
            "rawPromptStored": False,
            "rawProjectContentStored": False,
            "rawModelOutputStored": False,
        }

    def _sample_resources_locked(self) -> None:
        try:
            current, peak = tracemalloc.get_traced_memory()
        except RuntimeError:
            return
        self._resource_samples += 1
        self._latest_python_heap = max(0, int(current))
        self._peak_python_heap = max(self._peak_python_heap, int(peak))

    def _alerts_locked(self) -> dict[str, Any]:
        high_latency = bool(
            self._request_latency.values
            and max(self._request_latency.values) >= self.latency_alert_ms
        )
        capacity = self._active_requests >= self.active_request_alert
        repeated_fallback = (
            self._neural_attempts > 0
            and sum(self._fallback_reasons.values()) >= self.fallback_alert_count
        )
        retrieval_errors = sum(self._retrieval_failures.values()) >= self.retrieval_error_alert_count
        model_failure = bool(self._model_load_failures)
        active = {
            "modelLoadFailure": model_failure,
            "latencySaturation": high_latency,
            "resourceSaturation": capacity,
            "repeatedFallback": repeated_fallback,
            "retrievalErrors": retrieval_errors,
        }
        return {
            "active": [key for key, value in active.items() if value],
            "state": active,
            "thresholds": {
                "latencyAlertMs": self.latency_alert_ms,
                "activeRequestAlert": self.active_request_alert,
                "fallbackAlertCount": self.fallback_alert_count,
                "retrievalErrorAlertCount": self.retrieval_error_alert_count,
            },
        }

    def _emit_alert_transitions_locked(self) -> None:
        alerts = self._alerts_locked()["state"]
        for name, active in alerts.items():
            previous = self._alert_state.get(name, False)
            if active != previous:
                self._alert_state[name] = active
                logger.warning(
                    "AI observability alert transition alert=%s active=%s",
                    name,
                    active,
                )


observability = ObservabilityRegistry()


__all__ = [
    "SCHEMA_VERSION",
    "ObservabilityRegistry",
    "RequestObservation",
    "StreamObservation",
    "observability",
    "route_label",
    "safe_artifact_id",
    "safe_code",
    "safe_request_id",
]
