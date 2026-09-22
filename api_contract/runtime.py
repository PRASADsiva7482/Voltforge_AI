"""Bounded SSE emission, cancellation, readiness, and public-safe failures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import threading
from typing import Any, Awaitable, Callable, Iterable

from api_contract.schema import (
    CONTRACT_VERSION,
    POLICY_ID,
    POLICY_SHA256,
    ArtifactIdentity,
    ApiContractError,
    ContractErrorResponse,
    SseEvent,
    load_policy,
)


MODEL_ID = "voltforge-local-engine-v1"
FALLBACK_ARTIFACT_VERSION = "deterministic-tools-v1"


class RequestStopped(RuntimeError):
    def __init__(self, code: str, message: str, *, disconnected: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.disconnected = disconnected


@dataclass
class CancellationToken:
    request_id: str
    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise RequestStopped(
                "REQUEST_CANCELLED", "The local AI request was cancelled."
            )


class CancellationRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tokens: dict[str, CancellationToken] = {}

    def register(self, request_id: str) -> CancellationToken:
        with self._lock:
            if request_id in self._tokens:
                raise ApiContractError(
                    "API_REQUEST_ID_CONFLICT", "The request identifier is already active."
                )
            token = CancellationToken(request_id)
            self._tokens[request_id] = token
            return token

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            token = self._tokens.get(request_id)
            if token is None:
                return False
            token.cancel()
            return True

    def release(self, request_id: str) -> None:
        with self._lock:
            self._tokens.pop(request_id, None)

    def active_count(self) -> int:
        with self._lock:
            return len(self._tokens)


cancellation_registry = CancellationRegistry()


def artifact_identity(model_health: dict[str, Any]) -> ArtifactIdentity:
    artifact_id = model_health.get("activeArtifactId") or model_health.get("artifactId")
    registry_revision = model_health.get("registryRevision")
    return ArtifactIdentity(
        artifactId=str(artifact_id) if artifact_id else None,
        artifactVersion=(
            str(artifact_id) if artifact_id else FALLBACK_ARTIFACT_VERSION
        ),
        registryRevision=(
            int(registry_revision) if isinstance(registry_revision, int) else None
        ),
        runtimeState=str(
            model_health.get("runtimeState")
            or model_health.get("state")
            or "unavailable"
        ),
        ready=bool(model_health.get("ready")),
    )


class SseContractEmitter:
    def __init__(
        self,
        *,
        request_id: str,
        session_id: str,
        project_revision: str,
        mode: str,
        artifact: ArtifactIdentity,
        event_observer: Callable[[str, int], None] | None = None,
    ) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self.project_revision = project_revision
        self.mode = mode
        self.artifact = artifact
        self.event_observer = event_observer
        self.sequence = 0
        self.total_bytes = 0
        self.counts: dict[str, int] = {}
        self.terminal_emitted = False
        self.policy = load_policy()

    def emit(self, name: str, payload: dict[str, Any]) -> str:
        if self.terminal_emitted:
            raise ApiContractError(
                "API_EVENT_AFTER_TERMINAL", "No SSE event may follow a terminal event."
            )
        limits = self.policy["limits"]
        if (self.sequence == 0 and name != "start") or (
            self.sequence > 0 and name == "start"
        ):
            raise ApiContractError(
                "API_EVENT_ORDER_INVALID", "The SSE stream must begin with one start event."
            )
        if self.sequence >= int(limits["maximumStreamEvents"]):
            raise ApiContractError(
                "API_STREAM_EVENT_LIMIT_EXCEEDED", "The SSE event limit was exceeded."
            )
        maximum_by_type = {
            "tool": int(limits["maximumToolEvents"]),
            "citation": int(limits["maximumCitationEvents"]),
        }
        count = self.counts.get(name, 0) + 1
        if name in maximum_by_type and count > maximum_by_type[name]:
            raise ApiContractError(
                "API_STREAM_TYPED_EVENT_LIMIT_EXCEEDED",
                f"The {name} SSE event limit was exceeded.",
            )
        reserved = {
            "schemaVersion",
            "contractVersion",
            "type",
            "eventId",
            "sequence",
            "requestId",
            "sessionId",
            "projectRevision",
            "model",
            "mode",
            "artifact",
            "payload",
        }
        if reserved.intersection(payload):
            raise ApiContractError(
                "API_EVENT_RESERVED_FIELD",
                "An SSE payload attempted to replace contract envelope fields.",
            )
        event = SseEvent(
            type=name,
            eventId=f"event:v1:{self.request_id}:{self.sequence}",
            sequence=self.sequence,
            requestId=self.request_id,
            sessionId=self.session_id,
            projectRevision=self.project_revision,
            model=MODEL_ID,
            mode=self.mode,
            artifact=self.artifact,
            payload=payload,
        )
        # Preserve the original flat payload for Spring/UI compatibility while
        # retaining the strict payload object as the canonical v1 contract.
        data = {**event.model_dump(mode="json"), **payload}
        encoded = json.dumps(data, separators=(",", ":"), default=str)
        framed = f"event: {name}\ndata: {encoded}\n\n"
        event_bytes = len(framed.encode("utf-8"))
        if event_bytes > int(limits["maximumEventBytes"]):
            raise ApiContractError(
                "API_EVENT_SIZE_LIMIT_EXCEEDED", "An SSE event exceeded its byte limit."
            )
        if self.total_bytes + event_bytes > int(limits["maximumStreamBytes"]):
            raise ApiContractError(
                "API_STREAM_SIZE_LIMIT_EXCEEDED", "The SSE stream exceeded its byte limit."
            )
        self.sequence += 1
        self.total_bytes += event_bytes
        self.counts[name] = count
        if name in {"complete", "error"}:
            self.terminal_emitted = True
        if self.event_observer is not None:
            try:
                self.event_observer(name, event_bytes)
            except Exception:
                # Telemetry is never allowed to change the API contract or
                # interrupt a valid local response.
                pass
        return framed


async def wait_for_stage(
    function: Callable[[], Any],
    *,
    token: CancellationToken,
    disconnected: Callable[[], Awaitable[bool]] | None,
    deadline_seconds: float,
) -> Any:
    """Run one bounded blocking stage while observing cancel/disconnect/deadline."""
    token.check()
    worker = asyncio.create_task(asyncio.to_thread(function))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_seconds
    try:
        while not worker.done():
            if token.cancelled:
                raise RequestStopped(
                    "REQUEST_CANCELLED", "The local AI request was cancelled."
                )
            if disconnected is not None and await disconnected():
                token.cancel()
                raise RequestStopped(
                    "CLIENT_DISCONNECTED",
                    "The client disconnected.",
                    disconnected=True,
                )
            remaining = deadline - loop.time()
            if remaining <= 0:
                token.cancel()
                raise RequestStopped(
                    "REQUEST_TIMEOUT", "The local AI request exceeded its deadline."
                )
            await asyncio.wait({worker}, timeout=min(0.025, remaining))
        token.check()
        return worker.result()
    finally:
        if not worker.done():
            worker.cancel()


def error_response(
    code: str,
    message: str,
    *,
    retryable: bool,
    request_id: str | None = None,
    field_errors: Iterable[str] = (),
) -> dict[str, Any]:
    return ContractErrorResponse(
        requestId=request_id,
        code=code,
        message=message,
        retryable=retryable,
        fieldErrors=list(field_errors)[:32],
    ).model_dump(mode="json")


def contract_health(
    *,
    routes: Iterable[Any],
    model_health: dict[str, Any],
    context_health: dict[str, Any],
    engineering_health: dict[str, Any],
    local_retrieval_health: dict[str, Any],
    internet_health: dict[str, Any],
    grounding_health: dict[str, Any],
    memory_health: dict[str, Any],
    feedback_health: dict[str, Any],
) -> dict[str, Any]:
    subsystem_values = {
        "localModel": model_health,
        "contextCompiler": context_health,
        "engineeringTools": engineering_health,
        "localRetrieval": local_retrieval_health,
        "internetRetrieval": internet_health,
        "grounding": grounding_health,
        "memory": memory_health,
        "feedbackGovernance": feedback_health,
    }

    def dependency_ready(name: str) -> bool:
        state = subsystem_values[name]
        if name == "internetRetrieval":
            return bool(state.get("enabled")) and state.get("state") not in {
                "offline",
                "unavailable",
            }
        return bool(state.get("ready"))

    route_states = []
    for route in routes:
        path = str(getattr(route, "path", ""))
        methods = sorted(getattr(route, "methods", set()) or [])
        if not path:
            continue
        dependencies: list[str] = []
        execution_class = "deterministic-local"
        if path.endswith("/chat") or path.endswith("/chat/stream"):
            dependencies = [
                "contextCompiler",
                "engineeringTools",
                "localRetrieval",
                "grounding",
            ]
            execution_class = "local-generation-with-deterministic-fallback"
        elif "/retrieval/search" in path:
            dependencies = ["localRetrieval"]
        elif "/internet/" in path or "/datasheet/" in path:
            dependencies = ["internetRetrieval"]
        elif "/engineering/check" in path or path.endswith("/validate-circuit"):
            dependencies = ["engineeringTools"]
        elif "/memory" in path:
            dependencies = ["memory"]
        elif path.endswith("/feedback"):
            dependencies = ["feedbackGovernance"]
        dependency_states = {
            name: dependency_ready(name) for name in dependencies
        }
        fallback_available = execution_class == (
            "local-generation-with-deterministic-fallback"
        )
        route_states.append(
            {
                "path": path,
                "methods": methods,
                "executionClass": execution_class,
                "dependencies": dependencies,
                "dependencyStates": dependency_states,
                "available": fallback_available or all(dependency_states.values()),
                "deterministicFallbackAvailable": fallback_available,
            }
        )
    route_states.sort(key=lambda item: (item["path"], item["methods"]))
    policy = load_policy()
    return {
        "ready": True,
        "policyId": POLICY_ID,
        "policySha256": POLICY_SHA256,
        "contractVersion": CONTRACT_VERSION,
        "schemaVersion": 1,
        "routeCount": len(route_states),
        "routes": route_states,
        "activeRequestCount": cancellation_registry.active_count(),
        "generationNetworkRequired": False,
        "hiddenReasoningEventAllowed": False,
        "limits": dict(policy["limits"]),
        "subsystems": subsystem_values,
    }
