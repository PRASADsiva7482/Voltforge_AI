from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api.chat as chat_module
from api.chat import stream_chat_sse
from api.schemas import ChatRequest, ChatResponse
from api_contract import (
    ApiContractError,
    ArtifactIdentity,
    CancellationToken,
    SseContractEmitter,
    SseEvent,
    cancellation_registry,
    load_policy,
)
from api_contract.schema import POLICY_SHA256
from config import get_settings
from main import app
from tools.evaluate_api_contract import evaluate as evaluate_api_contract
from tools.evaluate_api_contract import verify as verify_api_contract


PRIVATE_TOKEN = "vfai026-private-token"


def event_payload(event: str) -> dict[str, object]:
    line = next(item for item in event.splitlines() if item.startswith("data:"))
    return json.loads(line.removeprefix("data:").strip())


def canonical_event(data: dict[str, object]) -> dict[str, object]:
    return {key: data[key] for key in SseEvent.model_fields}


async def collect(request: ChatRequest, **kwargs) -> list[str]:
    return [item async for item in stream_chat_sse(request, **kwargs)]


def test_policy_and_default_request_response_versions_are_pinned() -> None:
    policy = load_policy()
    request = ChatRequest(message="Explain this circuit.")
    response = ChatResponse(reply="Bounded local response.")

    assert policy["policyId"] == "vfai026-fastapi-sse-contract-v1"
    assert len(POLICY_SHA256) == 64
    assert request.schemaVersion == 1
    assert request.contractVersion == "1.0.0"
    assert response.schemaVersion == 1
    assert response.contractVersion == "1.0.0"


@pytest.mark.parametrize(
    "payload",
    [
        {"schemaVersion": 2, "message": "Incompatible."},
        {"schemaVersion": 1, "contractVersion": "2.0.0", "message": "Wrong."},
        {"schemaVersion": 1, "message": "Unknown.", "unexpected": True},
    ],
)
def test_request_contract_rejects_incompatible_versions_and_unknown_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(payload)


def test_stream_events_have_one_strict_versioned_envelope_and_bounded_sequence() -> None:
    events = asyncio.run(
        collect(
            ChatRequest(
                message="Explain the selected LED.",
                projectRevision="revision-026",
            ),
            request_id="request-026",
            session_id="session-026",
        )
    )
    parsed = [event_payload(item) for item in events]

    assert parsed[0]["type"] == "start"
    assert parsed[-1]["type"] == "complete"
    assert [item["sequence"] for item in parsed] == list(range(len(parsed)))
    assert len(parsed) <= 384
    assert {item["requestId"] for item in parsed} == {"request-026"}
    assert {item["sessionId"] for item in parsed} == {"session-026"}
    assert {item["projectRevision"] for item in parsed} == {"client:revision-026"}
    assert {item["contractVersion"] for item in parsed} == {"1.0.0"}
    assert all(item["artifact"]["artifactVersion"] for item in parsed)
    assert all(SseEvent.model_validate(canonical_event(item)) for item in parsed)
    assert all(
        all(item[key] == value for key, value in item["payload"].items())
        for item in parsed
    )
    assert all(len(raw.encode("utf-8")) <= 262_144 for raw in events)
    assert sum(len(raw.encode("utf-8")) for raw in events) <= 1_048_576


def test_emitter_rejects_oversized_delta_reserved_fields_and_post_terminal_events() -> None:
    artifact = ArtifactIdentity(
        artifactVersion="deterministic-tools-v1",
        runtimeState="unavailable",
        ready=False,
    )
    emitter = SseContractEmitter(
        request_id="request-limits",
        session_id="session-limits",
        project_revision="revision-limits",
        mode="deterministic-fallback",
        artifact=artifact,
    )
    emitter.emit("start", {"readiness": {}, "fallbackUsed": True})
    with pytest.raises(ValidationError):
        emitter.emit("delta", {"delta": "x" * 97})
    with pytest.raises(ApiContractError, match="envelope"):
        emitter.emit("delta", {"delta": "ok", "requestId": "forged"})

    terminal = SseContractEmitter(
        request_id="request-terminal",
        session_id="session-terminal",
        project_revision="revision-terminal",
        mode="deterministic-fallback",
        artifact=artifact,
    )
    terminal.emit("start", {"readiness": {}, "fallbackUsed": True})
    terminal.emit(
        "error",
        {"code": "REQUEST_CANCELLED", "message": "Cancelled.", "retryable": False},
    )
    with pytest.raises(ApiContractError, match="terminal"):
        terminal.emit("delta", {"delta": "late"})


def test_explicit_cancellation_stops_before_generation_and_memory_persistence() -> None:
    class MemoryProbe:
        recorded = False

        def record_turn(self, _message: str, _reply: str) -> bool:
            self.recorded = True
            return True

        def refreshed_metadata(self) -> dict[str, object]:
            return {"status": "ready"}

    probe = MemoryProbe()

    async def scenario() -> list[dict[str, object]]:
        token = CancellationToken("request-cancelled")
        generator = stream_chat_sse(
            ChatRequest(message="Explain this circuit."),
            memory_binding=probe,  # type: ignore[arg-type]
            request_id="request-cancelled",
            session_id="session-cancelled",
            cancellation=token,
        )
        first = event_payload(await anext(generator))
        token.cancel()
        remaining = [event_payload(item) async for item in generator]
        return [first, *remaining]

    parsed = asyncio.run(scenario())

    assert [item["type"] for item in parsed] == ["start", "error"]
    assert parsed[-1]["code"] == "REQUEST_CANCELLED"
    assert parsed[-1]["retryable"] is False
    assert probe.recorded is False


def test_disconnect_stops_stream_without_terminal_output() -> None:
    async def disconnected() -> bool:
        return True

    parsed = [
        event_payload(item)
        for item in asyncio.run(
            collect(
                ChatRequest(message="Explain this circuit."),
                request_id="request-disconnect",
                session_id="session-disconnect",
                disconnected=disconnected,
            )
        )
    ]

    assert [item["type"] for item in parsed] == ["start"]
    assert cancellation_registry.active_count() == 0


def test_timeout_emits_typed_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = chat_module.prepare_grounded_context

    def slow_grounding(*args, **kwargs):
        time.sleep(1.05)
        return original(*args, **kwargs)

    monkeypatch.setenv("VOLTFORGE_AI_CHAT_REQUEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(chat_module, "prepare_grounded_context", slow_grounding)
    parsed = [
        event_payload(item)
        for item in asyncio.run(
            collect(
                ChatRequest(message="Explain timeout handling."),
                settings=get_settings(),
                request_id="request-timeout",
                session_id="session-timeout",
            )
        )
    ]

    assert parsed[0]["type"] == "start"
    assert parsed[-1]["type"] == "error"
    assert parsed[-1]["code"] == "REQUEST_TIMEOUT"
    assert parsed[-1]["retryable"] is True


def test_malformed_generation_emits_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_module,
        "resolve_generation",
        lambda **_kwargs: SimpleNamespace(
            metadata={
                "mode": "deterministic-fallback",
                "fallbackUsed": True,
                "fallbackReasonCode": "MALFORMED_FIXTURE",
                "fallbackSource": "voltforge-deterministic-tools",
                "neuralAttempted": False,
                "neuralArtifactId": None,
                "qualityGate": {},
            },
            deterministic_value={"reply": "not-a-typed-response"},
        ),
    )
    parsed = [
        event_payload(item)
        for item in asyncio.run(
            collect(
                ChatRequest(message="Exercise malformed generation."),
                request_id="request-malformed",
                session_id="session-malformed",
            )
        )
    ]

    assert parsed[-1]["type"] == "error"
    assert parsed[-1]["code"] == "MALFORMED_GENERATION"
    assert all(item["type"] != "complete" for item in parsed)


def test_http_schema_and_size_failures_are_typed_and_content_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_API_TOKEN", PRIVATE_TOKEN)
    headers = {"X-Voltforge-AI-Token": PRIVATE_TOKEN}
    with TestClient(app) as client:
        incompatible = client.post(
            "/voltForge-ai/api/v1/model/chat/stream",
            headers=headers,
            json={"schemaVersion": 2, "message": "Do not echo this private prompt."},
        )
        oversized = client.post(
            "/voltForge-ai/api/v1/model/chat/stream",
            headers={**headers, "Content-Length": "2000001"},
            content=b"{}",
        )

    assert incompatible.status_code == 422
    assert incompatible.json()["code"] == "API_SCHEMA_VALIDATION_FAILED"
    assert "private prompt" not in incompatible.text
    assert oversized.status_code == 413
    assert oversized.json()["code"] == "API_REQUEST_TOO_LARGE"


def test_contract_health_covers_every_fastapi_route_and_reports_independent_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "false")
    with TestClient(app) as client:
        health = client.get("/voltForge-ai/api/v1/model/health").json()
        contract = client.get("/voltForge-ai/api/v1/model/contract").json()

    assert health["apiContract"]["policyId"] == "vfai026-fastapi-sse-contract-v1"
    assert contract["routeCount"] == len(contract["routes"])
    assert contract["routeCount"] >= 30
    assert all("available" in route for route in contract["routes"])
    stream = next(
        route for route in contract["routes"] if route["path"].endswith("/chat/stream")
    )
    internet = next(
        route for route in contract["routes"] if route["path"].endswith("/internet/search")
    )
    assert stream["available"] is True
    assert stream["deterministicFallbackAvailable"] is True
    assert internet["available"] is False
    assert contract["subsystems"]["internetRetrieval"]["state"] == "offline"
    assert contract["subsystems"]["localModel"]["ready"] is False


def test_sync_chat_response_carries_contract_request_artifact_revision_and_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_API_TOKEN", PRIVATE_TOKEN)
    headers = {
        "X-Voltforge-AI-Token": PRIVATE_TOKEN,
        "X-Voltforge-Request-Id": "request-sync-026",
    }
    with TestClient(app) as client:
        response = client.post(
            "/voltForge-ai/api/v1/model/chat",
            headers=headers,
            json={"schemaVersion": 1, "message": "Explain this LED."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["schemaVersion"] == 1
    assert body["contractVersion"] == "1.0.0"
    assert body["requestId"] == "request-sync-026"
    assert body["projectRevision"].startswith("snapshot-sha256:")
    assert body["mode"] == "deterministic-fallback"
    assert body["artifact"]["artifactVersion"] == "deterministic-tools-v1"
    assert body["readiness"]["engineeringTools"]["ready"] is True


def test_cancel_endpoint_changes_only_an_active_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_API_TOKEN", PRIVATE_TOKEN)
    active_id = "request-active-026"
    token = cancellation_registry.register(active_id)
    try:
        with TestClient(app) as client:
            accepted = client.delete(
                f"/voltForge-ai/api/v1/model/chat/requests/{active_id}",
                headers={"X-Voltforge-AI-Token": PRIVATE_TOKEN},
            )
            missing = client.delete(
                "/voltForge-ai/api/v1/model/chat/requests/request-missing-026",
                headers={"X-Voltforge-AI-Token": PRIVATE_TOKEN},
            )
        assert accepted.status_code == 202
        assert accepted.json()["status"] == "cancellation-requested"
        assert token.cancelled is True
        assert missing.status_code == 404
        assert missing.json()["code"] == "REQUEST_NOT_ACTIVE"
    finally:
        cancellation_registry.release(active_id)


def test_api_contract_evaluation_receipt_is_reproducible(tmp_path: Path) -> None:
    report_path = tmp_path / "fastapi-sse-contract-v1.json"

    evaluated = evaluate_api_contract(report_path)
    verified = verify_api_contract(report_path)

    assert evaluated == verified
    assert evaluated["checkCount"] == 54
    assert evaluated["passedCheckCount"] == 54
    assert evaluated["networkAccessed"] is False
    assert evaluated["rawPromptStored"] is False
