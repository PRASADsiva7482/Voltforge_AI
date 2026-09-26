import asyncio
import json
from pathlib import Path
import socket

import pytest
import requests
from fastapi.testclient import TestClient

import api.chat as chat_module
from api.chat import _get_local_orchestrator, stream_chat_sse
from api.schemas import ChatRequest
from config import get_settings
from main import app


from model.runtime_service import get_model_runtime_service


async def collect_events(request: ChatRequest) -> list[str]:
    get_model_runtime_service().start()
    return [event async for event in stream_chat_sse(request)]


def event_payload(event: str) -> dict[str, object]:
    data_line = next(line for line in event.splitlines() if line.startswith("data:"))
    return json.loads(data_line.removeprefix("data:").strip())


def test_chat_stream_is_explicitly_local_and_completes(monkeypatch) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_STORE_CONVERSATIONS", "false")
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "false")
    events = asyncio.run(collect_events(ChatRequest(message="Explain my LED circuit")))
    start = event_payload(events[0])
    complete = event_payload(events[-1])

    assert start["mode"] == "deterministic-fallback"
    assert start["model"] == "voltforge-local-engine-v1"
    assert start["generationNetworkAccess"] is False
    assert start["fallbackUsed"] is True
    assert start["fallbackReasonCode"] == "NO_APPROVED_MODEL_ARTIFACT"
    assert start["fallbackSource"] == "voltforge-deterministic-tools"
    assert start["neuralAttempted"] is False
    assert start["neuralArtifactId"] is None
    assert start["qualityGate"] == {
        "policyId": "vfai019-generation-quality-policy-v1",
        "status": "not-run",
        "code": "QG_NOT_RUN_MODEL_UNAVAILABLE",
        "rawOutputStored": False,
    }
    assert start["contextCompiler"]["ready"] is True
    assert start["contextCompiler"]["policyId"] == "vfai020-project-context-policy-v1"
    assert start["contextCompiler"]["rawPromptStored"] is False
    assert start["engineeringTools"]["ready"] is True
    assert start["engineeringTools"]["policyId"] == (
        "vfai021-authoritative-engineering-tools-v1"
    )
    assert start["engineeringTools"]["criticalModelOverrideAllowed"] is False
    assert start["localRetrieval"]["ready"] is True
    assert start["localRetrieval"]["policyId"] == (
        "vfai022-curated-local-retrieval-v1"
    )
    assert start["localRetrieval"]["embeddingsEnabled"] is False
    assert start["internetRetrieval"]["enabled"] is False
    assert start["internetRetrieval"]["state"] == "offline"
    assert start["internetRetrieval"]["policyId"] == (
        "vfai023-secure-internet-evidence-v1"
    )
    assert start["internetRetrieval"]["generationDependency"] is False
    assert start["internetRetrieval"]["arbitraryUrlFetchAllowed"] is False
    assert start["internetRetrieval"]["webContentTrainingAllowed"] is False
    assert any(event.startswith("event: delta") for event in events)
    assert complete["mode"] == "deterministic-fallback"
    assert complete["fallbackUsed"] is True
    assert complete["fallbackReasonCode"] == "NO_APPROVED_MODEL_ARTIFACT"
    assert complete["neuralAttempted"] is False
    assert complete["qualityGate"]["status"] == "not-run"
    assert complete["contextCompiler"]["status"] == "compiled"
    assert complete["contextCompiler"]["promptTokens"] <= complete["contextCompiler"][
        "promptTokenLimit"
    ]
    assert complete["contextCompiler"]["rawProjectContextStored"] is False
    assert complete["reply"]


def test_chat_stream_emits_local_engine_checks(monkeypatch) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_STORE_CONVERSATIONS", "false")
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "false")
    request = ChatRequest.model_validate(
        {
            "message": "Is this LED safe?",
            "boardType": "ARDUINO_UNO",
            "components": [{"id": "led-1", "type": "LED", "name": "Status LED"}],
            "wires": [],
            "code": "void setup() { digitalWrite(13, HIGH); }\nvoid loop() {}",
        }
    )
    events = asyncio.run(collect_events(request))
    assert any(
        event.startswith("event: tool") and "engineering-authority-index" in event
        for event in events
    )
    assert any("firmware.pin-mode-missing" in event for event in events)
    assert any(
        event.startswith("event: tool") and "curated-local-retrieval" in event
        for event in events
    )
    assert any('"modelOverrideAllowed":false' in event for event in events)


def test_ready_model_without_context_compiler_does_not_attempt_or_fabricate_neural(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_STORE_CONVERSATIONS", "false")
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "false")
    monkeypatch.setattr(
        chat_module.model_runtime_service,
        "health",
        lambda: {"ready": True, "code": "READY", "artifactId": "approved-future-artifact"},
    )

    def forbidden_runtime_access():
        raise AssertionError("Chat accessed neural runtime without a context compiler")

    monkeypatch.setattr(
        chat_module.model_runtime_service,
        "runtime_for_generation",
        forbidden_runtime_access,
    )

    events = asyncio.run(collect_events(ChatRequest(message="Explain this LED circuit")))
    start = event_payload(events[0])
    complete = event_payload(events[-1])

    assert start["neuralModelReady"] is True
    assert start["fallbackReasonCode"] == "NEURAL_GENERATION_NOT_ENABLED"
    assert start["neuralAttempted"] is False
    assert start["qualityGate"]["code"] == "QG_NOT_RUN_GENERATION_DISABLED"
    assert complete["fallbackReasonCode"] == "NEURAL_GENERATION_NOT_ENABLED"
    assert complete["neuralAttempted"] is False
    assert complete["reply"]


def test_local_generation_completes_when_all_outbound_network_is_blocked(monkeypatch) -> None:
    def network_denied(*_args, **_kwargs):
        raise AssertionError("Local generation attempted outbound network access")

    monkeypatch.setenv("VOLTFORGE_AI_STORE_CONVERSATIONS", "false")
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "false")
    monkeypatch.setattr(requests.sessions.Session, "request", network_denied)
    monkeypatch.setattr(socket, "create_connection", network_denied)
    _get_local_orchestrator.cache_clear()

    events = asyncio.run(
        collect_events(
            ChatRequest(message="Search the datasheet for UNKNOWN_PART_XYZ and explain its voltage")
        )
    )

    assert events[0].startswith("event: start")
    assert events[-1].startswith("event: complete")
    assert not any(event.startswith("event: error") for event in events)


def test_generation_sources_contain_no_hosted_model_client_or_credential() -> None:
    ai_root = Path(__file__).resolve().parents[1]
    assert not (ai_root / "api" / "llm.py").exists()
    source_paths = [
        *(ai_root / "api").glob("*.py"),
        ai_root / "config.py",
        ai_root / "config.json",
        ai_root / "requirements.txt",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = (
        "voltforge_ai_llm_",
        "api.openai.com",
        "openairesponsesprovider",
        "anthropic",
        "google.generativeai",
        "api.llm",
    )
    for marker in banned:
        assert marker not in source


def test_settings_expose_retrieval_not_model_provider(monkeypatch) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "true")
    settings = get_settings()

    assert settings.internet_retrieval_enabled is True
    assert not hasattr(settings, "llm_provider")
    assert not hasattr(settings, "llm_api_key")


def test_health_separates_local_generation_and_optional_retrieval(monkeypatch) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "false")
    with TestClient(app) as client:
        response = client.get("/voltForge-ai/api/v1/model/health")

    assert response.status_code == 200
    data = response.json()
    assert data["generation"] == {
        "mode": "deterministic-fallback",
        "networkRequired": False,
        "neuralOutputPolicy": "quality-gated-only",
        "qualityGatePolicyId": "vfai019-generation-quality-policy-v1",
        "fallbackSource": "voltforge-deterministic-tools",
    }
    assert data["generationQuality"]["policyId"] == "vfai019-generation-quality-policy-v1"
    assert data["generationQuality"]["rawPromptStored"] is False
    assert data["generationQuality"]["rawOutputStored"] is False
    assert data["contextCompiler"]["ready"] is True
    assert data["contextCompiler"]["policyId"] == "vfai020-project-context-policy-v1"
    assert data["contextCompiler"]["minimumSupportedContextWindowTokens"] == 768
    assert data["engineeringTools"]["ready"] is True
    assert data["engineeringTools"]["toolCount"] == 7
    assert data["engineeringTools"]["criticalModelOverrideAllowed"] is False
    assert data["localRetrieval"]["ready"] is True
    assert data["localRetrieval"]["indexVersion"] == "1.1.0"
    assert data["localRetrieval"]["staleResultsAllowed"] is False
    assert data["internetRetrieval"]["policy"] == "optional-evidence-only"
    assert data["internetRetrieval"]["policyId"] == (
        "vfai023-secure-internet-evidence-v1"
    )
    assert data["internetRetrieval"]["generationDependency"] is False
    assert data["internetRetrieval"]["arbitraryUrlFetchAllowed"] is False
    assert "llmProvider" not in data
    assert "llmConfigured" not in data


def test_chat_endpoint_requires_configured_service_token(monkeypatch) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_API_TOKEN", "test-private-token")
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "false")
    with TestClient(app) as client:
        rejected = client.post(
            "/voltForge-ai/api/v1/model/chat/stream",
            json={"message": "Explain my LED circuit"},
        )
        accepted = client.post(
            "/voltForge-ai/api/v1/model/chat/stream",
            headers={"X-Voltforge-AI-Token": "test-private-token"},
            json={"message": "Explain my LED circuit"},
        )
    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert "event: start" in accepted.text


def test_chat_contract_rejects_blank_message_and_oversized_history() -> None:
    with pytest.raises(ValueError):
        ChatRequest(message="   ")
    with pytest.raises(ValueError):
        ChatRequest(
            message="hello",
            history=[{"role": "user", "content": "x" * 8_000}] * 6,
        )
