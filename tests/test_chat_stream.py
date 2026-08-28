import asyncio
import json
from pathlib import Path
import socket

import pytest
import requests
from fastapi.testclient import TestClient

from api.chat import _get_local_orchestrator, stream_chat_sse
from api.schemas import ChatRequest
from config import get_settings
from main import app


async def collect_events(request: ChatRequest) -> list[str]:
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

    assert start["mode"] == "local-deterministic"
    assert start["model"] == "voltforge-local-engine-v1"
    assert start["generationNetworkAccess"] is False
    assert start["internetRetrieval"] == {
        "enabled": False,
        "policy": "optional-evidence-only",
        "generationDependency": False,
    }
    assert any(event.startswith("event: delta") for event in events)
    assert complete["mode"] == "local-deterministic"
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
    assert any(event.startswith("event: tool") and "validate_circuit" in event for event in events)
    assert any(event.startswith("event: tool") and "review_firmware" in event for event in events)
    assert any('"compiled":false' in event for event in events)


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
        ai_root / ".env.example",
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
        "mode": "local-deterministic",
        "networkRequired": False,
    }
    assert data["internetRetrieval"]["policy"] == "optional-evidence-only"
    assert data["internetRetrieval"]["generationDependency"] is False
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
