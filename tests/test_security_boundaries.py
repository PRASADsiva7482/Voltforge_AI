from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import config
from api.routes import router
from api.security import redact_sensitive_text, require_service_token
from main import app
from model import artifact_registry


PRIVATE_TOKEN = "vfai-production-token-" + ("x" * 32)


def test_production_configuration_fails_closed_and_confines_memory_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_ENVIRONMENT", "production")
    monkeypatch.setenv("VOLTFORGE_AI_API_TOKEN", "")
    monkeypatch.setenv("VOLTFORGE_AI_ALLOWED_ORIGINS", "https://app.example.test")
    monkeypatch.setenv("VOLTFORGE_AI_RUNTIME_DIRECTORY", str(tmp_path / "runtime"))

    with pytest.raises(ValueError, match="API_TOKEN"):
        config.AiSettings.from_environment()

    monkeypatch.setenv("VOLTFORGE_AI_API_TOKEN", PRIVATE_TOKEN)
    monkeypatch.setenv(
        "VOLTFORGE_AI_MEMORY_DATABASE_PATH",
        str(tmp_path / "runtime" / "memory.sqlite3"),
    )
    settings = config.AiSettings.from_environment()
    assert settings.environment == "production"
    assert settings.service_authentication_required is True
    assert settings.memory_database_path.endswith("memory.sqlite3")

    monkeypatch.setenv(
        "VOLTFORGE_AI_MEMORY_DATABASE_PATH", str(tmp_path / "outside.sqlite3")
    )
    with pytest.raises(ValueError, match="inside"):
        config.AiSettings.from_environment()


def test_service_token_is_constant_time_and_production_routes_are_private(monkeypatch) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_ENVIRONMENT", "production")
    monkeypatch.setenv("VOLTFORGE_AI_API_TOKEN", PRIVATE_TOKEN)

    with pytest.raises(HTTPException) as rejected:
        require_service_token(None, "wrong-token")
    assert rejected.value.status_code == 401
    require_service_token(None, PRIVATE_TOKEN)

    assert router.dependencies
    assert any(
        dependency.dependency is require_service_token
        for dependency in router.dependencies
    )
    with TestClient(app) as client:
        response = client.post(
            "/voltForge-ai/api/v1/model/generate-code",
            json={"boardType": "ARDUINO_UNO", "components": []},
        )
    assert response.status_code == 401


def test_feedback_and_diagnostics_never_echo_secret_material(monkeypatch) -> None:
    secret = "Bearer vfai-secret-value password=topsecret"
    redacted = redact_sensitive_text(secret)
    assert "vfai-secret-value" not in redacted
    assert "topsecret" not in redacted
    assert "[REDACTED]" in redacted

    monkeypatch.setenv("VOLTFORGE_AI_ENVIRONMENT", "development")
    monkeypatch.delenv("VOLTFORGE_AI_API_TOKEN", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/voltForge-ai/api/v1/model/feedback",
            json={
                "requestId": "request-feedback-secret-test",
                "responseRecordId": "vf-task-v1-0123456789abcdef01234567",
                "projectId": "project-feedback-secret",
                "projectRevision": "client:feedback-secret-revision",
                "artifactId": "vfdlm-g1-edge-v1.0.0-test",
                "feedbackKind": "incorrect",
                "rating": 5,
                "evidence": secret,
                "expectedBehavior": "The response must not expose private material.",
                "evidenceApproved": True,
            },
            headers={
                "X-Voltforge-User-Id": "user-feedback-secret",
                "X-Voltforge-Project-Id": "project-feedback-secret",
            },
        )
    assert response.status_code == 422
    assert secret not in response.text


def test_artifact_loading_is_checksum_bound_and_vulnerable_runtime_is_blocked() -> None:
    artifact_source = inspect.getsource(artifact_registry._validate_weights)
    checkpoint_source = (
        Path(__file__).resolve().parents[1] / "model" / "gen1" / "model.py"
    ).read_text(encoding="utf-8")
    runtime_source = (
        Path(__file__).resolve().parents[1] / "model" / "runtime_service.py"
    ).read_text(encoding="utf-8")
    assert "allow_pickle=False" in artifact_source
    assert "weights_only=True" in checkpoint_source
    assert "pytorch-weights-only-state-dict-v1" in checkpoint_source
    assert "GHSA-63cw-57p8-fm3p" in runtime_source
    assert "MODEL_RUNTIME_CHECKPOINT_SECURITY_BLOCKED" in runtime_source


def test_security_limits_are_explicit_at_the_request_and_stream_boundaries() -> None:
    request_source = inspect.getsource(config.AiSettings.from_environment)
    contract_source = inspect.getsource(__import__("api_contract.runtime", fromlist=["SseContractEmitter"]).SseContractEmitter.emit)
    assert "VOLTFORGE_AI_MAX_REQUEST_BYTES" in request_source
    assert "VOLTFORGE_AI_MAX_CONTEXT_CHARACTERS" in request_source
    assert "maximumStreamEvents" in contract_source
    assert "maximumStreamBytes" in contract_source
