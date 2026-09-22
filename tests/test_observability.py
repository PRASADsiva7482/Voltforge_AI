"""Adversarial and contract tests for VFAI-031 telemetry."""

import json

from fastapi.testclient import TestClient

from main import app
from observability import ObservabilityRegistry, route_label


def test_telemetry_is_bounded_and_does_not_retain_request_content(caplog) -> None:
    registry = ObservabilityRegistry(
        latency_alert_ms=1,
        active_request_alert=1,
        fallback_alert_count=1,
        retrieval_error_alert_count=1,
    )
    sentinel = "VFAI031_PRIVATE_PROMPT_AND_PROJECT_CONTENT"
    request = registry.begin_request(sentinel, f"/voltForge-ai/api/v1/model/chat/{sentinel}")
    registry.finish_request(request, status_code=200, duration_ms=20)
    registry.record_generation(
        mode="deterministic-fallback",
        fallback_used=True,
        fallback_reason=sentinel,
        neural_attempted=True,
        tool_grounded=False,
        artifact_id=sentinel,
    )
    registry.record_retrieval(status="degraded", failure_code=sentinel, duration_ms=3)

    serialized = json.dumps(registry.snapshot(), sort_keys=True)
    assert sentinel not in serialized
    assert route_label(f"/voltForge-ai/api/v1/model/chat/{sentinel}") == "unknown"
    assert registry.snapshot()["alerts"]["state"]["latencySaturation"] is True
    assert registry.snapshot()["alerts"]["state"]["retrievalErrors"] is True
    assert "rawPromptStored" in registry.snapshot()
    assert registry.snapshot()["rawPromptStored"] is False
    assert sentinel not in "\n".join(caplog.messages)


def test_stream_metrics_record_first_token_capacity_and_outcome() -> None:
    registry = ObservabilityRegistry()
    stream = registry.start_stream("stream-vfai031")
    registry.observe_stream_event(stream, "start", 100)
    registry.observe_stream_event(stream, "delta", 120)
    registry.observe_stream_event(stream, "complete", 180)
    registry.finish_stream(stream, outcome="success", output_characters=42, input_tokens=12)

    snapshot = registry.snapshot()
    assert snapshot["streams"]["completed"] == 1
    assert snapshot["streams"]["eventCount"] == 3
    assert snapshot["streams"]["deltaCount"] == 1
    assert snapshot["streams"]["firstTokenLatencyMs"]["count"] == 1
    assert snapshot["tokens"]["inputCount"] == 12
    assert snapshot["tokens"]["estimated"] is False


def test_metrics_endpoint_is_content_free() -> None:
    with TestClient(app) as client:
        response = client.get("/voltForge-ai/api/v1/model/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["schemaVersion"] == "vfai031-observability-v1"
    assert body["rawPromptStored"] is False
    assert body["rawProjectContentStored"] is False
    assert "requests" in body
    assert "streams" in body
    assert "alerts" in body
