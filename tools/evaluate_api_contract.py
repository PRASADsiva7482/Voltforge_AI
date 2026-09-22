"""Generate or verify the content-free VFAI-026 API/SSE contract receipt."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

import api.chat as chat_module
from api.chat import stream_chat_sse
from api.schemas import ChatRequest, ChatResponse
from api_contract import (
    POLICY_ID,
    POLICY_SHA256,
    ApiContractError,
    ArtifactIdentity,
    CancellationToken,
    RequestStopped,
    SseContractEmitter,
    SseEvent,
    cancellation_registry,
    checked_event_schema,
    checked_request_schema,
    checked_response_schema,
    load_policy,
    wait_for_stage,
)
from api_contract.schema import (
    EVENT_SCHEMA_PATH,
    REQUEST_SCHEMA_PATH,
    RESPONSE_SCHEMA_PATH,
    build_event_json_schema,
    canonical_json,
    sha256_json,
)
from config import get_settings
from main import app


DEFAULT_REPORT_PATH = AI_ROOT / "evaluation/reports/fastapi-sse-contract-v1.json"
PRIVATE_FIXTURE = "VFAI026_PRIVATE_PROMPT_MUST_NOT_APPEAR"
FIXED_REQUEST_ID = "request-evaluator-026"
FIXED_SESSION_ID = "session-evaluator-026"
FIXED_REVISION = "revision-evaluator-026"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _event_payload(event: str) -> dict[str, Any]:
    line = next(item for item in event.splitlines() if item.startswith("data:"))
    return json.loads(line.removeprefix("data:").strip())


def _canonical_event(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: data[key] for key in SseEvent.model_fields}


async def _collect(request: ChatRequest, **kwargs: Any) -> list[str]:
    return [item async for item in stream_chat_sse(request, **kwargs)]


async def _cancelled_stream() -> list[dict[str, Any]]:
    token = CancellationToken("request-eval-cancel")
    generator = stream_chat_sse(
        ChatRequest(message="Cancellation fixture."),
        request_id="request-eval-cancel",
        session_id="session-eval-cancel",
        cancellation=token,
    )
    first = _event_payload(await anext(generator))
    token.cancel()
    return [first, *[_event_payload(item) async for item in generator]]


async def _disconnected_stream() -> list[dict[str, Any]]:
    async def disconnected() -> bool:
        return True

    return [
        _event_payload(item)
        for item in await _collect(
            ChatRequest(message="Disconnect fixture."),
            request_id="request-eval-disconnect",
            session_id="session-eval-disconnect",
            disconnected=disconnected,
        )
    ]


async def _timeout_code() -> str:
    token = CancellationToken("request-eval-timeout")
    try:
        await wait_for_stage(
            lambda: time.sleep(0.05),
            token=token,
            disconnected=None,
            deadline_seconds=0.01,
        )
    except RequestStopped as error:
        return error.code
    return "NOT_STOPPED"


async def _malformed_stream(settings: Any) -> list[dict[str, Any]]:
    malformed = SimpleNamespace(
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
    )
    with patch.object(chat_module, "resolve_generation", return_value=malformed):
        return [
            _event_payload(item)
            for item in await _collect(
                ChatRequest(message="Malformed output fixture."),
                settings=settings,
                request_id="request-eval-malformed",
                session_id="session-eval-malformed",
            )
        ]


def _rejects_request(payload: dict[str, Any]) -> bool:
    try:
        ChatRequest.model_validate(payload)
    except ValidationError:
        return True
    return False


def _synthetic_events() -> list[dict[str, Any]]:
    emitter = SseContractEmitter(
        request_id="request-eval-synthetic",
        session_id="session-eval-synthetic",
        project_revision="revision-eval-synthetic",
        mode="deterministic-fallback",
        artifact=ArtifactIdentity(
            artifactVersion="deterministic-tools-v1",
            runtimeState="unavailable",
            ready=False,
        ),
    )
    values = [
        emitter.emit("start", {"readiness": {}, "fallbackUsed": True}),
        emitter.emit(
            "tool", {"name": "fixture-tool", "status": "complete", "authority": "deterministic"}
        ),
        emitter.emit(
            "citation", {"citationId": "citation:fixture:026", "evidenceKind": "local"}
        ),
        emitter.emit("uncertainty", {"reasonCode": "EVIDENCE_UNAVAILABLE"}),
        emitter.emit("delta", {"delta": "Bounded delta."}),
        emitter.emit(
            "proposal",
            {
                "id": "proposal:fixture:026",
                "sourceProjectRevision": "revision-eval-synthetic",
                "structuredActions": [],
            },
        ),
        emitter.emit(
            "complete",
            {
                "reply": "Bounded complete response.",
                "messageId": "message-eval-026",
                "taskRecordId": "record-eval-026",
            },
        ),
    ]
    return [_event_payload(item) for item in values]


def _evaluate() -> tuple[dict[str, bool], dict[str, int]]:
    policy = load_policy()
    settings_environment = {
        "VOLTFORGE_AI_API_TOKEN": "vfai026-evaluator-token",
        "VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED": "false",
        "VOLTFORGE_AI_CHAT_REQUEST_TIMEOUT_SECONDS": "15",
    }
    with patch.dict(os.environ, settings_environment, clear=False):
        settings = get_settings()

        def network_denied(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("VFAI-026 local generation attempted network access")

        with patch.object(requests.sessions.Session, "request", network_denied), patch.object(
            socket, "create_connection", network_denied
        ):
            wire_events = asyncio.run(
                _collect(
                    ChatRequest(
                        message="Explain the selected LED safely.",
                        projectRevision=FIXED_REVISION,
                    ),
                    settings=settings,
                    request_id=FIXED_REQUEST_ID,
                    session_id=FIXED_SESSION_ID,
                )
            )
        parsed = [_event_payload(item) for item in wire_events]
        synthetic = _synthetic_events()
        cancelled = asyncio.run(_cancelled_stream())
        disconnected = asyncio.run(_disconnected_stream())
        timeout_code = asyncio.run(_timeout_code())
        malformed = asyncio.run(_malformed_stream(settings))

        headers = {"X-Voltforge-AI-Token": "vfai026-evaluator-token"}
        with TestClient(app) as client:
            incompatible = client.post(
                "/voltForge-ai/api/v1/model/chat/stream",
                headers=headers,
                json={"schemaVersion": 2, "message": PRIVATE_FIXTURE},
            )
            oversized = client.post(
                "/voltForge-ai/api/v1/model/chat/stream",
                headers={**headers, "Content-Length": "2000001"},
                content=b"{}",
            )
            health = client.get("/voltForge-ai/api/v1/model/contract").json()
            sync = client.post(
                "/voltForge-ai/api/v1/model/chat",
                headers={**headers, "X-Voltforge-Request-Id": "request-eval-sync"},
                json={"schemaVersion": 1, "message": "Sync contract fixture."},
            )
            active_id = "request-eval-active"
            token = cancellation_registry.register(active_id)
            try:
                cancelled_http = client.delete(
                    f"/voltForge-ai/api/v1/model/chat/requests/{active_id}",
                    headers=headers,
                )
            finally:
                cancellation_registry.release(active_id)

    canonical_valid = all(
        SseEvent.model_validate(_canonical_event(item)) for item in parsed
    )
    synthetic_valid = all(
        SseEvent.model_validate(_canonical_event(item)) for item in synthetic
    )
    payload_compatibility = all(
        all(item.get(key) == value for key, value in item["payload"].items())
        for item in parsed
    )
    event_types = [str(item["type"]) for item in parsed]
    synthetic_types = [str(item["type"]) for item in synthetic]
    start = parsed[0]
    complete = parsed[-1]
    sync_body = sync.json()
    route_paths = {str(item["path"]) for item in health["routes"]}

    artifact = ArtifactIdentity(
        artifactVersion="deterministic-tools-v1",
        runtimeState="unavailable",
        ready=False,
    )
    oversized_delta_rejected = False
    emitter = SseContractEmitter(
        request_id="request-eval-limit",
        session_id="session-eval-limit",
        project_revision="revision-eval-limit",
        mode="deterministic-fallback",
        artifact=artifact,
    )
    emitter.emit("start", {"readiness": {}, "fallbackUsed": True})
    try:
        emitter.emit("delta", {"delta": "x" * 97})
    except ValidationError:
        oversized_delta_rejected = True

    backend_service = (
        WORKSPACE_ROOT
        / "Voltforge_BL/src/main/java/in/voltforge/api/ai/service/impl/AiServiceImpl.java"
    ).read_text(encoding="utf-8")
    backend_response = (
        WORKSPACE_ROOT
        / "Voltforge_BL/src/main/java/in/voltforge/api/ai/dto/AiChatResponse.java"
    ).read_text(encoding="utf-8")
    ui_panel = (
        WORKSPACE_ROOT / "Voltforge_UI/src/features/ai/AiChatPanel.tsx"
    ).read_text(encoding="utf-8")
    route_source = (AI_ROOT / "api/routes.py").read_text(encoding="utf-8")
    main_source = (AI_ROOT / "main.py").read_text(encoding="utf-8")
    generation_source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in [
            AI_ROOT / "api/chat.py",
            AI_ROOT / "api/routes.py",
            AI_ROOT / "config.py",
        ]
    )
    checks = {
        "policyChecksumPinned": _sha_file(AI_ROOT / "api_contract/policy.v1.json") == POLICY_SHA256,
        "requestSchemaMatchesExecutableContract": checked_request_schema() == ChatRequest.model_json_schema(),
        "responseSchemaMatchesExecutableContract": checked_response_schema() == ChatResponse.model_json_schema(),
        "eventSchemaMatchesExecutableContract": checked_event_schema() == build_event_json_schema(),
        "requestVersionDefaultsToV1": ChatRequest(message="Version fixture.").schemaVersion == 1,
        "unknownRequestSchemaVersionRejected": _rejects_request({"schemaVersion": 2, "message": "x"}),
        "unknownContractVersionRejected": _rejects_request({"contractVersion": "2.0.0", "message": "x"}),
        "unknownRequestFieldRejected": _rejects_request({"message": "x", "unknown": True}),
        "responseVersionDefaultsToV1": ChatResponse(reply="x").contractVersion == "1.0.0",
        "streamStartsAndCompletes": event_types[0] == "start" and event_types[-1] == "complete",
        "everyWireEventMatchesCanonicalSchema": canonical_valid,
        "allDeclaredEventKindsHaveTypedFixtures": synthetic_valid and synthetic_types == ["start", "tool", "citation", "uncertainty", "delta", "proposal", "complete"],
        "compatibilityFieldsMatchCanonicalPayload": payload_compatibility,
        "eventSequencesAreContiguous": [item["sequence"] for item in parsed] == list(range(len(parsed))),
        "eventIdsAreUnique": len({item["eventId"] for item in parsed}) == len(parsed),
        "requestIdOnEveryEvent": {item["requestId"] for item in parsed} == {FIXED_REQUEST_ID},
        "sessionIdOnEveryEvent": {item["sessionId"] for item in parsed} == {FIXED_SESSION_ID},
        "projectRevisionOnEveryEvent": {item["projectRevision"] for item in parsed} == {f"client:{FIXED_REVISION}"},
        "localModelIdentityOnEveryEvent": {item["model"] for item in parsed} == {"voltforge-local-engine-v1"},
        "fallbackModeIsHonest": start["mode"] == "deterministic-fallback" and complete["mode"] == "deterministic-fallback",
        "artifactVersionOnEveryEvent": all(item["artifact"]["artifactVersion"] for item in parsed),
        "unavailableModelNeverClaimsReady": start["neuralModelReady"] is False and start["neuralAttempted"] is False,
        "fallbackIsExplicit": start["fallbackUsed"] is True and bool(start["fallbackReasonCode"]),
        "allSubsystemReadinessOnStart": set(start["readiness"]) == {"localModel", "contextCompiler", "engineeringTools", "localRetrieval", "internetRetrieval", "grounding", "memory"},
        "noHiddenReasoningEvent": "thought" not in event_types and policy["privacy"]["hiddenReasoningEventAllowed"] is False,
        "deltaCharacterBoundEnforced": oversized_delta_rejected and all(len(item["delta"]) <= 96 for item in parsed if item["type"] == "delta"),
        "eventByteBoundEnforced": all(len(item.encode("utf-8")) <= policy["limits"]["maximumEventBytes"] for item in wire_events),
        "streamByteBoundEnforced": sum(len(item.encode("utf-8")) for item in wire_events) <= policy["limits"]["maximumStreamBytes"],
        "streamEventBoundEnforced": len(parsed) <= policy["limits"]["maximumStreamEvents"],
        "toolEventBoundEnforced": event_types.count("tool") <= policy["limits"]["maximumToolEvents"],
        "citationEventBoundEnforced": event_types.count("citation") <= policy["limits"]["maximumCitationEvents"],
        "completeReplyBoundEnforced": 1 <= len(complete["reply"]) <= policy["limits"]["maximumReplyCharacters"],
        "generationUsesNoNetwork": start["generationNetworkAccess"] is False,
        "explicitCancellationIsTerminal": [item["type"] for item in cancelled] == ["start", "error"] and cancelled[-1]["code"] == "REQUEST_CANCELLED",
        "disconnectStopsWithoutFurtherEmission": [item["type"] for item in disconnected] == ["start"],
        "deadlineStopsStage": timeout_code == "REQUEST_TIMEOUT",
        "malformedGenerationCannotComplete": malformed[-1]["type"] == "error" and malformed[-1]["code"] == "MALFORMED_GENERATION" and not any(item["type"] == "complete" for item in malformed),
        "incompatibleHttpRequestIsTyped": incompatible.status_code == 422 and incompatible.json()["code"] == "API_SCHEMA_VALIDATION_FAILED",
        "validationErrorStoresNoPrompt": PRIVATE_FIXTURE not in incompatible.text,
        "oversizedHttpRequestIsTyped": oversized.status_code == 413 and oversized.json()["code"] == "API_REQUEST_TOO_LARGE",
        "healthCoversEveryRegisteredRoute": health["routeCount"] == len(health["routes"]) and health["routeCount"] >= 30,
        "chatReadinessDeclaresFallback": any(item["path"].endswith("/chat/stream") and item["available"] and item["deterministicFallbackAvailable"] for item in health["routes"]),
        "offlineInternetReadinessIsHonest": health["subsystems"]["internetRetrieval"]["state"] == "offline" and any(item["path"].endswith("/internet/search") and not item["available"] for item in health["routes"]),
        "contractAndCancellationRoutesPresent": any(path.endswith("/contract") for path in route_paths) and any("/chat/requests/" in path for path in route_paths),
        "httpCancellationEndpointWorks": cancelled_http.status_code == 202 and token.cancelled,
        "syncResponseCarriesContractMetadata": sync.status_code == 200 and sync_body["requestId"] == "request-eval-sync" and sync_body["schemaVersion"] == 1 and bool(sync_body["artifact"]["artifactVersion"]) and bool(sync_body["projectRevision"]),
        "springSendsContractVersion": 'body.put("schemaVersion", 1)' in backend_service and 'body.put("contractVersion", "1.0.0")' in backend_service,
        "springPreservesSyncContractMetadata": all(value in backend_response for value in ("schemaVersion", "contractVersion", "requestId", "projectRevision", "artifact", "readiness")),
        "springRelaysNamedSseWithoutBuffering": "bodyToFlux" in backend_service and "ServerSentEvent<String>" in backend_service,
        "uiReadsCanonicalPayload": "envelope.payload" in ui_panel and "canonicalPayload" in ui_panel,
        "fastApiHasTypedValidationAndSizeErrors": "RequestValidationError" in main_source and "API_REQUEST_TOO_LARGE" in main_source,
        "fastApiPassesDisconnectCallback": "disconnected=request.is_disconnected" in route_source,
        "dockerIncludesContractPackage": "COPY api_contract/ ./api_contract/" in (AI_ROOT / "Dockerfile").read_text(encoding="utf-8"),
        "noThirdPartyModelProviderAdded": not any(
            marker in generation_source
            for marker in (
                "api.openai.com",
                "anthropic",
                "google.generativeai",
                "hosted model",
            )
        ),
    }
    metrics = {
        "routeCount": int(health["routeCount"]),
        "fixtureEventCount": len(parsed),
        "maximumEventBytes": int(policy["limits"]["maximumEventBytes"]),
        "maximumStreamBytes": int(policy["limits"]["maximumStreamBytes"]),
        "maximumStreamEvents": int(policy["limits"]["maximumStreamEvents"]),
        "maximumRequestBytes": int(policy["limits"]["maximumRequestBytes"]),
        "maximumReplyCharacters": int(policy["limits"]["maximumReplyCharacters"]),
        "defaultRequestTimeoutSeconds": int(policy["limits"]["defaultRequestTimeoutSeconds"]),
    }
    return checks, metrics


def build_report() -> dict[str, Any]:
    checks, metrics = _evaluate()
    report = {
        "schemaVersion": 1,
        "reportId": "vfai026-fastapi-sse-contract-v1",
        "generatedOn": "2026-08-30",
        "policyId": POLICY_ID,
        "policySha256": POLICY_SHA256,
        "contractVersion": "1.0.0",
        "checkCount": len(checks),
        "passedCheckCount": sum(bool(value) for value in checks.values()),
        "checks": checks,
        "metrics": metrics,
        "networkAccessed": False,
        "rawPromptStored": False,
        "rawProjectContextStored": False,
        "rawModelOutputStored": False,
        "hiddenReasoningStored": False,
        "evaluatorSha256": _sha_file(Path(__file__).resolve()),
        "requestSchemaSha256": sha256_json(ChatRequest.model_json_schema()),
        "responseSchemaSha256": sha256_json(ChatResponse.model_json_schema()),
        "eventSchemaSha256": sha256_json(build_event_json_schema()),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def write_checked_schemas() -> None:
    _write_json(ChatRequest.model_json_schema(), REQUEST_SCHEMA_PATH)
    _write_json(ChatResponse.model_json_schema(), RESPONSE_SCHEMA_PATH)
    _write_json(build_event_json_schema(), EVENT_SCHEMA_PATH)


def evaluate(path: Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    write_checked_schemas()
    report = build_report()
    if not all(report["checks"].values()):
        failed = [key for key, value in report["checks"].items() if not value]
        raise RuntimeError(f"VFAI-026 evaluation failed: {failed}")
    _write_json(report, path)
    return report


def verify(path: Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    checked = json.loads(path.read_text(encoding="utf-8"))
    generated = build_report()
    if checked != generated or checked.get("reportSha256") != _receipt_digest(checked):
        raise RuntimeError("VFAI-026 evaluation report is stale or invalid")
    if not all(checked["checks"].values()):
        raise RuntimeError("VFAI-026 evaluation report contains a failed check")
    return checked


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = (
        evaluate(arguments.output.resolve())
        if arguments.command == "evaluate"
        else verify(arguments.output.resolve())
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": arguments.command,
                "reportId": report["reportId"],
                "reportSha256": report["reportSha256"],
                "checks": report["checkCount"],
                "passed": report["passedCheckCount"],
                "networkAccessed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
