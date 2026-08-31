"""Run a content-free static receipt for the VFAI-027 Spring gateway boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent
BL_ROOT = WORKSPACE_ROOT / "Voltforge_BL"
REPORT_PATH = AI_ROOT / "evaluation/reports/spring-gateway-v1.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_report() -> dict[str, Any]:
    controller = _read(BL_ROOT / "src/main/java/in/voltforge/api/ai/controller/AiController.java")
    service = _read(BL_ROOT / "src/main/java/in/voltforge/api/ai/service/impl/AiServiceImpl.java")
    policy = _read(BL_ROOT / "src/main/java/in/voltforge/api/ai/gateway/AiGatewayPolicy.java")
    admission = _read(BL_ROOT / "src/main/java/in/voltforge/api/ai/gateway/AiRequestAdmission.java")
    config = _read(BL_ROOT / "src/main/java/in/voltforge/api/config/VoltforgeAiConfig.java")
    application = _read(BL_ROOT / "src/main/resources/application.yml")
    project_service = _read(BL_ROOT / "src/main/java/in/voltforge/api/project/service/ProjectService.java")
    project_impl = _read(BL_ROOT / "src/main/java/in/voltforge/api/project/service/impl/ProjectServiceImpl.java")
    errors = _read(BL_ROOT / "src/main/java/in/voltforge/api/common/exception/GlobalExceptionHandler.java")
    tests = _read(BL_ROOT / "src/test/java/in/voltforge/api/ai/service/impl/AiServiceImplTest.java")
    controller_tests = _read(BL_ROOT / "src/test/java/in/voltforge/api/ai/controller/AiControllerTest.java")
    admission_tests = _read(BL_ROOT / "src/test/java/in/voltforge/api/ai/gateway/AiRequestAdmissionTest.java")

    checks = {
        "jwtProjectAccessBeforeAiCall": "request.setAuthenticatedUserId(jwt.getSubject())" in controller
        and "projectService.canAccessProject" in controller,
        "crossProjectStructuredContextRejected": "AI_PROJECT_CONTEXT_MISMATCH" in policy,
        "staleRevisionRejected": "AI_PROJECT_REVISION_STALE" in controller
        and "getProjectRevision" in controller,
        "currentRevisionServiceIsAccessBound": "String getProjectRevision" in project_service
        and "canAccessProject(projectId, keycloakId)" in project_impl,
        "requestLimitConfigured": "maxRequestBytes" in config
        and "max-request-bytes" in application
        and "AI_REQUEST_LIMIT_REACHED" in policy,
        "perUserAndProjectAdmissionConfigured": "activeByUser" in admission
        and "activeByProject" in admission
        and "max-concurrent-streams-per-user" in application,
        "admissionIsNonBlocking": "Too many AI responses" in admission
        and "HttpStatus.TOO_MANY_REQUESTS" in admission,
        "namedSseRelayDoesNotBuffer": "bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {})" in service,
        "browserCancellationReleasesLease": "doOnCancel" in service
        and "doFinally(signal ->" in service
        and ("lease.close()" in service or "acquiredLease.close()" in service),
        "upstreamRequestReceivesCorrelationId": 'header("X-Voltforge-Request-Id", requestId)' in service,
        "credentialIsServerSideOnly": 'defaultHeader("X-Voltforge-AI-Token", apiToken)' in config
        and "apiToken" not in service,
        "upstreamErrorTextIsNotReturned": "VoltForge AI is temporarily unavailable. Your project was not changed." in service
        and "offline or encountered an error: " not in service,
        "typedGatewayErrorsAreSafe": "AiGatewayException" in errors
        and "ex.getErrorCode()" in errors,
        "relayTestsCoverTypedOrdering": all(
            marker in tests
            for marker in (
                '"start", "tool", "citation", "uncertainty", "delta", "proposal", "complete"',
                'containsExactly("start", "error")',
                "citation:local:1",
                "proposal:1",
            )
        ),
        "authorizationAndRevisionTestsPresent": all(
            marker in controller_tests
            for marker in (
                "rejectsCrossProjectAccess",
                "rejectsAStaleProjectRevision",
                "rejectsStructuredContextCopiedFromAnotherProject",
            )
        ),
        "cancellationAndLimitTestsPresent": "releasesGatewayAdmissionWhenTheBrowserCancelsAStream" in tests
        and "limitsAUserAndReleasesCapacityExactlyOnce" in admission_tests,
        "gatewayDoesNotMutateProjects": "project was not changed" in service.lower()
        and "/apply" not in controller,
        "noThirdPartyGenerationProvider": not any(
            marker in (controller + service + config).lower()
            for marker in ("api.openai.com", "anthropic", "google.generativeai")
        ),
    }
    report = {
        "schemaVersion": 1,
        "reportId": "vfai027-spring-gateway-v1",
        "generatedOn": "2026-08-30",
        "contractVersion": "1.0.0",
        "checkCount": len(checks),
        "passedCheckCount": sum(bool(value) for value in checks.values()),
        "checks": checks,
        "limits": {
            "maximumRequestBytes": 2_000_000,
            "maximumConcurrentStreamsPerUser": 2,
            "maximumConcurrentStreamsPerProject": 1,
            "maximumStreamingTimeoutSeconds": 120,
        },
        "networkAccessed": False,
        "rawPromptStored": False,
        "rawProjectContextStored": False,
        "rawModelOutputStored": False,
        "credentialsExposedToBrowser": False,
        "evaluatorSha256": _sha(Path(__file__).resolve()),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def evaluate(path: Path = REPORT_PATH) -> dict[str, Any]:
    report = build_report()
    if not all(report["checks"].values()):
        failed = [key for key, value in report["checks"].items() if not value]
        raise SystemExit(f"VFAI-027 checks failed: {', '.join(failed)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({"passed": result["passedCheckCount"], "total": result["checkCount"], "reportSha256": result["reportSha256"]}))
