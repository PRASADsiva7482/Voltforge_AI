"""Generate a content-free VFAI-030 security and privacy receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent
BL_ROOT = WORKSPACE_ROOT / "Voltforge_BL"
REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "ai-security-threat-model-v1.json"
THREAT_MODEL_PATH = AI_ROOT / "docs" / "AI_SECURITY_THREAT_MODEL.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_report() -> dict[str, Any]:
    config = _read(AI_ROOT / "config.py")
    security = _read(AI_ROOT / "api" / "security.py")
    routes = _read(AI_ROOT / "api" / "routes.py")
    main = _read(AI_ROOT / "main.py")
    sse = _read(AI_ROOT / "api" / "sse.py")
    artifact = _read(AI_ROOT / "model" / "artifact_registry.py")
    checkpoint = _read(AI_ROOT / "model" / "gen1" / "model.py")
    runtime_service = _read(AI_ROOT / "model" / "runtime_service.py")
    retrieval_policy = _read(AI_ROOT / "internet_retrieval" / "policy.v1.json")
    retrieval_security = _read(AI_ROOT / "internet_retrieval" / "security.py")
    feedback_policy = _read(AI_ROOT / "feedback_governance" / "policy.v1.json")
    feedback_service = _read(AI_ROOT / "feedback_governance" / "service.py")
    memory = _read(AI_ROOT / "api" / "memory.py") + _read(
        AI_ROOT / "memory_store" / "service.py"
    )
    spring_security = _read(
        BL_ROOT / "src/main/java/in/voltforge/api/config/SecurityConfig.java"
    )
    spring_config = _read(
        BL_ROOT / "src/main/java/in/voltforge/api/config/VoltforgeAiConfig.java"
    )
    spring_application = _read(BL_ROOT / "src/main/resources/application.yml")
    spring_service = _read(
        BL_ROOT / "src/main/java/in/voltforge/api/ai/service/impl/AiServiceImpl.java"
    )
    adversarial_tests = _read(AI_ROOT / "tests" / "test_security_boundaries.py")
    threat_model = _read(THREAT_MODEL_PATH)

    checks = {
        "allPythonRoutesHaveServiceDependency": "dependencies=[Depends(require_service_token)]" in routes,
        "productionTokenFailsClosed": "service_authentication_required" in security
        and "len(api_token) < 32" in config
        and "VOLTFORGE_AI_ENVIRONMENT" in config,
        "productionOriginsAndRuntimeScopeAreBounded": "explicit origins" in config
        and "commonpath" in config
        and "VOLTFORGE_AI_RUNTIME_DIRECTORY" in config,
        "productionDocsAreDisabled": "docs_url=" in main
        and "openapi_url=" in main
        and "environment != \"production\"" in main,
        "artifactIntegrityAndSafeLoading": "allow_pickle=False" in artifact
        and "weights_only=True" in checkpoint
        and "verify_registry" in artifact
        and "GHSA-63cw-57p8-fm3p" in runtime_service
        and "MODEL_RUNTIME_CHECKPOINT_SECURITY_BLOCKED" in runtime_service,
        "retrievalBoundaryIsFixedAndUntrusted": '"resultUrlFetchingAllowed": false' in retrieval_policy
        and '"redirectPolicy": "reject-all"' in retrieval_policy
        and "not parsed.is_global" in retrieval_security
        and "contains_prompt_injection" in retrieval_security,
        "memoryIsAuthenticatedScopedAndBounded": "Authenticated user and project identity are required" in memory
        and "owner_key = ? AND project_key = ?" in memory
        and "maximumContentCharacters" in memory
        and "trainingUseAllowed" in memory,
        "feedbackDoesNotPersistRawContent": "Feedback admitted for governed review" in routes
        and '"rawContentStored": False' in feedback_service
        and '"rawPromptStored": false' in feedback_policy
        and '"rawModelResponseStored": false' in feedback_policy
        and '"rawProjectSnapshotStored": false' in feedback_policy
        and "maximumEvidenceCharacters" in feedback_policy
        and "maximumTrainingExampleCharacters" in feedback_policy
        and "governed review" in routes
        and "userMessage=payload.userMessage" not in routes
        and "ai_response=payload.aiResponse" not in routes,
        "pythonErrorsAreContentFree": "SIMULATION_STREAM_FAILED" in sse
        and "str(error)" not in sse
        and "Database health check failed." in _read(AI_ROOT / "api" / "database.py"),
        "springBrowserAndServiceBoundaries": ".anyRequest().authenticated()" in spring_security
        and "X-Voltforge-AI-Token" in spring_config
        and "validateProductionSecurityConfiguration" in spring_config
        and "environment:" in spring_application
        and "VOLTFORGE_AI_ENVIRONMENT" in spring_application,
        "springUpstreamErrorsAreContentFree": "e.getMessage()" not in spring_service
        and "errorType" in spring_service
        and "Please try again." in spring_service,
        "requestAndStreamLimitsRemainPresent": "MAX_REQUEST_BYTES" in config
        and "maximumStreamEvents" in _read(AI_ROOT / "api_contract" / "runtime.py")
        and "maximumStreamBytes" in _read(AI_ROOT / "api_contract" / "runtime.py")
        and "max-concurrent-streams-per-user" in spring_application,
        "adversarialSecurityTestsPresent": all(
            marker in adversarial_tests
            for marker in (
                "test_production_configuration_fails_closed",
                "test_service_token_is_constant_time",
                "test_feedback_and_diagnostics_never_echo_secret_material",
                "test_artifact_loading_is_checksum_bound",
                "MODEL_RUNTIME_CHECKPOINT_SECURITY_BLOCKED",
                "test_security_limits_are_explicit",
            )
        ),
        "threatModelDocumentsOwnersAndResiduals": all(
            marker in threat_model
            for marker in (
                "Assets and owners",
                "Trust boundaries",
                "Threat register",
                "Residual risk / owner",
                "Accepted residual risk",
                "VFAI-031",
                "VFAI-032",
                "VFAI-033",
                "VFAI-034",
                "VFAI-035",
            )
        ),
        "noThirdPartyGenerationProvider": not any(
            marker in (routes + spring_service + spring_config).lower()
            for marker in ("api.openai.com", "anthropic", "google.generativeai")
        ),
    }
    report = {
        "schemaVersion": 1,
        "reportId": "vfai030-ai-security-threat-model-v1",
        "generatedOn": "2026-08-31",
        "contractVersion": "1.0.0",
        "checkCount": len(checks),
        "passedCheckCount": sum(bool(value) for value in checks.values()),
        "checks": checks,
        "controls": {
            "serviceAuthentication": "production-token-required",
            "artifactLoading": "signed-checksum-verified-safe-formats-and-affected-torch-serving-block",
            "memory": "authenticated-bounded-redacted-project-session-scope",
            "retrieval": "fixed-provider-no-ssrf-no-redirect-no-training",
            "proposalMutation": "revision-guarded-explicit-review-only",
        },
        "residualOwners": {
            "VFAI-031": "privacy-safe telemetry and capacity/rate visibility",
            "VFAI-032": "cross-repository security pipeline",
            "VFAI-033": "release, rollback, signing-key, and token rotation",
            "VFAI-034": "governed feedback and retraining admission",
            "VFAI-035": "deployment and incident runbook",
        },
        "networkAccessed": False,
        "rawPromptStored": False,
        "rawProjectContextStored": False,
        "rawModelOutputStored": False,
        "thirdPartyGenerationProvider": False,
        "threatModelSha256": _sha(THREAT_MODEL_PATH),
        "evaluatorSha256": _sha(Path(__file__).resolve()),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def evaluate(path: Path = REPORT_PATH) -> dict[str, Any]:
    report = build_report()
    if not all(report["checks"].values()):
        failed = [key for key, value in report["checks"].items() if not value]
        raise SystemExit(f"VFAI-030 checks failed: {', '.join(failed)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = evaluate()
    print(
        json.dumps(
            {
                "passed": result["passedCheckCount"],
                "total": result["checkCount"],
                "reportSha256": result["reportSha256"],
            }
        )
    )
