"""Evaluate and verify the content-free VFAI-019 quality-gate receipt."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
from typing import Any, Callable


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from api.schemas import ChatRequest  # noqa: E402
from engine.reasoning import ElectronicsReasoningOrchestrator  # noqa: E402
from model.generation_quality import (  # noqa: E402
    QUALITY_GATE_POLICY_ID,
    GenerationQualityGate,
    GenerationQualityMetrics,
    resolve_generation,
)
from task_schema.adapters import runtime_request_to_task_record  # noqa: E402


POLICY_PATH = AI_ROOT / "model" / "generation-quality-policy.v1.json"
REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "generation-quality-gates-v1.json"
REGISTRY_PATH = AI_ROOT / "model" / "registry" / "active_model.json"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_policy() -> dict[str, Any]:
    policy = read_json(POLICY_PATH)
    expected_digest = policy.get("policySha256")
    unsigned = dict(policy)
    unsigned.pop("policySha256", None)
    if policy.get("policyId") != QUALITY_GATE_POLICY_ID:
        raise RuntimeError("Generation quality policy ID does not match runtime code")
    if expected_digest != sha256_value(unsigned):
        raise RuntimeError("Generation quality policy digest is invalid")
    if policy.get("currentReleaseBoundary", {}).get("neuralServingApproved") is not False:
        raise RuntimeError("VFAI-019 must not approve neural serving")
    return policy


def _request_record(*, reported: bool = False) -> dict[str, Any]:
    return runtime_request_to_task_record(
        ChatRequest(
            message="Review this LED circuit safely.",
            projectId="project-quality-evaluation",
            projectRevision="revision-19",
            boardType="ARDUINO_UNO",
        ),
        tool_events=[
            {
                "name": "inspect_simulation_state" if reported else "validate_circuit",
                "status": "reported" if reported else "complete",
                "summary": "The checked LED circuit requires a current-limiting resistor.",
                "evidence": {
                    "compiled": False,
                    "issue": "missing-current-limiting-resistor",
                    "approvedActions": [
                        {
                            "actionKind": "wire-suggestion",
                            "payload": {
                                "suggestion": "Add the reviewed current-limiting resistor."
                            },
                        }
                    ],
                },
            }
        ],
    )


def _valid_envelope(record: dict[str, Any]) -> dict[str, Any]:
    evidence_id = record["input"]["toolEvidence"][0]["evidenceId"]
    revision = record["input"]["projectContext"]["sourceProjectRevision"]
    return {
        "schemaVersion": 1,
        "domain": "voltforge-electronics",
        "finishReason": "stop",
        "confidence": {"score": 0.9, "basis": "evidence-aligned"},
        "segments": [
            {
                "type": "grounded-claim",
                "text": "The checked LED circuit requires a current-limiting resistor.",
                "evidenceRefs": [evidence_id],
            }
        ],
        "structuredActions": [
            {
                "type": "structured-action",
                "actionId": "action:qg-evaluation:wire:001",
                "actionKind": "wire-suggestion",
                "sourceProjectRevision": revision,
                "applicationMode": "proposal-only",
                "requiresUserConfirmation": True,
                "evidenceRefs": [evidence_id],
                "payload": {
                    "suggestion": "Add the reviewed current-limiting resistor."
                },
            }
        ],
        "citationEvidenceIds": [evidence_id],
    }


def _case_matrix(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    base = _valid_envelope(record)
    cases: list[tuple[str, str, str]] = [
        ("accepted-grounded-action", json.dumps(base), "QG_ACCEPTED"),
        ("plain-text", "Connect the LED directly.", "QG_SCHEMA_INVALID"),
    ]

    mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
        ("wrong-domain", lambda item: item.update(domain="general-assistant"), "QG_OUT_OF_DOMAIN"),
        ("truncated-finish", lambda item: item.update(finishReason="length"), "QG_TRUNCATED"),
        (
            "missing-claim-evidence",
            lambda item: item["segments"][0].update(evidenceRefs=[]),
            "QG_UNSUPPORTED_CLAIM",
        ),
        (
            "unknown-citation",
            lambda item: item["segments"][0].update(evidenceRefs=["evidence:unknown"]),
            "QG_CITATION_INVALID",
        ),
        (
            "unsupported-electronics-claim",
            lambda item: item["segments"][0].update(
                text="The microcontroller oscillator runs at 240 MHz."
            ),
            "QG_UNSUPPORTED_CLAIM",
        ),
        (
            "irrelevant-claim",
            lambda item: item["segments"][0].update(text="The recipe needs bread flour."),
            "QG_OUT_OF_DOMAIN",
        ),
        (
            "dangerous-instruction",
            lambda item: item["segments"][0].update(text="Bypass the fuse."),
            "QG_DANGEROUS_INSTRUCTION",
        ),
        (
            "repeated-segment",
            lambda item: item["segments"].append(deepcopy(item["segments"][0])),
            "QG_REPETITION",
        ),
        (
            "stale-action-revision",
            lambda item: item["structuredActions"][0].update(
                sourceProjectRevision="client:stale"
            ),
            "QG_ACTION_UNSAFE",
        ),
        (
            "hidden-auto-apply",
            lambda item: item["structuredActions"][0]["payload"].update(
                applyAutomatically=True
            ),
            "QG_ACTION_SCHEMA_INVALID",
        ),
        (
            "malformed-action-evidence",
            lambda item: item["structuredActions"][0].update(
                evidenceRefs=[{"unexpected": "object"}]
            ),
            "QG_ACTION_SCHEMA_INVALID",
        ),
        (
            "unapproved-action-payload",
            lambda item: item["structuredActions"][0].update(
                payload={"suggestion": "Replace the board with an unrelated action."}
            ),
            "QG_ACTION_UNSUPPORTED",
        ),
    ]
    for case_id, mutate, expected in mutations:
        candidate = deepcopy(base)
        mutate(candidate)
        cases.append((case_id, json.dumps(candidate), expected))
    return cases


def _evaluate_cases() -> list[dict[str, Any]]:
    record = _request_record()
    gate = GenerationQualityGate()
    results = []
    for case_id, candidate, expected in _case_matrix(record):
        decision = gate.evaluate(candidate, record)
        results.append(
            {
                "caseId": case_id,
                "expectedCode": expected,
                "actualCode": decision.code,
                "accepted": decision.accepted,
                "typedResponseCreated": decision.response_record is not None,
                "typedActionCount": decision.action_count,
                "passed": decision.code == expected
                and (
                    decision.response_record is not None
                    if expected == "QG_ACCEPTED"
                    else decision.response_record is None
                ),
            }
        )

    reported = _request_record(reported=True)
    overconfident = _valid_envelope(reported)
    overconfidence = gate.evaluate(json.dumps(overconfident), reported)
    results.append(
        {
            "caseId": "client-evidence-overconfidence",
            "expectedCode": "QG_CONFIDENCE_UNSUPPORTED",
            "actualCode": overconfidence.code,
            "accepted": overconfidence.accepted,
            "typedResponseCreated": overconfidence.response_record is not None,
            "typedActionCount": overconfidence.action_count,
            "passed": overconfidence.code == "QG_CONFIDENCE_UNSUPPORTED"
            and overconfidence.response_record is None,
        }
    )
    return results


def _evaluate_fallbacks() -> tuple[dict[str, Any], dict[str, Any]]:
    record = _request_record()
    metrics = GenerationQualityMetrics()
    orchestrator = ElectronicsReasoningOrchestrator(internet_retrieval_enabled=False)
    request = ChatRequest(
        message="Review this LED circuit safely.",
        boardType="ARDUINO_UNO",
    )

    def deterministic_reply():
        return orchestrator.process_chat(
            message=request.message,
            board_type=request.boardType or "ARDUINO_UNO",
            components=request.components,
            wires=request.wires,
            code=request.code or "",
            context={},
        )

    original_connect = socket.socket.connect

    def deny_connect(_socket, _address):
        raise AssertionError("VFAI-019 deterministic fallback attempted network access")

    socket.socket.connect = deny_connect
    try:
        rejected = resolve_generation(
            candidate_text="UNTRUSTED_NEURAL_OUTPUT",
            request_record=record,
            model_health={
                "ready": True,
                "code": "READY",
                "artifactId": "offline-evaluation-artifact",
            },
            deterministic_factory=deterministic_reply,
            metrics=metrics,
        )
        unavailable = resolve_generation(
            candidate_text=None,
            request_record=record,
            model_health={
                "ready": False,
                "code": "NO_APPROVED_MODEL_ARTIFACT",
                "artifactId": None,
            },
            deterministic_factory=deterministic_reply,
            metrics=metrics,
        )
    finally:
        socket.socket.connect = original_connect

    rejected_value = rejected.deterministic_value
    unavailable_value = unavailable.deterministic_value
    result = {
        "rejectedCandidate": {
            "mode": rejected.metadata["mode"],
            "fallbackUsed": rejected.metadata["fallbackUsed"],
            "fallbackReasonCode": rejected.metadata["fallbackReasonCode"],
            "neuralAttempted": rejected.metadata["neuralAttempted"],
            "qualityGateStatus": rejected.metadata["qualityGate"]["status"],
            "neuralTextReturned": rejected.response_text is not None,
            "neuralTypedResponseCreated": rejected.response_record is not None,
            "deterministicReplySha256": hashlib.sha256(
                rejected_value.reply.encode("utf-8")
            ).hexdigest(),
        },
        "unavailableModel": {
            "mode": unavailable.metadata["mode"],
            "fallbackUsed": unavailable.metadata["fallbackUsed"],
            "fallbackReasonCode": unavailable.metadata["fallbackReasonCode"],
            "neuralAttempted": unavailable.metadata["neuralAttempted"],
            "neuralArtifactId": unavailable.metadata["neuralArtifactId"],
            "qualityGateStatus": unavailable.metadata["qualityGate"]["status"],
            "deterministicReplySha256": hashlib.sha256(
                unavailable_value.reply.encode("utf-8")
            ).hexdigest(),
        },
        "generationNetworkAccess": False,
        "rawCandidateStored": False,
        "rawDeterministicReplyStored": False,
    }
    return result, metrics.snapshot()


def run_evaluation(report_path: Path = REPORT_PATH) -> dict[str, Any]:
    policy = validate_policy()
    registry = read_json(REGISTRY_PATH)
    cases = _evaluate_cases()
    fallback, metrics = _evaluate_fallbacks()
    rejected_cases = [item for item in cases if item["expectedCode"] != "QG_ACCEPTED"]
    checks = {
        "allGateCasesPassed": all(item["passed"] for item in cases),
        "invalidCandidatesCreatedNoTypedResponse": all(
            item["typedResponseCreated"] is False for item in rejected_cases
        ),
        "invalidCandidatesCreatedNoTypedAction": all(
            item["typedActionCount"] == 0 for item in rejected_cases
        ),
        "rejectionUsesVisibleFallback": fallback["rejectedCandidate"]["fallbackUsed"] is True
        and fallback["rejectedCandidate"]["fallbackReasonCode"] == "QG_SCHEMA_INVALID"
        and fallback["rejectedCandidate"]["qualityGateStatus"] == "rejected",
        "rejectedNeuralTextNeverReturned": fallback["rejectedCandidate"][
            "neuralTextReturned"
        ]
        is False
        and fallback["rejectedCandidate"]["neuralTypedResponseCreated"] is False,
        "unavailableModelNeverClaimsNeuralAttempt": fallback["unavailableModel"][
            "neuralAttempted"
        ]
        is False
        and fallback["unavailableModel"]["neuralArtifactId"] is None,
        "deterministicFallbacksObserved": metrics["deterministicFallbacks"] == 2,
        "rawContentNotStored": fallback["rawCandidateStored"] is False
        and fallback["rawDeterministicReplyStored"] is False
        and metrics["rawPromptStored"] is False
        and metrics["rawOutputStored"] is False,
        "noActiveNeuralArtifact": registry.get("activeArtifactId") is None,
        "experimentalArtifactRemainsUnserved": policy["currentReleaseBoundary"][
            "experimentalArtifactMayServeChat"
        ]
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"VFAI-019 generation quality evaluation failed: {checks}")

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai019-generation-quality-gates-v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "serviceRegistryRevision": registry.get("revision"),
        "activeNeuralArtifactId": registry.get("activeArtifactId"),
        "caseResults": cases,
        "fallbackEvaluation": fallback,
        "observabilitySnapshot": metrics,
        "checks": checks,
        "neuralServingApproved": False,
        "blockingReason": (
            "VFAI-019 validates the boundary only; the experimental checkpoint remains "
            "below release quality and VFAI-020 project-context compilation is not complete."
        ),
    }
    report["reportSha256"] = sha256_value(report)
    write_json(report_path, report)
    return report


def verify_report(report_path: Path = REPORT_PATH) -> dict[str, Any]:
    policy = validate_policy()
    report = read_json(report_path)
    digest = report.get("reportSha256")
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    if digest != sha256_value(unsigned):
        raise RuntimeError("Generation quality report digest is invalid")
    if report.get("policySha256") != policy["policySha256"]:
        raise RuntimeError("Generation quality report references a different policy")
    if report.get("neuralServingApproved") is not False:
        raise RuntimeError("Generation quality report may not approve neural serving")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise RuntimeError("Generation quality report contains a failed check")
    forbidden_keys = {"rawPrompt", "rawOutput", "candidateText", "replyText"}
    if forbidden_keys.intersection(report):
        raise RuntimeError("Generation quality report stores forbidden raw content")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "verify"))
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run_evaluation(args.report) if args.command == "evaluate" else verify_report(args.report)
    print(
        json.dumps(
            {
                "ok": True,
                "command": args.command,
                "reportId": report["reportId"],
                "reportSha256": report["reportSha256"],
                "checks": report["checks"],
                "neuralServingApproved": report["neuralServingApproved"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
