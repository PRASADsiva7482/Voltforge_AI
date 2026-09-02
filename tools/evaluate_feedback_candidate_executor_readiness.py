"""Generate or verify the VFAI-FU-009 feedback-executor readiness receipt.

This evaluator performs content-free source and capability checks only. It does
not open the feedback database, load a training shard, load model weights or the
governed corpus, allocate training state, or modify registry/runtime state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from model.registry_manager import RegistryManagerError, verify_registry  # noqa: E402


MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "feedback_training/policy.v1.json"
CONTEXT_READINESS_PATH = AI_ROOT / "evaluation/reports/context-revision-readiness-v1.json"
REGISTRY_PATH = AI_ROOT / "model/registry/active_model.json"
FEEDBACK_POLICY_PATH = AI_ROOT / "feedback_governance/policy.v1.json"
FEEDBACK_SERVICE_PATH = AI_ROOT / "feedback_governance/service.py"
EXECUTOR_PATH = AI_ROOT / "feedback_training/executor.py"
DEFAULT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/feedback-candidate-executor-readiness-v1.json"
)
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class FeedbackExecutorReadinessError(RuntimeError):
    """The feedback-executor readiness evidence is invalid or stale."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declared_digest_valid(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    return isinstance(declared, str) and declared == hashlib.sha256(
        _canonical_json(unsigned)
    ).hexdigest()


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeedbackExecutorReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise FeedbackExecutorReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _backlog_item(backlog: Mapping[str, Any], item_id: str) -> Mapping[str, Any]:
    items = backlog.get("items")
    if not isinstance(items, list):
        raise FeedbackExecutorReadinessError("follow-up backlog items are invalid")
    matches = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("id") == item_id
    ]
    if len(matches) != 1:
        raise FeedbackExecutorReadinessError(f"expected exactly one {item_id} item")
    return matches[0]


def _master_complete(master: Mapping[str, Any]) -> bool:
    items = master.get("items")
    return bool(
        master.get("status") in COMPLETE_STATUSES
        and isinstance(items, list)
        and items
        and all(
            isinstance(item, Mapping) and item.get("status") in COMPLETE_STATUSES
            for item in items
        )
    )


def _context_evidence(
    report: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    required = policy.get("dependencies", {}).get("requiredContextRevisionDecision")
    receipt_valid = bool(
        report.get("reportId") == "vfai-fu-008-context-revision-readiness-v1"
        and _declared_digest_valid(report, "reportSha256")
        and report.get("completionClaimed") is False
        and report.get("modelWeightsLoadedByEvaluator") is False
        and report.get("governedCorpusLoadedByEvaluator") is False
        and report.get("networkAccessed") is False
        and report.get("releaseOrActivationApproved") is False
    )
    return {
        "reportId": report.get("reportId"),
        "reportSha256": report.get("reportSha256"),
        "receiptValid": receipt_valid,
        "decision": report.get("decision"),
        "requiredDecision": required,
        "contextCapableBaseRevisionReady": receipt_valid
        and report.get("decision") == required,
        "currentApprovedCorpusStage": report.get("corpus", {}).get(
            "currentApprovedStage"
        ),
        "currentUniqueTrainingPredictedTokens": report.get("corpus", {}).get(
            "currentUniqueTrainingPredictedTokens"
        ),
    }


def _registry_evidence(registry_path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    try:
        registry = verify_registry(registry_path)
        signature_valid = True
    except RegistryManagerError:
        registry = {}
        signature_valid = False
    active_id = registry.get("activeArtifactId")
    entry = next(
        (
            item
            for item in registry.get("artifacts", [])
            if isinstance(item, Mapping) and item.get("artifactId") == active_id
        ),
        None,
    )
    contract = policy.get("baseArtifact", {})
    base_ready = bool(
        signature_valid
        and active_id is not None
        and isinstance(entry, Mapping)
        and entry.get("releaseStatus") == contract.get("releaseStatusRequired")
        and entry.get("activationEligible") is contract.get(
            "activationEligibleRequired"
        )
        and isinstance(entry.get("contextLength"), int)
        and entry["contextLength"] >= int(contract.get("minimumContextLength", 4096))
    )
    return {
        "signatureValid": signature_valid,
        "revision": registry.get("revision"),
        "registrySha256": registry.get("registrySha256"),
        "activeArtifactId": active_id,
        "activeContextLength": entry.get("contextLength")
        if isinstance(entry, Mapping)
        else None,
        "activeReleaseStatus": entry.get("releaseStatus")
        if isinstance(entry, Mapping)
        else None,
        "activeActivationEligible": entry.get("activationEligible")
        if isinstance(entry, Mapping)
        else None,
        "activeStableContextCapableBaseReady": base_ready,
    }


def _implementation_evidence(
    policy: Mapping[str, Any], feedback_policy: Mapping[str, Any]
) -> dict[str, Any]:
    executor_source = EXECUTOR_PATH.read_text(encoding="utf-8")
    service_source = FEEDBACK_SERVICE_PATH.read_text(encoding="utf-8")
    required_markers = (
        "verify_retraining_run",
        "validate_release_compatibility",
        "verify_registry",
        "verify_artifact_directory",
        "verify_execution_admission",
    )
    service_markers = (
        "consentAndReviewReverified",
        "currentHeldoutLeakageRejected",
        "stateMutated",
    )
    required_dependencies = policy.get("scheduledRun", {}).get(
        "requiredDependencyPaths"
    )
    required_gates = policy.get("scheduledRun", {}).get("requiredReleaseGates")
    implementation_ready = bool(
        EXECUTOR_PATH.is_file()
        and FEEDBACK_SERVICE_PATH.is_file()
        and all(marker in executor_source for marker in required_markers)
        and all(marker in service_source for marker in service_markers)
        and required_gates == feedback_policy.get("retraining", {}).get(
            "requiredReleaseGates"
        )
        and isinstance(required_dependencies, list)
        and {
            "feedback_training/policy.v1.json",
            "feedback_training/executor.py",
            "gen1_training/data.py",
            "gen1_training/trainer.py",
        }.issubset(set(required_dependencies))
    )
    return {
        "executorPath": EXECUTOR_PATH.relative_to(AI_ROOT).as_posix(),
        "executorSha256": _sha256_file(EXECUTOR_PATH),
        "feedbackServiceSha256": _sha256_file(FEEDBACK_SERVICE_PATH),
        "requiredDependencyCount": len(required_dependencies or []),
        "requiredReleaseGates": required_gates,
        "preAllocationAdmissionImplemented": implementation_ready,
        "failedValidationMayMutateState": policy.get("executionAdmission", {}).get(
            "failedValidationMayMutateState"
        ),
    }


def _assigned_inputs(policy: Mapping[str, Any]) -> dict[str, Any]:
    scheduled = policy.get("scheduledRun", {})
    isolation = policy.get("isolation", {})
    manifest_path = scheduled.get("manifestPath")
    database_path = scheduled.get("feedbackDatabasePath")
    adapter_path = isolation.get("trainingAdapterPath")
    result_path = isolation.get("candidateResultReceiptPath")
    return {
        "scheduledRunManifestPath": manifest_path,
        "scheduledRunManifestAssigned": isinstance(manifest_path, str)
        and bool(manifest_path),
        "feedbackDatabasePath": database_path,
        "feedbackDatabaseAssigned": isinstance(database_path, str) and bool(database_path),
        "trainingAdapterPath": adapter_path,
        "trainingAdapterAssigned": isinstance(adapter_path, str) and bool(adapter_path),
        "candidateResultReceiptPath": result_path,
        "candidateResultReceiptAssigned": isinstance(result_path, str) and bool(result_path),
    }


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    context: Mapping[str, Any],
    registry: Mapping[str, Any],
    implementation: Mapping[str, Any],
    inputs: Mapping[str, Any],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-009":
        raise FeedbackExecutorReadinessError("policy is not bound to VFAI-FU-009")
    fu009 = _backlog_item(followup_backlog, "VFAI-FU-009")
    master_ready = _master_complete(master_backlog)
    gates = {
        "masterBacklogComplete": master_ready,
        "contextReadinessReceiptValid": context.get("receiptValid") is True,
        "contextCapableBaseRevisionReady": context.get(
            "contextCapableBaseRevisionReady"
        )
        is True,
        "signedActiveStableBaseReady": registry.get(
            "activeStableContextCapableBaseReady"
        )
        is True,
        "preAllocationAdmissionImplemented": implementation.get(
            "preAllocationAdmissionImplemented"
        )
        is True,
        "scheduledRunManifestAssigned": inputs.get("scheduledRunManifestAssigned")
        is True,
        "feedbackDatabaseAssigned": inputs.get("feedbackDatabaseAssigned") is True,
        "trainingAdapterAssigned": inputs.get("trainingAdapterAssigned") is True,
        "candidateResultReceiptAssigned": inputs.get("candidateResultReceiptAssigned")
        is True,
    }
    if not master_ready:
        decision = "await-master-backlog-completion"
    elif gates["contextReadinessReceiptValid"] is not True:
        decision = "repair-context-readiness-evidence"
    elif gates["contextCapableBaseRevisionReady"] is not True:
        decision = "await-context-capable-base-revision"
    elif gates["signedActiveStableBaseReady"] is not True:
        decision = "await-active-stable-base-artifact"
    elif gates["preAllocationAdmissionImplemented"] is not True:
        decision = "repair-pre-allocation-admission"
    elif gates["scheduledRunManifestAssigned"] is not True:
        decision = "await-governed-feedback-run"
    elif gates["feedbackDatabaseAssigned"] is not True:
        decision = "await-feedback-database-binding"
    elif gates["trainingAdapterAssigned"] is not True:
        decision = "implement-context-training-adapter"
    elif gates["candidateResultReceiptAssigned"] is not True:
        decision = "ready-for-isolated-feedback-candidate-execution"
    else:
        decision = "candidate-results-ready-for-independent-release-review"
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-009-feedback-candidate-executor-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": fu009.get("id"), "recordedStatus": fu009.get("status")},
        "policy": {
            "policyId": policy.get("policyId"),
            "status": policy.get("status"),
        },
        "contextRevision": dict(context),
        "registry": dict(registry),
        "implementation": dict(implementation),
        "assignedInputs": dict(inputs),
        "gates": gates,
        "decision": decision,
        "completionClaimed": False,
        "feedbackDatabaseOpenedByEvaluator": False,
        "scheduledTrainingShardLoadedByEvaluator": False,
        "modelWeightsLoadedByEvaluator": False,
        "governedCorpusLoadedByEvaluator": False,
        "trainingAllocationsExecutedByEvaluator": 0,
        "candidateArtifactsCreatedByEvaluator": 0,
        "liveStateMutated": False,
        "networkAccessed": False,
        "releaseOrActivationApproved": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def _current_inputs() -> dict[str, Any]:
    master = _read_json_object(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json_object(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json_object(POLICY_PATH, "feedback executor policy")
    context_report = _read_json_object(CONTEXT_READINESS_PATH, "context readiness report")
    feedback_policy = _read_json_object(FEEDBACK_POLICY_PATH, "feedback policy")
    source_paths = (
        MASTER_BACKLOG_PATH,
        FOLLOWUP_BACKLOG_PATH,
        POLICY_PATH,
        CONTEXT_READINESS_PATH,
        REGISTRY_PATH,
        FEEDBACK_POLICY_PATH,
        FEEDBACK_SERVICE_PATH,
        EXECUTOR_PATH,
        Path(__file__),
    )
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "context": _context_evidence(context_report, policy),
        "registry": _registry_evidence(REGISTRY_PATH, policy),
        "implementation": _implementation_evidence(policy, feedback_policy),
        "inputs": _assigned_inputs(policy),
        "source_sha256": {
            path.relative_to(AI_ROOT).as_posix(): _sha256_file(path)
            for path in source_paths
        },
    }


def _build_current_report(generated_on: str) -> dict[str, Any]:
    return build_report(generated_on=generated_on, **_current_inputs())


def evaluate(path: Path, generated_on: str) -> dict[str, Any]:
    report = _build_current_report(generated_on)
    _write_json(report, path)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _receipt_digest(report):
        raise FeedbackExecutorReadinessError("VFAI-FU-009 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        raise FeedbackExecutorReadinessError("VFAI-FU-009 readiness gates are invalid")
    if gates.get("masterBacklogComplete") is not True:
        expected = "await-master-backlog-completion"
    elif gates.get("contextReadinessReceiptValid") is not True:
        expected = "repair-context-readiness-evidence"
    elif gates.get("contextCapableBaseRevisionReady") is not True:
        expected = "await-context-capable-base-revision"
    elif gates.get("signedActiveStableBaseReady") is not True:
        expected = "await-active-stable-base-artifact"
    elif gates.get("preAllocationAdmissionImplemented") is not True:
        expected = "repair-pre-allocation-admission"
    elif gates.get("scheduledRunManifestAssigned") is not True:
        expected = "await-governed-feedback-run"
    elif gates.get("feedbackDatabaseAssigned") is not True:
        expected = "await-feedback-database-binding"
    elif gates.get("trainingAdapterAssigned") is not True:
        expected = "implement-context-training-adapter"
    elif gates.get("candidateResultReceiptAssigned") is not True:
        expected = "ready-for-isolated-feedback-candidate-execution"
    else:
        expected = "candidate-results-ready-for-independent-release-review"
    if report.get("decision") != expected:
        raise FeedbackExecutorReadinessError("VFAI-FU-009 receipt decision is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("feedbackDatabaseOpenedByEvaluator") is not False,
            report.get("scheduledTrainingShardLoadedByEvaluator") is not False,
            report.get("modelWeightsLoadedByEvaluator") is not False,
            report.get("governedCorpusLoadedByEvaluator") is not False,
            report.get("trainingAllocationsExecutedByEvaluator") != 0,
            report.get("candidateArtifactsCreatedByEvaluator") != 0,
            report.get("liveStateMutated") is not False,
            report.get("networkAccessed") is not False,
            report.get("releaseOrActivationApproved") is not False,
        )
    ):
        raise FeedbackExecutorReadinessError("VFAI-FU-009 receipt overclaims execution")


def verify(path: Path) -> dict[str, Any]:
    report = _read_json_object(path, "VFAI-FU-009 readiness receipt")
    validate_report(report)
    expected = _build_current_report(str(report.get("generatedOn")))
    if report != expected:
        raise FeedbackExecutorReadinessError("VFAI-FU-009 readiness receipt is stale")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--generated-on", required=True)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = (
        evaluate(arguments.output.resolve(), arguments.generated_on)
        if arguments.command == "evaluate"
        else verify(arguments.input.resolve())
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": arguments.command,
                "reportId": report["reportId"],
                "decision": report["decision"],
                "contextDecision": report["contextRevision"]["decision"],
                "activeArtifactId": report["registry"]["activeArtifactId"],
                "preAllocationAdmissionImplemented": report["implementation"][
                    "preAllocationAdmissionImplemented"
                ],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
