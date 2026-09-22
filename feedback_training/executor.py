"""Execution-time admission for a VFAI-034 feedback training candidate.

All validation occurs before model or governed-corpus allocation. The current
policy intentionally has no training adapter, so this module can verify and
report admission evidence but cannot start a training run yet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from feedback_governance import FeedbackStore, verify_retraining_run
from model.registry_manager import (
    RegistryManagerError,
    artifact_root_from_entry,
    verify_artifact_directory,
    verify_registry,
)
from model.release_lifecycle import (
    ReleaseLifecycleError,
    validate_release_compatibility,
)


AI_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = AI_ROOT / "feedback_training/policy.v1.json"
CONTEXT_READINESS_PATH = AI_ROOT / "evaluation/reports/context-revision-readiness-v1.json"
REGISTRY_PATH = AI_ROOT / "model/registry/active_model.json"


class FeedbackCandidateExecutorError(RuntimeError):
    """A scheduled feedback candidate failed pre-allocation admission."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_DOCUMENT_INVALID", f"Unable to read {label}."
        ) from exc
    if not isinstance(value, dict):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_DOCUMENT_INVALID", f"{label} must be an object."
        )
    return value


def _digest_valid(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    return isinstance(declared, str) and declared == hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()


def _resolve_workspace_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_DEPENDENCY_INVALID", f"{label} path is missing."
        )
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_DEPENDENCY_INVALID", f"{label} escapes the workspace."
        ) from exc
    return path


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    policy = _read_json(path, "feedback candidate executor policy")
    if (
        policy.get("schemaVersion") != 1
        or policy.get("policyId") != "vfai-fu-009-feedback-candidate-executor-v1"
        or policy.get("sourceFollowup") != "VFAI-FU-009"
    ):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_POLICY_INVALID", "Executor policy identity is invalid."
        )
    return policy


def _verify_context_revision(
    policy: Mapping[str, Any], path: Path = CONTEXT_READINESS_PATH
) -> dict[str, Any]:
    report = _read_json(path, "context revision readiness report")
    required = policy.get("dependencies", {}).get("requiredContextRevisionDecision")
    valid = bool(
        report.get("reportId") == "vfai-fu-008-context-revision-readiness-v1"
        and _digest_valid(report, "reportSha256")
        and report.get("decision") == required
        and report.get("completionClaimed") is False
        and report.get("networkAccessed") is False
        and report.get("releaseOrActivationApproved") is False
    )
    if not valid:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_CONTEXT_BASE_NOT_READY",
            "The governed context-capable model/data revision is not ready.",
        )
    return {
        "reportId": report.get("reportId"),
        "reportSha256": report.get("reportSha256"),
        "decision": report.get("decision"),
    }


def _verify_run_layout(
    run_path: Path, run: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    expected_keys = {
        "schemaVersion",
        "runFormat",
        "baseModel",
        "feedbackInputs",
        "trainingRecordCount",
        "trainingShardSha256",
        "compatibility",
        "deterministic",
        "dependencies",
        "requiredGates",
        "liveChatWeightMutation",
        "runId",
        "state",
        "createdAtUtc",
        "runSha256",
        "trainingPath",
        "trainingSha256",
    }
    if set(run) != expected_keys:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_RUN_SCHEMA_INVALID",
            "The scheduled manifest has missing or unknown fields.",
        )
    run_id = run.get("runId")
    if (
        run.get("runFormat") != policy.get("scheduledRun", {}).get("requiredFormat")
        or run.get("state") != "scheduled"
        or run.get("liveChatWeightMutation") is not False
        or not isinstance(run_id, str)
    ):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_RUN_SCHEMA_INVALID",
            "The scheduled manifest execution boundary is invalid.",
        )
    training_path = Path(str(run.get("trainingPath") or "")).resolve()
    if (
        training_path.parent != run_path.parent
        or training_path.name != f"{run_id}.training.jsonl"
        or run_path.name != f"{run_id}.json"
    ):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_TRAINING_PATH_INVALID",
            "The candidate training shard must be the scheduled manifest sibling.",
        )
    required_gates = policy.get("scheduledRun", {}).get("requiredReleaseGates")
    if run.get("requiredGates") != required_gates:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_RELEASE_GATES_INVALID",
            "The scheduled release gates do not match the executor policy.",
        )
    return {
        "runId": run_id,
        "runSha256": run.get("runSha256"),
        "trainingSha256": run.get("trainingSha256"),
        "trainingRecordCount": run.get("trainingRecordCount"),
        "trainingPathIsSibling": True,
        "requiredReleaseGatesMatch": True,
    }


def _verify_dependencies(
    run: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    dependencies = run.get("dependencies")
    required = policy.get("scheduledRun", {}).get("requiredDependencyPaths")
    if not isinstance(dependencies, list) or not isinstance(required, list):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_DEPENDENCY_INVALID", "Run dependencies are invalid."
        )
    by_path: dict[str, str] = {}
    for item in dependencies:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or item["path"] in by_path
        ):
            raise FeedbackCandidateExecutorError(
                "FEEDBACK_EXECUTOR_DEPENDENCY_INVALID",
                "A scheduled dependency is malformed or duplicated.",
            )
        by_path[item["path"]] = item["sha256"]
    if set(by_path) != set(required):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_DEPENDENCY_INVALID",
            "The scheduled dependency set is incomplete or unexpected.",
        )
    for relative, declared in by_path.items():
        path = _resolve_workspace_path(relative, "scheduled dependency")
        if not path.is_file() or _sha256_file(path) != declared:
            raise FeedbackCandidateExecutorError(
                "FEEDBACK_EXECUTOR_DEPENDENCY_STALE",
                "A scheduled dependency changed after review.",
            )
    return {
        "dependencyCount": len(by_path),
        "dependencySetSha256": hashlib.sha256(
            _canonical({"dependencies": sorted(by_path.items())})
        ).hexdigest(),
        "allCurrent": True,
    }


def _verify_base_artifact(
    run: Mapping[str, Any],
    policy: Mapping[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    try:
        registry = verify_registry(registry_path)
    except RegistryManagerError as exc:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_REGISTRY_INVALID", "The signed model registry is invalid."
        ) from exc
    base = run.get("baseModel")
    if not isinstance(base, Mapping):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_BASE_INVALID", "The scheduled base model is missing."
        )
    artifact_id = base.get("artifactId")
    revision = base.get("registryRevision")
    contract = policy.get("baseArtifact", {})
    entry = next(
        (
            item
            for item in registry.get("artifacts", [])
            if isinstance(item, Mapping) and item.get("artifactId") == artifact_id
        ),
        None,
    )
    if (
        registry.get("revision") != revision
        or registry.get("activeArtifactId") != artifact_id
        or not isinstance(entry, Mapping)
        or entry.get("releaseStatus") != contract.get("releaseStatusRequired")
        or entry.get("activationEligible") is not contract.get(
            "activationEligibleRequired"
        )
        or not isinstance(entry.get("contextLength"), int)
        or entry["contextLength"] < int(contract.get("minimumContextLength", 4096))
    ):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_BASE_INVALID",
            "The scheduled base is not the current approved context-capable artifact.",
        )
    root = artifact_root_from_entry(registry_path, entry)
    try:
        verified = verify_artifact_directory(root)
    except RegistryManagerError as exc:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_BASE_INVALID", "The signed base artifact is invalid."
        ) from exc
    tokenizer = verified.manifest.get("tokenizer", {})
    if (
        tokenizer.get("tokenizerId") != contract.get("tokenizerId")
        or tokenizer.get("version") != contract.get("tokenizerVersion")
    ):
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_BASE_TOKENIZER_INVALID",
            "The base artifact tokenizer is incompatible with feedback training.",
        )
    return {
        "artifactId": artifact_id,
        "registryRevision": revision,
        "registrySha256": registry.get("registrySha256"),
        "manifestSha256": verified.manifest_sha256,
        "contextLength": verified.context_length,
        "tokenizerId": tokenizer.get("tokenizerId"),
        "tokenizerVersion": tokenizer.get("version"),
        "signedAndActive": True,
    }


def inspect_scheduled_candidate(
    run_path: str | Path,
    feedback_database_path: str | Path,
    *,
    policy_path: Path = POLICY_PATH,
    context_readiness_path: Path = CONTEXT_READINESS_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Return content-free pre-allocation evidence for one scheduled run.

    The function never loads a model or the approved static corpus and never
    writes to the feedback database, memory, registry, or artifact directories.
    """

    policy = load_policy(policy_path)
    context = _verify_context_revision(policy, context_readiness_path)
    manifest_path = Path(run_path).resolve()
    verify_retraining_run(manifest_path)
    run = _read_json(manifest_path, "scheduled feedback run")
    layout = _verify_run_layout(manifest_path, run, policy)
    dependencies = _verify_dependencies(run, policy)
    try:
        validate_release_compatibility(run.get("compatibility"))
    except ReleaseLifecycleError as exc:
        raise FeedbackCandidateExecutorError(
            "FEEDBACK_EXECUTOR_COMPATIBILITY_STALE",
            "The scheduled runtime compatibility contract is stale.",
        ) from exc
    base = _verify_base_artifact(run, policy, registry_path.resolve())
    store = FeedbackStore(feedback_database_path)
    feedback = store.verify_execution_admission(run, run_path=manifest_path)
    adapter_path = policy.get("isolation", {}).get("trainingAdapterPath")
    adapter_assigned = isinstance(adapter_path, str) and bool(adapter_path)
    return {
        "schemaVersion": 1,
        "evidenceKind": "vfai-fu-009-feedback-candidate-admission-v1",
        "policyId": policy.get("policyId"),
        "contextRevision": context,
        "scheduledRun": layout,
        "dependencies": dependencies,
        "compatibilityCurrent": True,
        "baseArtifact": base,
        "feedbackAdmission": feedback,
        "trainingAdapterAssigned": adapter_assigned,
        "allocationAuthorized": adapter_assigned,
        "decision": "ready-for-isolated-training-allocation"
        if adapter_assigned
        else "await-context-training-adapter",
        "modelWeightsLoaded": False,
        "governedStaticCorpusLoaded": False,
        "stateMutated": False,
        "networkAccessed": False,
        "releaseOrActivationApproved": False,
    }
