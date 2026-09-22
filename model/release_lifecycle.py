"""Signed model release decisions layered over the immutable artifact registry.

The artifact registry answers whether a package is intact and compatible with
the runtime.  This module answers whether that package is allowed to move
through the operational release lifecycle.  Release records are immutable,
signed, content-free metadata; model weights, prompts, and generated text are
never copied into them.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from model.registry_manager import (
    DEFAULT_REGISTRY_PATH,
    DEFAULT_TRUST_STORE_PATH,
    RegistryManagerError,
    RuntimeContract,
    activate_artifact,
    artifact_root_from_entry,
    json_file_bytes,
    rollback_registry,
    sign_document,
    verify_artifact_directory,
    verify_registry,
    verify_signed_document,
)
from model.identity import parse_artifact_id


MODEL_ROOT = Path(__file__).resolve().parent
AI_ROOT = MODEL_ROOT.parent
RELEASE_POLICY_PATH = MODEL_ROOT / "release-policy.v1.json"
DEFAULT_RELEASE_DIRECTORY = MODEL_ROOT / "registry" / "releases"
DEFAULT_RETRIEVAL_INDEX_PATH = AI_ROOT / "local_retrieval" / "index" / "v1" / "index.json"
RELEASE_SCHEMA_VERSION = 1
RELEASE_STATES = frozenset({"experimental", "candidate", "stable", "rejected"})
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
SEMVER_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class ReleaseLifecycleError(RuntimeError):
    """Safe, typed failure for release decisions and rollout operations."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseLifecycleError(code, f"Required release document is invalid: {path.name}.") from error


def load_release_policy(path: Path = RELEASE_POLICY_PATH) -> dict[str, Any]:
    policy = _read_json(path, "MODEL_RELEASE_POLICY_INVALID")
    if not isinstance(policy, dict) or policy.get("schemaVersion") != RELEASE_SCHEMA_VERSION:
        raise ReleaseLifecycleError("MODEL_RELEASE_POLICY_INVALID", "The model release policy schema is unsupported.")
    lifecycle = policy.get("lifecycle")
    if not isinstance(lifecycle, dict) or set(lifecycle.get("states", [])) != RELEASE_STATES:
        raise ReleaseLifecycleError("MODEL_RELEASE_POLICY_INVALID", "The model release state set is incomplete.")
    if policy.get("policyId") != "vfai033-model-release-lifecycle-v1":
        raise ReleaseLifecycleError("MODEL_RELEASE_POLICY_INVALID", "The model release policy identity is invalid.")
    return policy


def _require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ReleaseLifecycleError("MODEL_RELEASE_SCORECARD_INVALID", f"{field} must be a lowercase SHA-256 value.")
    return value


def _semver_major(value: Any, field: str) -> int:
    if not isinstance(value, str):
        raise ReleaseLifecycleError("MODEL_RELEASE_COMPATIBILITY_INVALID", f"{field} must be semantic version text.")
    match = SEMVER_PATTERN.fullmatch(value)
    if match is None:
        raise ReleaseLifecycleError("MODEL_RELEASE_COMPATIBILITY_INVALID", f"{field} must use MAJOR.MINOR.PATCH.")
    return int(match.group(1))


def current_release_compatibility(
    *,
    runtime_contract: RuntimeContract | None = None,
    api_policy_path: Path = AI_ROOT / "api_contract" / "policy.v1.json",
    data_schema_path: Path = AI_ROOT / "task_schema" / "task-record.schema.json",
    retrieval_index_path: Path = DEFAULT_RETRIEVAL_INDEX_PATH,
) -> dict[str, Any]:
    """Return the exact application/model/index contract of this checkout."""

    runtime = (runtime_contract or RuntimeContract.current()).as_compatibility()
    api_policy = _read_json(api_policy_path, "MODEL_RELEASE_API_CONTRACT_INVALID")
    data_schema = _read_json(data_schema_path, "MODEL_RELEASE_DATA_SCHEMA_INVALID")
    retrieval_index = _read_json(retrieval_index_path, "MODEL_RELEASE_RETRIEVAL_INDEX_INVALID")
    try:
        api_properties = api_policy["compatibility"]
        data_properties = data_schema["properties"]
        api = {
            "schemaVersion": int(api_policy["schemaVersion"]),
            "contractVersion": str(api_policy["contractVersion"]),
            "acceptedRequestSchemaVersions": list(api_properties["acceptedRequestSchemaVersions"]),
            "emittedResponseSchemaVersion": int(api_properties["emittedResponseSchemaVersion"]),
        }
        data = {
            "schemaVersion": int(data_properties["schemaVersion"]["const"]),
            "contractVersion": str(data_properties["contractVersion"]["const"]),
        }
        retrieval = {
            "contractVersion": str(retrieval_index["contractVersion"]),
            "indexId": str(retrieval_index["indexId"]),
            "indexVersion": str(retrieval_index["indexVersion"]),
            "indexSha256": str(retrieval_index["indexSha256"]),
            "policyId": str(retrieval_index["policyId"]),
            "policySha256": str(retrieval_index["policySha256"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseLifecycleError("MODEL_RELEASE_COMPATIBILITY_INVALID", "The current application contract is incomplete.") from error
    return {
        "runtime": runtime,
        "api": api,
        "data": data,
        "tokenizer": {
            "tokenizerId": runtime["tokenizerId"],
            "contractVersion": runtime["tokenizerContractVersion"],
        },
        "retrievalIndex": retrieval,
        "migrationPolicyId": "vfai033-migrations-v1",
    }


def validate_release_compatibility(
    compatibility: Mapping[str, Any],
    *,
    current: Mapping[str, Any] | None = None,
) -> None:
    """Reject runtime, API, data, tokenizer, or index drift before serving."""

    if not isinstance(compatibility, Mapping):
        raise ReleaseLifecycleError("MODEL_RELEASE_COMPATIBILITY_INVALID", "Release compatibility must be an object.")
    expected = current or current_release_compatibility()
    if compatibility.get("migrationPolicyId") != expected.get("migrationPolicyId"):
        raise ReleaseLifecycleError("MODEL_RELEASE_MIGRATION_REQUIRED", "This release uses an unsupported migration policy.")
    if compatibility.get("runtime") != expected.get("runtime"):
        raise ReleaseLifecycleError("MODEL_RELEASE_COMPATIBILITY_INVALID", "Runtime compatibility is not exact.")

    candidate_api = compatibility.get("api")
    expected_api = expected.get("api")
    if not isinstance(candidate_api, Mapping) or not isinstance(expected_api, Mapping):
        raise ReleaseLifecycleError("MODEL_RELEASE_COMPATIBILITY_INVALID", "API compatibility is missing.")
    if _semver_major(candidate_api.get("contractVersion"), "api.contractVersion") != _semver_major(expected_api.get("contractVersion"), "api.contractVersion"):
        raise ReleaseLifecycleError("MODEL_RELEASE_API_MIGRATION_REQUIRED", "An API major-version migration is required before activation.")
    if candidate_api != expected_api:
        raise ReleaseLifecycleError("MODEL_RELEASE_API_MIGRATION_REQUIRED", "API schema or contract changes require an explicit migration release.")

    for group, code in (("data", "MODEL_RELEASE_DATA_MIGRATION_REQUIRED"), ("tokenizer", "MODEL_RELEASE_TOKENIZER_INCOMPATIBLE"), ("retrievalIndex", "MODEL_RELEASE_INDEX_MIGRATION_REQUIRED")):
        if compatibility.get(group) != expected.get(group):
            raise ReleaseLifecycleError(code, f"{group} compatibility does not match the serving contract.")


def _validate_scorecard(scorecard: Mapping[str, Any], artifact_manifest_sha256: str) -> None:
    if not isinstance(scorecard, Mapping):
        raise ReleaseLifecycleError("MODEL_RELEASE_SCORECARD_INVALID", "A signed release scorecard is required.")
    for field in ("reportSha256", "scorecardSha256", "artifactManifestSha256"):
        _require_hash(scorecard.get(field), f"scorecard.{field}")
    if scorecard["artifactManifestSha256"] != artifact_manifest_sha256:
        raise ReleaseLifecycleError("MODEL_RELEASE_SCORECARD_ARTIFACT_MISMATCH", "The scorecard is for a different artifact manifest.")
    if scorecard.get("releaseDecision") != "pass":
        raise ReleaseLifecycleError("MODEL_RELEASE_SCORECARD_NOT_APPROVED", "Only a passing release scorecard may be promoted.")
    for field in ("criticalCaseFailures", "failedMetricIds"):
        if scorecard.get(field) != []:
            raise ReleaseLifecycleError("MODEL_RELEASE_SCORECARD_NOT_APPROVED", f"Release scorecard field {field} is not empty.")


def _validate_canary(canary: Mapping[str, Any], scorecard_sha256: str, policy: Mapping[str, Any]) -> None:
    if not isinstance(canary, Mapping) or canary.get("status") != "pass":
        raise ReleaseLifecycleError("MODEL_RELEASE_CANARY_NOT_APPROVED", "A passing canary receipt is required for stability.")
    if canary.get("scorecardSha256") != scorecard_sha256:
        raise ReleaseLifecycleError("MODEL_RELEASE_CANARY_SCORECARD_MISMATCH", "The canary is bound to a different scorecard.")
    canary_policy = policy["canary"]
    observations = canary.get("observations")
    if isinstance(observations, bool) or not isinstance(observations, int) or observations < int(canary_policy["minimumObservations"]):
        raise ReleaseLifecycleError("MODEL_RELEASE_CANARY_INSUFFICIENT", "The canary has too few observations.")
    for field in ("errorRate", "contractFailureRate", "p95LatencyMs"):
        value = canary.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ReleaseLifecycleError("MODEL_RELEASE_CANARY_INVALID", f"Canary field {field} is invalid.")
    if canary["errorRate"] > float(canary_policy["maximumErrorRate"]):
        raise ReleaseLifecycleError("MODEL_RELEASE_CANARY_NOT_APPROVED", "The canary error rate exceeds policy.")
    if canary["contractFailureRate"] > float(canary_policy["maximumContractFailureRate"]):
        raise ReleaseLifecycleError("MODEL_RELEASE_CANARY_NOT_APPROVED", "The canary contract failure rate exceeds policy.")
    if canary.get("safetyFailureCount") != int(canary_policy["maximumSafetyFailureCount"]):
        raise ReleaseLifecycleError("MODEL_RELEASE_CANARY_NOT_APPROVED", "The canary contains a safety failure.")
    if canary["p95LatencyMs"] > float(canary_policy["maximumP95LatencyMs"]):
        raise ReleaseLifecycleError("MODEL_RELEASE_CANARY_NOT_APPROVED", "The canary latency exceeds policy.")


def rollback_reasons(metrics: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> list[str]:
    """Return deterministic rollback reasons without mutating the active release."""

    if not isinstance(metrics, Mapping):
        raise ReleaseLifecycleError("MODEL_RELEASE_ROLLBACK_METRICS_INVALID", "Rollback metrics must be an object.")

    def metric_float(field: str, default: float) -> float:
        value = metrics.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ReleaseLifecycleError("MODEL_RELEASE_ROLLBACK_METRICS_INVALID", f"Rollback metric {field} is invalid.")
        return float(value)

    def metric_count(field: str) -> int:
        value = metrics.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReleaseLifecycleError("MODEL_RELEASE_ROLLBACK_METRICS_INVALID", f"Rollback metric {field} is invalid.")
        return value

    def metric_flag(field: str) -> bool | None:
        value = metrics.get(field)
        if value is not None and not isinstance(value, bool):
            raise ReleaseLifecycleError("MODEL_RELEASE_ROLLBACK_METRICS_INVALID", f"Rollback metric {field} is invalid.")
        return value

    selected_policy = policy or load_release_policy()
    canary_policy = selected_policy["canary"]
    reasons: list[str] = []
    if metric_count("safetyFailureCount") > 0:
        reasons.append("SAFETY_FAILURE")
    if metric_float("contractFailureRate", 0.0) > float(canary_policy["maximumContractFailureRate"]):
        reasons.append("CONTRACT_FAILURE_RATE")
    if metric_float("errorRate", 0.0) > float(canary_policy["maximumErrorRate"]):
        reasons.append("ERROR_RATE")
    if metric_float("p95LatencyMs", 0.0) > float(canary_policy["maximumP95LatencyMs"]):
        reasons.append("P95_LATENCY")
    if metric_flag("artifactIntegrity") is False:
        reasons.append("ARTIFACT_INTEGRITY")
    if metric_flag("compatibility") is False:
        reasons.append("COMPATIBILITY_FAILURE")
    return reasons


def _transition_allowed(from_state: str, to_state: str, policy: Mapping[str, Any]) -> None:
    allowed = policy["lifecycle"]["transitions"].get(from_state, [])
    if to_state not in allowed:
        raise ReleaseLifecycleError("MODEL_RELEASE_INVALID_TRANSITION", f"Release transition {from_state}->{to_state} is not allowed.")


def _release_record_path(release_dir: Path, artifact_id: str, state: str) -> Path:
    return release_dir / f"{artifact_id}.{state}.json"


def _atomic_write(path: Path, data: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_signed_record(
    record: Mapping[str, Any],
    output_path: Path,
    *,
    private_key_path: Path,
    key_id: str,
) -> dict[str, Any]:
    signed = sign_document(record, digest_field="releaseSha256", private_key_path=private_key_path, key_id=key_id)
    _atomic_write(output_path, json_file_bytes(signed))
    return signed


def _registry_artifact(
    registry_path: Path,
    artifact_id: str,
    *,
    trust_store_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    registry = verify_registry(registry_path, trust_store_path=trust_store_path)
    entry = next((item for item in registry["artifacts"] if item.get("artifactId") == artifact_id), None)
    if entry is None:
        raise ReleaseLifecycleError("MODEL_RELEASE_ARTIFACT_NOT_REGISTERED", f"Artifact {artifact_id!r} is not registered.")
    return registry, artifact_root_from_entry(registry_path, entry), entry


def _verify_release_record(
    path: Path,
    *,
    trust_store_path: Path,
    expected_state: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    runtime_contract: RuntimeContract | None = None,
    verify_artifact: bool = False,
) -> tuple[dict[str, Any], Any | None]:
    record = _read_json(path.resolve(), "MODEL_RELEASE_RECORD_INVALID")
    try:
        verify_signed_document(record, digest_field="releaseSha256", trust_store_path=trust_store_path, invalid_code="MODEL_RELEASE_SIGNATURE_INVALID")
    except RegistryManagerError as error:
        raise ReleaseLifecycleError(error.code, error.message) from error
    if not isinstance(record, dict) or record.get("schemaVersion") != RELEASE_SCHEMA_VERSION:
        raise ReleaseLifecycleError("MODEL_RELEASE_RECORD_INVALID", "Release record schema is unsupported.")
    artifact_id = record.get("artifactId")
    try:
        parse_artifact_id(artifact_id)
    except (TypeError, ValueError) as error:
        raise ReleaseLifecycleError("MODEL_RELEASE_RECORD_INVALID", "Release record artifact identity is invalid.") from error
    state = record.get("state")
    if state not in RELEASE_STATES or (expected_state is not None and state != expected_state):
        raise ReleaseLifecycleError("MODEL_RELEASE_INVALID_STATE", "Release record state is invalid for this operation.")
    transition = record.get("transition")
    if not isinstance(transition, dict) or transition.get("toState") != state or not isinstance(transition.get("revision"), int):
        raise ReleaseLifecycleError("MODEL_RELEASE_RECORD_INVALID", "Release transition metadata is invalid.")
    policy = load_release_policy()
    if state != "experimental":
        _transition_allowed(str(transition.get("fromState")), state, policy)
    manifest_sha = _require_hash(record.get("artifactManifestSha256"), "artifactManifestSha256")
    _validate_scorecard(record.get("scorecard"), manifest_sha)
    validate_release_compatibility(record.get("compatibility"))
    if state == "stable":
        _validate_canary(record.get("canary"), record["scorecard"]["scorecardSha256"], policy)
    if verify_artifact:
        registry, root, _entry = _registry_artifact(registry_path.resolve(), artifact_id, trust_store_path=trust_store_path)
        try:
            artifact = verify_artifact_directory(root, trust_store_path=trust_store_path, runtime_contract=runtime_contract, require_activation=state == "stable")
        except RegistryManagerError as error:
            raise ReleaseLifecycleError(error.code, error.message) from error
        if artifact.manifest_sha256 != manifest_sha or artifact.manifest_file_sha256 != record.get("artifactManifestFileSha256"):
            raise ReleaseLifecycleError("MODEL_RELEASE_ARTIFACT_MISMATCH", "Release record does not match the immutable artifact.")
        return record, (registry, artifact)
    return record, None


def create_candidate_release(
    artifact_id: str,
    *,
    scorecard: Mapping[str, Any],
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    compatibility: Mapping[str, Any] | None = None,
    output_path: Path | None = None,
    runtime_contract: RuntimeContract | None = None,
) -> dict[str, Any]:
    """Create a signed candidate decision without changing the active pointer."""

    registry, root, _entry = _registry_artifact(registry_path.resolve(), artifact_id, trust_store_path=trust_store_path)
    try:
        artifact = verify_artifact_directory(root, trust_store_path=trust_store_path, runtime_contract=runtime_contract)
    except RegistryManagerError as error:
        raise ReleaseLifecycleError(error.code, error.message) from error
    _validate_scorecard(scorecard, artifact.manifest_sha256)
    selected_compatibility = deepcopy(dict(compatibility or current_release_compatibility(runtime_contract=runtime_contract)))
    validate_release_compatibility(selected_compatibility)
    policy = load_release_policy()
    record = {
        "schemaVersion": RELEASE_SCHEMA_VERSION,
        "releaseId": f"vf-release-{artifact_id}-candidate-r{int(registry['revision']) + 1}",
        "artifactId": artifact_id,
        "artifactManifestSha256": artifact.manifest_sha256,
        "artifactManifestFileSha256": artifact.manifest_file_sha256,
        "state": "candidate",
        "transition": {"fromState": "experimental", "toState": "candidate", "revision": 1},
        "registry": {"revision": registry["revision"], "registrySha256": registry["registrySha256"]},
        "scorecard": dict(scorecard),
        "compatibility": selected_compatibility,
        "canary": None,
        "rollout": policy["rollout"],
        "rollbackPolicy": {"triggers": policy["rollbackTriggers"]},
        "createdAtUtc": _utc_now(),
    }
    destination = output_path or _release_record_path(DEFAULT_RELEASE_DIRECTORY, artifact_id, "candidate")
    return _write_signed_record(record, destination, private_key_path=private_key_path, key_id=key_id)


def promote_candidate_to_stable(
    candidate_path: Path,
    *,
    canary: Mapping[str, Any],
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    output_path: Path | None = None,
    runtime_contract: RuntimeContract | None = None,
) -> dict[str, Any]:
    """Promote only a signed passing candidate and passing canary to stable."""

    candidate, _ = _verify_release_record(candidate_path, trust_store_path=trust_store_path, expected_state="candidate", registry_path=registry_path, runtime_contract=runtime_contract)
    registry, root, _entry = _registry_artifact(registry_path.resolve(), candidate["artifactId"], trust_store_path=trust_store_path)
    if candidate.get("registry", {}).get("registrySha256") != registry.get("registrySha256"):
        raise ReleaseLifecycleError("MODEL_RELEASE_REGISTRY_CHANGED", "The catalog changed after candidate approval; reissue the candidate.")
    try:
        artifact = verify_artifact_directory(root, trust_store_path=trust_store_path, runtime_contract=runtime_contract, require_activation=True)
    except RegistryManagerError as error:
        raise ReleaseLifecycleError(error.code, error.message) from error
    if artifact.manifest_sha256 != candidate["artifactManifestSha256"]:
        raise ReleaseLifecycleError("MODEL_RELEASE_ARTIFACT_MISMATCH", "Candidate artifact content changed.")
    policy = load_release_policy()
    _validate_canary(canary, candidate["scorecard"]["scorecardSha256"], policy)
    stable = deepcopy(candidate)
    stable.update(
        {
            "releaseId": f"vf-release-{candidate['artifactId']}-stable-r{int(candidate['transition']['revision']) + 1}",
            "state": "stable",
            "transition": {"fromState": "candidate", "toState": "stable", "revision": int(candidate["transition"]["revision"]) + 1},
            "canary": dict(canary),
            "approvedAtUtc": _utc_now(),
        }
    )
    stable.pop("releaseSha256", None)
    destination = output_path or _release_record_path(candidate_path.resolve().parent, candidate["artifactId"], "stable")
    return _write_signed_record(stable, destination, private_key_path=private_key_path, key_id=key_id)


def reject_candidate(
    candidate_path: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    candidate, _ = _verify_release_record(candidate_path, trust_store_path=trust_store_path, expected_state="candidate", registry_path=registry_path)
    rejected = deepcopy(candidate)
    rejected.update(
        {
            "releaseId": f"vf-release-{candidate['artifactId']}-rejected-r{int(candidate['transition']['revision']) + 1}",
            "state": "rejected",
            "transition": {"fromState": "candidate", "toState": "rejected", "revision": int(candidate["transition"]["revision"]) + 1},
            "rejectedAtUtc": _utc_now(),
        }
    )
    rejected.pop("releaseSha256", None)
    destination = output_path or _release_record_path(candidate_path.resolve().parent, candidate["artifactId"], "rejected")
    return _write_signed_record(rejected, destination, private_key_path=private_key_path, key_id=key_id)


def activate_stable_release(
    stable_path: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    expected_revision: int | None = None,
    runtime_contract: RuntimeContract | None = None,
) -> dict[str, Any]:
    """Atomically activate a stable release and bind its release ID to history."""

    stable, verified = _verify_release_record(stable_path, trust_store_path=trust_store_path, expected_state="stable", registry_path=registry_path, runtime_contract=runtime_contract, verify_artifact=True)
    registry, _artifact = verified
    if stable.get("registry", {}).get("registrySha256") != registry.get("registrySha256"):
        raise ReleaseLifecycleError("MODEL_RELEASE_REGISTRY_CHANGED", "The catalog changed after stable approval; reissue the release.")
    try:
        return activate_artifact(
            stable["artifactId"],
            registry_path=registry_path,
            trust_store_path=trust_store_path,
            private_key_path=private_key_path,
            key_id=key_id,
            expected_revision=expected_revision,
            runtime_contract=runtime_contract,
            release_id=stable["releaseId"],
        )
    except RegistryManagerError as error:
        raise ReleaseLifecycleError(error.code, error.message) from error


def rollback_stable_release(
    stable_path: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    expected_revision: int | None = None,
    runtime_contract: RuntimeContract | None = None,
) -> dict[str, Any]:
    """Restore the prior stable artifact through the signed registry rollback."""

    current, _ = _verify_release_record(stable_path, trust_store_path=trust_store_path, expected_state="stable", registry_path=registry_path, runtime_contract=runtime_contract, verify_artifact=True)
    registry = verify_registry(registry_path.resolve(), trust_store_path=trust_store_path)
    if expected_revision is not None and registry["revision"] != expected_revision:
        raise ReleaseLifecycleError("MODEL_REGISTRY_REVISION_CONFLICT", "Registry revision changed before rollback.")
    target_id = registry.get("previousActiveArtifactId")
    if not isinstance(target_id, str):
        raise ReleaseLifecycleError("MODEL_RELEASE_ROLLBACK_UNAVAILABLE", "No previous active stable artifact is retained.")
    target_path = stable_path.resolve().parent / f"{target_id}.stable.json"
    target, _ = _verify_release_record(target_path, trust_store_path=trust_store_path, expected_state="stable", registry_path=registry_path, runtime_contract=runtime_contract, verify_artifact=True)
    if current.get("artifactId") != registry.get("activeArtifactId"):
        raise ReleaseLifecycleError("MODEL_RELEASE_ACTIVE_MISMATCH", "The supplied stable release is not the active release.")
    try:
        return rollback_registry(
            registry_path=registry_path,
            trust_store_path=trust_store_path,
            private_key_path=private_key_path,
            key_id=key_id,
            expected_revision=expected_revision,
            runtime_contract=runtime_contract,
            release_id=target["releaseId"],
        )
    except RegistryManagerError as error:
        raise ReleaseLifecycleError(error.code, error.message) from error


def verify_release_record(
    path: Path,
    *,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    runtime_contract: RuntimeContract | None = None,
) -> dict[str, Any]:
    """Verify a signed record and its compatibility; stable records verify artifacts too."""

    record, _ = _verify_release_record(
        path,
        trust_store_path=trust_store_path,
        registry_path=registry_path,
        runtime_contract=runtime_contract,
        verify_artifact=True,
    )
    return record
