"""Fail-closed provenance, license, privacy, and retention enforcement.

The source registry is the authority for whether data may be used. A shard is
eligible only when its immutable manifest, content checksum, source snapshots,
generator checksum, and requested use all pass validation. Retained datasets
from before this contract are represented by quarantine manifests and can
never become eligible merely because a file exists.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = GOVERNANCE_ROOT / "policy.v1.json"
DEFAULT_SOURCE_REGISTRY_PATH = GOVERNANCE_ROOT / "source-registry.v1.json"
DEFAULT_CATALOG_PATH = GOVERNANCE_ROOT / "retained-dataset-catalog.v1.json"
DEFAULT_MANIFEST_DIR = GOVERNANCE_ROOT / "manifests"
DEFAULT_REPORT_PATH = GOVERNANCE_ROOT / "reports" / "current-corpus-audit.json"
DEFAULT_MODEL_REGISTRY_PATH = AI_ROOT / "model" / "registry" / "active_model.json"
DEFAULT_ARTIFACT_INVENTORY_PATH = AI_ROOT / "model" / "artifact_inventory.json"
SUPPORTED_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DataGovernanceError(RuntimeError):
    """A precise, public-safe data governance validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DataGovernanceError(code, f"Required governance file is missing: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataGovernanceError(code, f"Governance JSON is invalid: {path}") from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _relative_to_ai(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(AI_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_path_within(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise DataGovernanceError(
            "DATA_SOURCE_PATH_INVALID", "Registered repository paths must be relative."
        )
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise DataGovernanceError(
            "DATA_SOURCE_PATH_INVALID", "Registered source path escapes Voltforge_AI."
        ) from error
    return resolved


def _resolve_ai_path(value: str) -> Path:
    return _resolve_path_within(AI_ROOT, value)


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    policy_path = Path(path or DEFAULT_POLICY_PATH)
    policy = _read_json(policy_path, "DATA_POLICY_INVALID")
    if not isinstance(policy, dict) or policy.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise DataGovernanceError(
            "DATA_POLICY_INVALID",
            f"Data policy schemaVersion must be {SUPPORTED_SCHEMA_VERSION}.",
        )
    if policy.get("defaultDecision") != "deny":
        raise DataGovernanceError(
            "DATA_POLICY_NOT_FAIL_CLOSED", "Data policy defaultDecision must be deny."
        )
    return policy


def load_source_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path or DEFAULT_SOURCE_REGISTRY_PATH)
    registry = _read_json(registry_path, "DATA_SOURCE_REGISTRY_INVALID")
    if not isinstance(registry, dict) or registry.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise DataGovernanceError(
            "DATA_SOURCE_REGISTRY_INVALID",
            f"Source registry schemaVersion must be {SUPPORTED_SCHEMA_VERSION}.",
        )
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise DataGovernanceError(
            "DATA_SOURCE_REGISTRY_INVALID", "Source registry sources must be an array."
        )
    seen: set[str] = set()
    required = {
        "sourceId",
        "name",
        "kind",
        "origin",
        "revision",
        "license",
        "privacy",
        "allowedUses",
        "prohibitedUses",
        "approval",
        "preprocessing",
        "checksum",
        "retention",
        "deletionProcedure",
    }
    for source in sources:
        if not isinstance(source, dict) or not required.issubset(source):
            raise DataGovernanceError(
                "DATA_SOURCE_REGISTRY_INVALID", "Every source must contain the complete governance contract."
            )
        source_id = source.get("sourceId")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise DataGovernanceError(
                "DATA_SOURCE_REGISTRY_INVALID", "Source IDs must be non-empty and unique."
            )
        seen.add(source_id)
        checksum = source.get("checksum")
        if checksum is not None and not (
            isinstance(checksum, str) and SHA256_RE.fullmatch(checksum)
        ):
            raise DataGovernanceError(
                "DATA_SOURCE_REGISTRY_INVALID", f"Source {source_id} has an invalid SHA-256 checksum."
            )
    return registry


def _source_by_id(registry: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    for source in registry.get("sources", []):
        if isinstance(source, dict) and source.get("sourceId") == source_id:
            return source
    raise DataGovernanceError(
        "DATA_SOURCE_NOT_REGISTERED", f"Data source is not registered: {source_id}"
    )


def source_approval_decision(
    source: Mapping[str, Any],
    usage: str,
    *,
    policy: Mapping[str, Any] | None = None,
    verify_checksum: bool = True,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic allow/deny decision for one source and use."""

    active_policy = dict(policy or load_policy())
    source_id = str(source.get("sourceId") or "unidentified-source")
    reasons: list[str] = []
    denied_kinds = set(active_policy.get("trainingDeniedByDefaultKinds", []))
    if usage == "training" and source.get("kind") in denied_kinds:
        reasons.append("source-kind-denied-for-training")

    license_record = source.get("license")
    if not isinstance(license_record, Mapping) or license_record.get("status") != "approved":
        reasons.append("license-not-approved")

    privacy = source.get("privacy")
    if not isinstance(privacy, Mapping):
        reasons.append("privacy-classification-missing")
    else:
        if privacy.get("containsPrivateUserData") is not False:
            reasons.append("private-user-data-not-allowed")
        if usage == "training" and privacy.get("trainingAllowed") is not True:
            reasons.append("privacy-not-approved-for-training")

    allowed_uses = source.get("allowedUses")
    if not isinstance(allowed_uses, list) or usage not in allowed_uses:
        reasons.append("requested-use-not-allowed")
    if usage in set(source.get("prohibitedUses") or []):
        reasons.append("requested-use-explicitly-prohibited")

    approval = source.get("approval")
    if not isinstance(approval, Mapping) or approval.get("status") != "approved":
        reasons.append("source-not-approved")
    elif usage not in set(approval.get("approvedUses") or []):
        reasons.append("use-not-approved")

    origin = source.get("origin")
    checksum = source.get("checksum")
    if verify_checksum and isinstance(origin, Mapping) and origin.get("path"):
        try:
            source_path = _resolve_path_within(
                Path(source_root or AI_ROOT), str(origin["path"])
            )
            if not source_path.is_file():
                reasons.append("source-file-missing")
            elif not isinstance(checksum, str) or sha256_file(source_path) != checksum:
                reasons.append("source-checksum-mismatch")
        except DataGovernanceError:
            reasons.append("source-path-invalid")

    return {
        "sourceId": source_id,
        "usage": usage,
        "allowed": not reasons,
        "decision": "allow" if not reasons else "deny",
        "reasons": reasons,
    }


def require_approved_source(
    source_id: str,
    usage: str,
    *,
    registry_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    registry = load_source_registry(registry_path)
    source = _source_by_id(registry, source_id)
    decision = source_approval_decision(
        source,
        usage,
        policy=load_policy(policy_path),
        verify_checksum=True,
        source_root=source_root,
    )
    if not decision["allowed"]:
        reason = ", ".join(decision["reasons"])
        raise DataGovernanceError(
            "DATA_SOURCE_USE_DENIED",
            f"Source {source_id!r} is denied for {usage}: {reason}.",
        )
    return source


def default_manifest_path(shard_path: str | Path) -> Path:
    shard = Path(shard_path).resolve()
    try:
        relative = shard.relative_to(AI_ROOT)
    except ValueError:
        return shard.with_suffix(shard.suffix + ".manifest.json")
    encoded = "__".join(relative.parts)
    return DEFAULT_MANIFEST_DIR / f"{encoded}.manifest.json"


def count_records(path: str | Path, record_format: str | None = None) -> int:
    corpus = Path(path)
    if record_format == "qa-text-v0" or corpus.name == "dataset.txt":
        content = corpus.read_text(encoding="utf-8")
        return sum(1 for chunk in content.split("[Q]") if "[A]" in chunk)
    with corpus.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _file_descriptor(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"path": None, "sha256": None}
    path = _resolve_ai_path(path_value)
    return {
        "path": path_value,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _manifest_integrity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifestSha256"}


def _source_snapshot(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sourceId": source["sourceId"],
        "revision": source["revision"],
        "checksum": source.get("checksum"),
        "licenseStatus": source["license"]["status"],
        "privacyClass": source["privacy"]["classification"],
        "approvalStatus": source["approval"]["status"],
    }


def _build_shard_manifest(
    shard_path: Path,
    *,
    source_ids: Sequence[str],
    producer: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    task_schema: Mapping[str, Any],
    record_format: str,
    eligible_uses: Sequence[str],
    status: str,
    reason: str | None,
    registry_path: str | Path | None,
) -> dict[str, Any]:
    if not shard_path.is_file():
        raise DataGovernanceError(
            "DATA_SHARD_NOT_FOUND", f"Dataset shard is missing: {shard_path}"
        )
    registry_file = Path(registry_path or DEFAULT_SOURCE_REGISTRY_PATH)
    registry = load_source_registry(registry_file)
    sources = [_source_by_id(registry, source_id) for source_id in source_ids]
    if not sources:
        raise DataGovernanceError(
            "DATA_SHARD_SOURCES_MISSING", "A shard must declare at least one registered source."
        )
    if status == "approved":
        for usage in eligible_uses:
            for source in sources:
                decision = source_approval_decision(source, usage)
                if not decision["allowed"]:
                    raise DataGovernanceError(
                        "DATA_SHARD_SOURCE_DENIED",
                        f"Cannot approve shard because {source['sourceId']} is denied for {usage}: "
                        + ", ".join(decision["reasons"]),
                    )

    shard_hash = sha256_file(shard_path)
    producer_path = producer.get("path")
    preprocessing_path = preprocessing.get("path")
    manifest: dict[str, Any] = {
        "schemaVersion": SUPPORTED_SCHEMA_VERSION,
        "shardId": f"vf-shard-{shard_hash[:20]}",
        "path": _relative_to_ai(shard_path),
        "recordFormat": record_format,
        "recordCount": count_records(shard_path, record_format),
        "sizeBytes": shard_path.stat().st_size,
        "sha256": shard_hash,
        "status": status,
        "eligibleUses": list(eligible_uses) if status == "approved" else [],
        "reason": reason,
        "createdAtUtc": utc_now(),
        "sourceRegistry": {
            "registryId": registry.get("registryId"),
            "version": registry.get("version"),
            "sha256": sha256_file(registry_file),
        },
        "sources": [_source_snapshot(source) for source in sources],
        "producer": {
            "id": producer.get("id"),
            "version": producer.get("version"),
            **_file_descriptor(str(producer_path) if producer_path else None),
        },
        "preprocessing": {
            "id": preprocessing.get("id"),
            "version": preprocessing.get("version"),
            **_file_descriptor(str(preprocessing_path) if preprocessing_path else None),
        },
        "taskSchema": {
            "id": task_schema.get("id"),
            "version": task_schema.get("version"),
            **_file_descriptor(
                str(task_schema.get("path")) if task_schema.get("path") else None
            ),
        },
        "recordProvenance": {
            "mode": "shard-inherited",
            "selector": "all-records",
            "sourceIds": list(source_ids),
            "statement": "Every non-empty record in this shard inherits this immutable source and producer snapshot.",
        },
        "retention": {
            "sourcePolicies": [
                {
                    "sourceId": source["sourceId"],
                    "policy": source["retention"],
                    "deletionProcedure": source["deletionProcedure"],
                }
                for source in sources
            ]
        },
    }
    manifest["manifestSha256"] = _canonical_hash(_manifest_integrity_payload(manifest))
    return manifest


def write_approved_shard_manifest(
    shard_path: str | Path,
    *,
    source_ids: Sequence[str],
    producer_id: str,
    producer_version: str,
    producer_path: str,
    preprocessing_id: str = "vf-held-out-exclusion",
    preprocessing_version: str = "1.0.0",
    preprocessing_path: str = "evaluation/leakage.py",
    task_schema_id: str = "voltforge-task-record",
    task_schema_version: str = "1.0.0",
    task_schema_path: str = "task_schema/task-record.schema.json",
    record_format: str = "vf-task-record-jsonl-v1",
    eligible_uses: Sequence[str] = ("training",),
    manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> Path:
    """Write an immutable approved sidecar after all source checks pass."""

    shard = Path(shard_path).resolve()
    manifest = _build_shard_manifest(
        shard,
        source_ids=source_ids,
        producer={"id": producer_id, "version": producer_version, "path": producer_path},
        preprocessing={
            "id": preprocessing_id,
            "version": preprocessing_version,
            "path": preprocessing_path,
        },
        task_schema={
            "id": task_schema_id,
            "version": task_schema_version,
            "path": task_schema_path,
        },
        record_format=record_format,
        eligible_uses=eligible_uses,
        status="approved",
        reason=None,
        registry_path=registry_path,
    )
    destination = Path(manifest_path) if manifest_path else default_manifest_path(shard)
    _write_json(destination, manifest)
    return destination


def _write_quarantine_manifest(
    shard_path: Path,
    catalog_entry: Mapping[str, Any],
    *,
    registry_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest = _build_shard_manifest(
        shard_path,
        source_ids=list(catalog_entry["sourceIds"]),
        producer=catalog_entry["producer"],
        preprocessing=catalog_entry["preprocessing"],
        task_schema={"id": "legacy-untyped", "version": "0", "path": None},
        record_format=str(catalog_entry["recordFormat"]),
        eligible_uses=(),
        status="quarantined",
        reason=str(catalog_entry["quarantineReason"]),
        registry_path=registry_path,
    )
    destination = default_manifest_path(shard_path)
    _write_json(destination, manifest)
    return destination, manifest


def _validate_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    declared = manifest.get("manifestSha256")
    actual = _canonical_hash(_manifest_integrity_payload(manifest))
    if declared != actual:
        raise DataGovernanceError(
            "DATA_SHARD_MANIFEST_CHECKSUM_MISMATCH", "Shard manifest integrity checksum is invalid."
        )


def require_approved_shard(
    shard_path: str | Path,
    usage: str = "training",
    *,
    manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    dependency_root: str | Path | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one shard and return its manifest, or fail closed."""

    shard = Path(shard_path).resolve()
    manifest_file = Path(manifest_path) if manifest_path else default_manifest_path(shard)
    manifest = _read_json(manifest_file, "DATA_SHARD_MANIFEST_MISSING")
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise DataGovernanceError(
            "DATA_SHARD_MANIFEST_INVALID", "Shard manifest has an unsupported schema."
        )
    _validate_manifest_integrity(manifest)
    if manifest.get("status") != "approved" or usage not in set(manifest.get("eligibleUses") or []):
        raise DataGovernanceError(
            "DATA_SHARD_USE_DENIED",
            f"Shard {_relative_to_ai(shard)!r} is not approved for {usage}: {manifest.get('reason') or 'use not eligible'}.",
        )
    if usage == "training" and manifest.get("recordFormat") != "vf-task-record-jsonl-v1":
        raise DataGovernanceError(
            "DATA_SHARD_TASK_SCHEMA_UNSUPPORTED",
            "Approved training shards must use vf-task-record-jsonl-v1.",
        )
    if not shard.is_file() or sha256_file(shard) != manifest.get("sha256"):
        raise DataGovernanceError(
            "DATA_SHARD_CHECKSUM_MISMATCH", "Dataset shard content does not match its manifest."
        )
    if count_records(shard, manifest.get("recordFormat")) != manifest.get("recordCount"):
        raise DataGovernanceError(
            "DATA_SHARD_RECORD_COUNT_MISMATCH", "Dataset shard record count does not match its manifest."
        )

    registry_file = Path(registry_path or DEFAULT_SOURCE_REGISTRY_PATH)
    registry = load_source_registry(registry_file)
    if sha256_file(registry_file) != manifest.get("sourceRegistry", {}).get("sha256"):
        raise DataGovernanceError(
            "DATA_SHARD_SOURCE_REGISTRY_MISMATCH",
            "Shard source-registry snapshot checksum changed.",
        )
    for snapshot in manifest.get("sources") or []:
        source = _source_by_id(registry, snapshot.get("sourceId"))
        if snapshot.get("revision") != source.get("revision") or snapshot.get("checksum") != source.get("checksum"):
            raise DataGovernanceError(
                "DATA_SHARD_SOURCE_REVISION_MISMATCH",
                f"Shard source snapshot is stale: {source['sourceId']}",
            )
        require_approved_source(
            source["sourceId"],
            usage,
            registry_path=registry_file,
            policy_path=policy_path,
            source_root=source_root,
        )

    for section_name in ("producer", "preprocessing", "taskSchema"):
        descriptor = manifest.get(section_name) or {}
        if descriptor.get("path"):
            path = _resolve_path_within(
                Path(dependency_root or AI_ROOT), descriptor["path"]
            )
            if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
                raise DataGovernanceError(
                    "DATA_SHARD_PIPELINE_REVISION_MISMATCH",
                    f"Shard {section_name} checksum no longer matches its manifest.",
                )
    return manifest


def _catalog_paths(catalog: Mapping[str, Any]) -> set[str]:
    return {str(entry["path"]) for entry in catalog.get("datasets", [])}


def _discover_retained_datasets() -> set[str]:
    paths = {"dataset.txt"} if (AI_ROOT / "dataset.txt").is_file() else set()
    paths.update(
        path.relative_to(AI_ROOT).as_posix()
        for path in sorted((AI_ROOT / "model" / "artifacts").glob("*.jsonl"))
    )
    return paths


def audit_current_corpora(
    *,
    write: bool = True,
    catalog_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit every retained corpus and optionally refresh quarantine evidence."""

    catalog_file = Path(catalog_path or DEFAULT_CATALOG_PATH)
    catalog = _read_json(catalog_file, "DATASET_CATALOG_INVALID")
    if not isinstance(catalog, dict) or catalog.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise DataGovernanceError("DATASET_CATALOG_INVALID", "Dataset catalog schema is invalid.")
    declared = _catalog_paths(catalog)
    discovered = _discover_retained_datasets()
    if declared != discovered:
        raise DataGovernanceError(
            "DATASET_CATALOG_COVERAGE_MISMATCH",
            f"Retained dataset catalog mismatch; missing={sorted(discovered - declared)}, stale={sorted(declared - discovered)}.",
        )

    registry = load_source_registry()
    entries: list[dict[str, Any]] = []
    total_records = 0
    for catalog_entry in catalog["datasets"]:
        shard = (AI_ROOT / catalog_entry["path"]).resolve()
        if write:
            manifest_path, manifest = _write_quarantine_manifest(shard, catalog_entry)
        else:
            manifest_path = default_manifest_path(shard)
            manifest = _read_json(manifest_path, "DATA_SHARD_MANIFEST_MISSING")
            _validate_manifest_integrity(manifest)
            if sha256_file(shard) != manifest.get("sha256"):
                raise DataGovernanceError(
                    "DATA_SHARD_CHECKSUM_MISMATCH", f"Retained shard changed: {catalog_entry['path']}"
                )
            expected_sources = list(catalog_entry["sourceIds"])
            manifest_sources = [
                snapshot.get("sourceId")
                for snapshot in manifest.get("sources", [])
                if isinstance(snapshot, Mapping)
            ]
            expected_producer = {
                "id": catalog_entry["producer"].get("id"),
                "version": catalog_entry["producer"].get("version"),
                **_file_descriptor(catalog_entry["producer"].get("path")),
            }
            expected_preprocessing = {
                "id": catalog_entry["preprocessing"].get("id"),
                "version": catalog_entry["preprocessing"].get("version"),
                **_file_descriptor(catalog_entry["preprocessing"].get("path")),
            }
            expected_task_schema = {
                "id": "legacy-untyped",
                "version": "0",
                "path": None,
                "sha256": None,
            }
            if (
                manifest.get("path") != catalog_entry["path"]
                or manifest.get("recordFormat") != catalog_entry["recordFormat"]
                or manifest.get("status") != "quarantined"
                or manifest.get("eligibleUses") != []
                or manifest.get("reason") != catalog_entry["quarantineReason"]
                or manifest_sources != expected_sources
                or manifest.get("producer") != expected_producer
                or manifest.get("preprocessing") != expected_preprocessing
                or manifest.get("taskSchema") != expected_task_schema
                or (manifest.get("sourceRegistry") or {}).get("sha256")
                != sha256_file(DEFAULT_SOURCE_REGISTRY_PATH)
            ):
                raise DataGovernanceError(
                    "DATA_SHARD_MANIFEST_CATALOG_MISMATCH",
                    f"Retained shard manifest is stale or differs from policy: {catalog_entry['path']}",
                )
            for snapshot in manifest.get("sources", []):
                current_source = _source_by_id(registry, snapshot["sourceId"])
                if (
                    snapshot.get("revision") != current_source.get("revision")
                    or snapshot.get("checksum") != current_source.get("checksum")
                    or snapshot.get("licenseStatus") != current_source["license"]["status"]
                    or snapshot.get("privacyClass") != current_source["privacy"]["classification"]
                    or snapshot.get("approvalStatus") != current_source["approval"]["status"]
                ):
                    raise DataGovernanceError(
                        "DATA_SHARD_SOURCE_REVISION_MISMATCH",
                        f"Retained shard source snapshot is stale: {catalog_entry['path']}",
                    )
        source_decisions = [
            source_approval_decision(_source_by_id(registry, source_id), "training")
            for source_id in catalog_entry["sourceIds"]
        ]
        records = int(manifest["recordCount"])
        total_records += records
        entries.append(
            {
                "path": catalog_entry["path"],
                "manifestPath": _relative_to_ai(manifest_path),
                "records": records,
                "sizeBytes": manifest["sizeBytes"],
                "sha256": manifest["sha256"],
                "status": manifest["status"],
                "trainingEligible": False,
                "runtimeRetrievalEligible": False,
                "sourceIds": list(catalog_entry["sourceIds"]),
                "sourceDecisions": source_decisions,
                "reason": manifest["reason"],
            }
        )

    policy = load_policy()
    report: dict[str, Any] = {
        "schemaVersion": SUPPORTED_SCHEMA_VERSION,
        "auditId": "vfai-006-current-corpus-audit",
        "generatedAtUtc": utc_now(),
        "policyVersion": policy.get("version"),
        "sourceRegistryVersion": registry.get("version"),
        "catalogSha256": sha256_file(catalog_file),
        "summary": {
            "retainedDatasetCount": len(entries),
            "recordCount": total_records,
            "approvedTrainingShardCount": 0,
            "approvedRuntimeRetrievalCount": 0,
            "quarantinedDatasetCount": len(entries),
            "catalogCoverageComplete": True,
            "decision": "pass",
        },
        "defaultExclusions": {
            "unlicensedScrapedTraining": "deny",
            "privateUserProjectTraining": "deny",
            "unregisteredSourceTraining": "deny",
            "liveWebResultsPersistedToTraining": False,
        },
        "datasets": entries,
    }
    report["reportSha256"] = _canonical_hash(
        {key: value for key, value in report.items() if key != "reportSha256"}
    )
    if write:
        _write_json(Path(report_path or DEFAULT_REPORT_PATH), report)
    else:
        checked_report = _read_json(
            Path(report_path or DEFAULT_REPORT_PATH), "DATA_GOVERNANCE_REPORT_MISSING"
        )
        if not isinstance(checked_report, dict):
            raise DataGovernanceError(
                "DATA_GOVERNANCE_REPORT_STALE", "Checked-in corpus audit must be a JSON object."
            )
        declared_report_hash = checked_report.get("reportSha256")
        actual_report_hash = _canonical_hash(
            {key: value for key, value in checked_report.items() if key != "reportSha256"}
        )
        comparable_keys = (
            "schemaVersion",
            "auditId",
            "policyVersion",
            "sourceRegistryVersion",
            "catalogSha256",
            "summary",
            "defaultExclusions",
            "datasets",
        )
        if declared_report_hash != actual_report_hash or any(
            checked_report.get(key) != report.get(key) for key in comparable_keys
        ):
            raise DataGovernanceError(
                "DATA_GOVERNANCE_REPORT_STALE",
                "Checked-in corpus audit is stale or its integrity checksum is invalid.",
            )
    return report


def _load_optional_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    value = _read_json(path, "DATA_LINEAGE_INPUT_INVALID")
    return value if isinstance(value, Mapping) else {}


def analyze_source_removal(
    source_id: str,
    *,
    manifest_paths: Iterable[str | Path] | None = None,
    model_registry_path: str | Path | None = None,
    artifact_inventory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a read-only deletion and retraining impact report for one source."""

    registry = load_source_registry()
    source = _source_by_id(registry, source_id)
    if manifest_paths is None:
        paths = sorted(
            {
                *DEFAULT_MANIFEST_DIR.glob("*.manifest.json"),
                *(AI_ROOT / "synthetic_data/releases").glob(
                    "*/*/content/data_governance/manifests/*.manifest.json"
                ),
            }
        )
    else:
        paths = [Path(path) for path in manifest_paths]

    affected_shards: list[dict[str, Any]] = []
    affected_shard_ids: set[str] = set()
    for path in paths:
        manifest = _load_optional_json(path)
        source_ids = {
            snapshot.get("sourceId")
            for snapshot in manifest.get("sources", [])
            if isinstance(snapshot, Mapping)
        }
        if source_id in source_ids:
            shard_id = str(manifest.get("shardId"))
            affected_shard_ids.add(shard_id)
            affected_shards.append(
                {
                    "shardId": shard_id,
                    "path": manifest.get("path"),
                    "manifestPath": _relative_to_ai(path),
                    "status": manifest.get("status"),
                    "recordCount": manifest.get("recordCount"),
                    "sha256": manifest.get("sha256"),
                    "retainedImmutableRelease": "synthetic_data/releases/"
                    in _relative_to_ai(path),
                }
            )

    model_registry = _load_optional_json(
        Path(model_registry_path or DEFAULT_MODEL_REGISTRY_PATH)
    )
    affected_artifacts: list[dict[str, Any]] = []
    for artifact in model_registry.get("artifacts", []):
        if not isinstance(artifact, Mapping):
            continue
        lineage = artifact.get("dataLineage") or {}
        artifact_sources = set(lineage.get("sourceIds") or [])
        artifact_shards = set(lineage.get("shardIds") or [])
        if source_id in artifact_sources or affected_shard_ids.intersection(artifact_shards):
            affected_artifacts.append(
                {
                    "artifactId": artifact.get("artifactId"),
                    "releaseStatus": artifact.get("releaseStatus"),
                    "action": "deactivate-delete-and-retrain",
                }
            )

    inventory = _load_optional_json(
        Path(artifact_inventory_path or DEFAULT_ARTIFACT_INVENTORY_PATH)
    )
    unresolved_legacy = [
        {
            "path": artifact.get("path"),
            "kind": artifact.get("kind"),
            "classification": artifact.get("classification"),
            "action": "retain-quarantine-or-delete; lineage cannot exclude impact",
        }
        for artifact in inventory.get("artifacts", [])
        if isinstance(artifact, Mapping) and artifact.get("kind") != "training-dataset"
    ]

    deletion_targets = sorted(
        {
            str(item["path"])
            for item in affected_shards
            if item.get("path")
        }
        | {
            str(item["manifestPath"])
            for item in affected_shards
            if item.get("manifestPath")
        }
    )
    return {
        "schemaVersion": SUPPORTED_SCHEMA_VERSION,
        "sourceId": source_id,
        "sourceStatus": source["approval"]["status"],
        "deletionProcedure": source["deletionProcedure"],
        "affectedShards": affected_shards,
        "affectedArtifacts": affected_artifacts,
        "potentiallyAffectedLegacyArtifacts": unresolved_legacy,
        "requiresRetraining": bool(affected_artifacts),
        "requiresLegacyQuarantineReview": bool(affected_shards and unresolved_legacy),
        "deletionTargets": deletion_targets,
        "nextActions": [
            "Deactivate every affected approved model before deleting source data.",
            "Delete affected shards and their manifests using the listed explicit paths.",
            "Regenerate shards only from remaining approved sources.",
            "Retrain and repeat all release gates before activation.",
        ],
    }
