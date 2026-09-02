"""Content-free backup inventory and restore/rotation verification.

This module never copies backup data, encrypts files, schedules tasks, reads a
private signing key, or contacts a secret manager. Those deployment operations
must be supplied by an approved external target and are verified through
content-free receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

from model.registry_manager import (
    RegistryManagerError,
    artifact_root_from_entry,
    verify_artifact_directory,
    verify_registry,
)
from model.release_lifecycle import ReleaseLifecycleError, verify_release_record


AI_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = AI_ROOT / "model/registry/active_model.json"
TRUST_STORE_PATH = AI_ROOT / "model/registry/trust/trusted-keys.json"
POLICY_PATH = AI_ROOT / "operations/deployment-continuity-policy.v1.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_FILE_NAMES = frozenset({".env", "id_rsa", "id_ed25519"})
FORBIDDEN_SUFFIXES = frozenset(
    {".key", ".pem", ".p12", ".pfx", ".db", ".sqlite", ".sqlite3"}
)


class ContinuityError(RuntimeError):
    """A continuity inventory or verification boundary is invalid."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("manifestSha256", None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuityError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuityError(f"{label} must be a JSON object")
    return value


def _safe_relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ContinuityError("continuity source path escapes the selected root") from exc
    if path.is_symlink():
        raise ContinuityError("continuity inventories may not follow symbolic links")
    return relative.as_posix()


def _ensure_public_file(path: Path) -> None:
    name = path.name.casefold()
    if (
        name in FORBIDDEN_FILE_NAMES
        or name.startswith(".env.")
        or name.endswith((".sqlite-wal", ".sqlite-shm", ".sqlite3-wal", ".sqlite3-shm"))
        or path.suffix.casefold() in FORBIDDEN_SUFFIXES
    ):
        raise ContinuityError("private key, environment, and database files require a separate owner-approved backup")


def _contains_forbidden_key(value: Any, forbidden: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in forbidden
            or _contains_forbidden_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


def _files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    return (item for item in sorted(directory.rglob("*")) if item.is_file())


def _entry(root: Path, path: Path, category: str) -> dict[str, Any]:
    _ensure_public_file(path)
    return {
        "path": _safe_relative(root, path),
        "category": category,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def build_registry_source_inventory(
    *,
    source_root: Path = AI_ROOT,
    generated_on: str,
) -> dict[str, Any]:
    """Inventory verified registry recovery sources without creating a backup."""

    root = source_root.resolve()
    registry_path = root / "model/registry/active_model.json"
    trust_path = root / "model/registry/trust/trusted-keys.json"
    try:
        registry = verify_registry(registry_path, trust_store_path=trust_path)
    except RegistryManagerError as exc:
        raise ContinuityError("signed source registry verification failed") from exc
    entries: list[dict[str, Any]] = [
        _entry(root, registry_path, "signed-registry"),
        _entry(root, trust_path, "public-trust"),
        _entry(root, root / "model/release-policy.v1.json", "release-policy"),
        _entry(root, root / "operations/deployment-continuity-policy.v1.json", "continuity-policy"),
    ]
    history_directory = root / "model/registry/history"
    for path in _files(history_directory):
        entries.append(_entry(root, path, "registry-history"))
    release_directory = root / "model/registry/releases"
    for path in _files(release_directory):
        entries.append(_entry(root, path, "signed-release"))
    verified_artifacts = []
    for registry_item in registry.get("artifacts", []):
        if not isinstance(registry_item, Mapping):
            raise ContinuityError("registry artifact entry is invalid")
        artifact_root = artifact_root_from_entry(registry_path, registry_item)
        try:
            artifact = verify_artifact_directory(
                artifact_root, trust_store_path=trust_path
            )
        except RegistryManagerError as exc:
            raise ContinuityError("registered artifact verification failed") from exc
        verified_artifacts.append(
            {
                "artifactId": artifact.artifact_id,
                "manifestSha256": artifact.manifest_sha256,
                "releaseStatus": artifact.release_status,
                "activationEligible": artifact.activation_eligible,
            }
        )
        for path in _files(artifact_root):
            entries.append(_entry(root, path, "immutable-artifact"))
    paths = [item["path"] for item in entries]
    if len(paths) != len(set(paths)):
        raise ContinuityError("continuity inventory contains duplicate paths")
    entries.sort(key=lambda item: item["path"])
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "manifestKind": "vfai-fu-010-continuity-source-inventory-v1",
        "generatedOn": generated_on,
        "registry": {
            "revision": registry.get("revision"),
            "registrySha256": registry.get("registrySha256"),
            "activeArtifactId": registry.get("activeArtifactId"),
        },
        "verifiedArtifacts": verified_artifacts,
        "entries": entries,
        "entryCount": len(entries),
        "totalBytes": sum(int(item["bytes"]) for item in entries),
        "registryHistoryEntryCount": sum(
            item["category"] == "registry-history" for item in entries
        ),
        "signedReleaseEntryCount": sum(
            item["category"] == "signed-release" for item in entries
        ),
        "backupCreated": False,
        "encryptionApplied": False,
        "restoreTestExecuted": False,
        "runtimeDatabasesIncluded": False,
        "environmentFileIncluded": False,
        "privateSigningKeyIncluded": False,
        "sourceRootStored": False,
        "rawSecretStored": False,
        "manifestSha256": "",
    }
    manifest["manifestSha256"] = _manifest_digest(manifest)
    return manifest


def validate_inventory(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("manifestKind")
        != "vfai-fu-010-continuity-source-inventory-v1"
        or manifest.get("manifestSha256") != _manifest_digest(manifest)
    ):
        raise ContinuityError("continuity inventory identity or checksum is invalid")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContinuityError("continuity inventory entries are missing")
    paths: set[str] = set()
    for item in entries:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "category", "bytes", "sha256"}
            or not isinstance(item.get("path"), str)
            or Path(item["path"]).is_absolute()
            or ".." in Path(item["path"]).parts
            or item["path"] in paths
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] < 0
            or not isinstance(item.get("sha256"), str)
            or SHA256.fullmatch(item["sha256"]) is None
        ):
            raise ContinuityError("continuity inventory entry is invalid")
        paths.add(item["path"])
        _ensure_public_file(Path(item["path"]))
    if manifest.get("entryCount") != len(entries) or manifest.get("totalBytes") != sum(
        item["bytes"] for item in entries
    ):
        raise ContinuityError("continuity inventory accounting is invalid")
    if any(
        manifest.get(field) is not False
        for field in (
            "backupCreated",
            "encryptionApplied",
            "restoreTestExecuted",
            "runtimeDatabasesIncluded",
            "environmentFileIncluded",
            "privateSigningKeyIncluded",
            "sourceRootStored",
            "rawSecretStored",
        )
    ):
        raise ContinuityError("source inventory overclaims backup or contains forbidden data")


def verify_restored_tree(
    manifest: Mapping[str, Any],
    restored_root: Path,
    *,
    verify_signed_registry: bool = True,
) -> dict[str, Any]:
    """Verify a caller-created disposable restore without modifying it."""

    validate_inventory(manifest)
    root = restored_root.resolve()
    try:
        root.relative_to(AI_ROOT.resolve())
        inside_source_worktree = True
    except ValueError:
        inside_source_worktree = False
    if not root.is_dir() or inside_source_worktree:
        raise ContinuityError("restore verification requires a separate existing directory")
    started = time.perf_counter()
    verified_bytes = 0
    for item in manifest["entries"]:
        path = root / item["path"]
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ContinuityError("a restored inventory path escapes the restore root") from exc
        relative_parts = Path(item["path"]).parts
        if (
            not path.is_file()
            or any((root.joinpath(*relative_parts[:index])).is_symlink() for index in range(1, len(relative_parts) + 1))
        ):
            raise ContinuityError("a restored inventory file is missing or unsafe")
        if path.stat().st_size != item["bytes"] or _sha256_file(path) != item["sha256"]:
            raise ContinuityError("a restored inventory file failed checksum verification")
        verified_bytes += int(item["bytes"])
    registry_verified = False
    artifact_count = 0
    release_count = 0
    if verify_signed_registry:
        registry_path = root / "model/registry/active_model.json"
        trust_path = root / "model/registry/trust/trusted-keys.json"
        try:
            registry = verify_registry(registry_path, trust_store_path=trust_path)
            for item in registry.get("artifacts", []):
                artifact_root = artifact_root_from_entry(registry_path, item)
                verify_artifact_directory(artifact_root, trust_store_path=trust_path)
                artifact_count += 1
        except RegistryManagerError as exc:
            raise ContinuityError("restored signed registry or artifact verification failed") from exc
        registry_verified = True
        release_root = root / "model/registry/releases"
        try:
            for release_path in _files(release_root):
                if release_path.suffix.casefold() != ".json":
                    raise ContinuityError("restored release directory contains an unsupported file")
                verify_release_record(
                    release_path,
                    trust_store_path=trust_path,
                    registry_path=registry_path,
                )
                release_count += 1
        except ReleaseLifecycleError as exc:
            raise ContinuityError("restored signed release verification failed") from exc
    return {
        "manifestSha256": manifest.get("manifestSha256"),
        "verifiedFileCount": len(manifest["entries"]),
        "verifiedBytes": verified_bytes,
        "signedRegistryVerified": registry_verified,
        "verifiedArtifactCount": artifact_count,
        "verifiedSignedReleaseCount": release_count,
        "elapsedMilliseconds": round((time.perf_counter() - started) * 1000, 3),
        "sourceTreeMutated": False,
        "restoredTreeMutated": False,
        "rawSecretStored": False,
    }


def verify_backup_receipt(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate content-free evidence from an external encrypted backup job."""

    validate_inventory(manifest)
    declared = receipt.get("receiptSha256")
    unsigned = dict(receipt)
    unsigned.pop("receiptSha256", None)
    if (
        receipt.get("schemaVersion") != 1
        or receipt.get("receiptKind") != "vfai-fu-010-encrypted-backup-v1"
        or declared != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise ContinuityError("backup receipt identity or checksum is invalid")
    if _contains_forbidden_key(
        receipt,
        frozenset(
            {
                "secret",
                "token",
                "privatekey",
                "encryptionkey",
                "credentialvalue",
                "keymaterial",
            }
        ),
    ):
        raise ContinuityError("backup receipt contains a raw secret field")
    backup_id = receipt.get("backupId")
    provider_id = receipt.get("providerId")
    destination_id = receipt.get("destinationId")
    scheduler_sha256 = receipt.get("schedulerDefinitionSha256")
    if (
        not isinstance(backup_id, str)
        or not backup_id.strip()
        or not isinstance(provider_id, str)
        or not provider_id.strip()
        or not isinstance(destination_id, str)
        or not destination_id.strip()
        or not isinstance(scheduler_sha256, str)
        or SHA256.fullmatch(scheduler_sha256) is None
        or not isinstance(receipt.get("createdOn"), str)
        or not receipt["createdOn"].strip()
        or receipt.get("sourceManifestSha256") != manifest.get("manifestSha256")
        or receipt.get("entryCount") != manifest.get("entryCount")
        or receipt.get("totalBytes") != manifest.get("totalBytes")
        or receipt.get("completed") is not True
        or receipt.get("encrypted") is not True
        or receipt.get("destinationOutsideSourceWorktree") is not True
        or receipt.get("rawSecretStored") is not False
    ):
        raise ContinuityError("backup receipt does not prove encrypted external completion")
    return {
        "backupId": backup_id,
        "providerId": provider_id,
        "destinationId": destination_id,
        "schedulerDefinitionSha256": scheduler_sha256,
        "receiptSha256": declared,
        "sourceManifestSha256": manifest.get("manifestSha256"),
        "encrypted": True,
        "destinationOutsideSourceWorktree": True,
        "rawSecretStored": False,
    }


def verify_owner_objectives_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate checksum-bound, content-free Operations and data-owner approval."""

    declared = receipt.get("receiptSha256")
    unsigned = dict(receipt)
    unsigned.pop("receiptSha256", None)
    if (
        receipt.get("schemaVersion") != 1
        or receipt.get("receiptKind") != "vfai-fu-010-owner-objectives-v1"
        or declared != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise ContinuityError("owner-objectives receipt identity or checksum is invalid")
    if _contains_forbidden_key(
        receipt,
        frozenset(
            {
                "secret",
                "token",
                "privatekey",
                "credentialvalue",
                "keymaterial",
            }
        ),
    ):
        raise ContinuityError("owner-objectives receipt contains a raw secret field")
    objectives = {
        key: receipt.get(key)
        for key in (
            "recoveryPointObjectiveSeconds",
            "recoveryTimeObjectiveSeconds",
            "availabilityObjectivePercent",
            "retentionDays",
        )
    }
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
        for value in objectives.values()
    ) or float(objectives["availabilityObjectivePercent"]) > 100:
        raise ContinuityError("owner objectives must be finite positive values")
    data_approval = receipt.get("dataOwnerApprovalId")
    operations_approval = receipt.get("operationsOwnerApprovalId")
    if (
        not isinstance(data_approval, str)
        or not data_approval.strip()
        or not isinstance(operations_approval, str)
        or not operations_approval.strip()
        or data_approval == operations_approval
        or receipt.get("dataOwnerApproved") is not True
        or receipt.get("operationsOwnerApproved") is not True
        or not isinstance(receipt.get("approvedOn"), str)
        or not receipt["approvedOn"].strip()
        or receipt.get("rawSecretStored") is not False
    ):
        raise ContinuityError("owner-objectives receipt lacks independent approvals")
    return {
        **objectives,
        "dataOwnerApprovalId": data_approval,
        "operationsOwnerApprovalId": operations_approval,
        "dataOwnerApproved": True,
        "operationsOwnerApproved": True,
        "receiptSha256": declared,
        "rawSecretStored": False,
    }


def build_restore_receipt(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    restore_target_id: str,
    restore_scheduler_sha256: str,
    verified_on: str,
) -> dict[str, Any]:
    """Build a content-free receipt for a completed clean restore verification."""

    validate_inventory(manifest)
    if not restore_target_id.strip() or SHA256.fullmatch(restore_scheduler_sha256) is None:
        raise ContinuityError("restore target id and scheduler checksum are required")
    receipt: dict[str, Any] = {
        "schemaVersion": 1,
        "receiptKind": "vfai-fu-010-clean-restore-v1",
        "verifiedOn": verified_on,
        "restoreTargetId": restore_target_id,
        "restoreSchedulerDefinitionSha256": restore_scheduler_sha256,
        "sourceManifestSha256": manifest.get("manifestSha256"),
        "verifiedFileCount": result.get("verifiedFileCount"),
        "verifiedBytes": result.get("verifiedBytes"),
        "signedRegistryVerified": result.get("signedRegistryVerified"),
        "verifiedArtifactCount": result.get("verifiedArtifactCount"),
        "verifiedSignedReleaseCount": result.get("verifiedSignedReleaseCount"),
        "elapsedMilliseconds": result.get("elapsedMilliseconds"),
        "cleanDisposableTarget": True,
        "sourceTreeMutated": result.get("sourceTreeMutated"),
        "restoredTreeMutated": result.get("restoredTreeMutated"),
        "rawSecretStored": False,
        "receiptSha256": "",
    }
    receipt["receiptSha256"] = hashlib.sha256(_canonical({
        key: value for key, value in receipt.items() if key != "receiptSha256"
    })).hexdigest()
    verify_restore_receipt(receipt, manifest)
    return receipt


def verify_restore_receipt(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a clean restore receipt against its source inventory."""

    validate_inventory(manifest)
    declared = receipt.get("receiptSha256")
    unsigned = dict(receipt)
    unsigned.pop("receiptSha256", None)
    elapsed = receipt.get("elapsedMilliseconds")
    target_id = receipt.get("restoreTargetId")
    scheduler_sha256 = receipt.get("restoreSchedulerDefinitionSha256")
    if _contains_forbidden_key(
        receipt,
        frozenset(
            {
                "secret",
                "token",
                "privatekey",
                "encryptionkey",
                "credentialvalue",
                "keymaterial",
            }
        ),
    ):
        raise ContinuityError("restore receipt contains a raw secret field")
    if (
        receipt.get("schemaVersion") != 1
        or receipt.get("receiptKind") != "vfai-fu-010-clean-restore-v1"
        or declared != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or not isinstance(target_id, str)
        or not target_id.strip()
        or not isinstance(scheduler_sha256, str)
        or SHA256.fullmatch(scheduler_sha256) is None
        or not isinstance(receipt.get("verifiedOn"), str)
        or not receipt["verifiedOn"].strip()
        or receipt.get("sourceManifestSha256") != manifest.get("manifestSha256")
        or receipt.get("verifiedFileCount") != manifest.get("entryCount")
        or receipt.get("verifiedBytes") != manifest.get("totalBytes")
        or receipt.get("signedRegistryVerified") is not True
        or receipt.get("verifiedArtifactCount") != len(manifest.get("verifiedArtifacts", []))
        or receipt.get("verifiedSignedReleaseCount")
        != manifest.get("signedReleaseEntryCount")
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or elapsed < 0
        or receipt.get("cleanDisposableTarget") is not True
        or receipt.get("sourceTreeMutated") is not False
        or receipt.get("restoredTreeMutated") is not False
        or receipt.get("rawSecretStored") is not False
    ):
        raise ContinuityError("restore receipt does not prove a clean verified restore")
    return {
        "restoreTargetId": target_id,
        "restoreSchedulerDefinitionSha256": scheduler_sha256,
        "receiptSha256": declared,
        "sourceManifestSha256": manifest.get("manifestSha256"),
        "elapsedMilliseconds": elapsed,
        "cleanDisposableTarget": True,
        "rawSecretStored": False,
    }


def verify_rotation_receipt(receipt: Mapping[str, Any], *, secret_kind: str) -> dict[str, Any]:
    """Validate content-free evidence produced by an external secret manager."""

    if secret_kind not in {"service-token", "signing-trust"}:
        raise ContinuityError("rotation secret kind is unsupported")
    declared = receipt.get("receiptSha256")
    unsigned = dict(receipt)
    unsigned.pop("receiptSha256", None)
    if (
        receipt.get("schemaVersion") != 1
        or receipt.get("receiptKind") != "vfai-fu-010-secret-rotation-v1"
        or receipt.get("secretKind") != secret_kind
        or declared != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise ContinuityError("rotation receipt identity or checksum is invalid")
    forbidden = frozenset(
        {
            "secret",
            "token",
            "privatekey",
            "oldsecret",
            "newsecret",
            "credentialvalue",
            "keymaterial",
        }
    )
    if _contains_forbidden_key(receipt, forbidden):
        raise ContinuityError("rotation receipt contains a raw secret field")
    old_fingerprint = receipt.get("oldFingerprintSha256")
    new_fingerprint = receipt.get("newFingerprintSha256")
    provider_id = receipt.get("secretManagerProvider")
    definition_sha256 = receipt.get("rotationDefinitionSha256")
    if (
        not isinstance(provider_id, str)
        or not provider_id.strip()
        or not isinstance(definition_sha256, str)
        or SHA256.fullmatch(definition_sha256) is None
        or not isinstance(receipt.get("rotatedOn"), str)
        or not receipt["rotatedOn"].strip()
        or not isinstance(old_fingerprint, str)
        or SHA256.fullmatch(old_fingerprint) is None
        or not isinstance(new_fingerprint, str)
        or SHA256.fullmatch(new_fingerprint) is None
        or old_fingerprint == new_fingerprint
        or receipt.get("oldCredentialRejected") is not True
        or receipt.get("newCredentialAccepted") is not True
        or receipt.get("rollbackTested") is not True
        or receipt.get("trustedReleaseHistoryStillValid") is not True
        or receipt.get("rawSecretStored") is not False
    ):
        raise ContinuityError("rotation receipt does not prove acceptance, rejection, rollback, and history gates")
    return {
        "secretKind": secret_kind,
        "secretManagerProvider": provider_id,
        "rotationDefinitionSha256": definition_sha256,
        "receiptSha256": declared,
        "oldCredentialRejected": True,
        "newCredentialAccepted": True,
        "rollbackTested": True,
        "trustedReleaseHistoryStillValid": True,
        "rawSecretStored": False,
    }
