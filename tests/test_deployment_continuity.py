from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

import operations.continuity as continuity
import tools.evaluate_deployment_continuity_readiness as continuity_readiness
from operations.continuity import (
    ContinuityError,
    build_restore_receipt,
    validate_inventory,
    verify_backup_receipt,
    verify_owner_objectives_receipt,
    verify_restored_tree,
    verify_restore_receipt,
    verify_rotation_receipt,
)
from tools.evaluate_deployment_continuity_readiness import (
    DeploymentContinuityReadinessError,
    POLICY_PATH,
    build_report,
    validate_report,
)


def _digest(value: dict[str, object], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    payload = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(path: str = "payload/data.json", content: bytes = b"safe") -> dict[str, object]:
    value: dict[str, object] = {
        "schemaVersion": 1,
        "manifestKind": "vfai-fu-010-continuity-source-inventory-v1",
        "generatedOn": "2026-08-31",
        "registry": {"revision": 1, "registrySha256": "a" * 64, "activeArtifactId": None},
        "verifiedArtifacts": [],
        "entries": [{
            "path": path,
            "category": "test",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }],
        "entryCount": 1,
        "totalBytes": len(content),
        "registryHistoryEntryCount": 0,
        "signedReleaseEntryCount": 0,
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
    value["manifestSha256"] = _digest(value, "manifestSha256")
    return value


def _backup_receipt(manifest: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "schemaVersion": 1,
        "receiptKind": "vfai-fu-010-encrypted-backup-v1",
        "backupId": "backup-opaque-001",
        "providerId": "encrypted-store-001",
        "destinationId": "external-destination-001",
        "schedulerDefinitionSha256": "3" * 64,
        "createdOn": "2026-09-01",
        "sourceManifestSha256": manifest["manifestSha256"],
        "entryCount": manifest["entryCount"],
        "totalBytes": manifest["totalBytes"],
        "completed": True,
        "encrypted": True,
        "destinationOutsideSourceWorktree": True,
        "rawSecretStored": False,
        "receiptSha256": "",
    }
    value["receiptSha256"] = _digest(value, "receiptSha256")
    return value


def _rotation_receipt(secret_kind: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schemaVersion": 1,
        "receiptKind": "vfai-fu-010-secret-rotation-v1",
        "secretKind": secret_kind,
        "secretManagerProvider": "secret-manager-001",
        "rotationDefinitionSha256": "3" * 64,
        "rotatedOn": "2026-09-01",
        "oldFingerprintSha256": "1" * 64,
        "newFingerprintSha256": "2" * 64,
        "oldCredentialRejected": True,
        "newCredentialAccepted": True,
        "rollbackTested": True,
        "trustedReleaseHistoryStillValid": True,
        "rawSecretStored": False,
        "receiptSha256": "",
    }
    value["receiptSha256"] = _digest(value, "receiptSha256")
    return value


def _objectives_receipt() -> dict[str, object]:
    value: dict[str, object] = {
        "schemaVersion": 1,
        "receiptKind": "vfai-fu-010-owner-objectives-v1",
        "recoveryPointObjectiveSeconds": 900,
        "recoveryTimeObjectiveSeconds": 1800,
        "availabilityObjectivePercent": 99.5,
        "retentionDays": 30,
        "dataOwnerApprovalId": "data-owner-approval-001",
        "operationsOwnerApprovalId": "operations-owner-approval-001",
        "dataOwnerApproved": True,
        "operationsOwnerApproved": True,
        "approvedOn": "2026-09-01",
        "rawSecretStored": False,
        "receiptSha256": "",
    }
    value["receiptSha256"] = _digest(value, "receiptSha256")
    return value


def test_inventory_rejects_environment_private_key_and_database_paths() -> None:
    for path in (
        ".env",
        ".env.production",
        "trust/private.pem",
        "runtime/memory.db",
        "runtime/memory.sqlite3-wal",
    ):
        with pytest.raises(ContinuityError, match="private key, environment, and database"):
            validate_inventory(_manifest(path=path))


def test_restore_verification_is_read_only_and_fails_on_tamper(tmp_path) -> None:
    content = b"safe"
    manifest = _manifest(content=content)
    restored = tmp_path / "restore"
    payload = restored / "payload"
    payload.mkdir(parents=True)
    restored_file = payload / "data.json"
    restored_file.write_bytes(content)

    result = verify_restored_tree(manifest, restored, verify_signed_registry=False)

    assert result["verifiedFileCount"] == 1
    assert result["sourceTreeMutated"] is False
    assert result["restoredTreeMutated"] is False
    restored_file.write_bytes(b"tampered")
    with pytest.raises(ContinuityError, match="checksum"):
        verify_restored_tree(manifest, restored, verify_signed_registry=False)


def test_restore_verification_rejects_any_target_inside_live_source(tmp_path, monkeypatch) -> None:
    manifest = _manifest()
    monkeypatch.setattr(continuity, "AI_ROOT", tmp_path)
    restored = tmp_path / "nested-restore" / "payload"
    restored.mkdir(parents=True)
    (restored / "data.json").write_bytes(b"safe")

    with pytest.raises(ContinuityError, match="separate existing directory"):
        verify_restored_tree(manifest, tmp_path / "nested-restore", verify_signed_registry=False)


def test_backup_receipt_requires_encryption_and_no_raw_secret() -> None:
    manifest = _manifest()
    receipt = _backup_receipt(manifest)

    assert verify_backup_receipt(receipt, manifest)["encrypted"] is True
    leaked = deepcopy(receipt)
    leaked["encryptionKey"] = "not-allowed"
    leaked["receiptSha256"] = _digest(leaked, "receiptSha256")
    with pytest.raises(ContinuityError, match="raw secret"):
        verify_backup_receipt(leaked, manifest)
    nested_leak = deepcopy(receipt)
    nested_leak["metadata"] = {"credentialValue": "not-allowed"}
    nested_leak["receiptSha256"] = _digest(nested_leak, "receiptSha256")
    with pytest.raises(ContinuityError, match="raw secret"):
        verify_backup_receipt(nested_leak, manifest)


def test_owner_objectives_require_finite_values_and_independent_approvals() -> None:
    receipt = _objectives_receipt()

    assert verify_owner_objectives_receipt(receipt)["availabilityObjectivePercent"] == 99.5
    invalid = deepcopy(receipt)
    invalid["recoveryTimeObjectiveSeconds"] = float("inf")
    invalid["receiptSha256"] = _digest(invalid, "receiptSha256")
    with pytest.raises(ContinuityError, match="finite positive"):
        verify_owner_objectives_receipt(invalid)


def test_restore_receipt_is_checksum_bound_to_inventory() -> None:
    manifest = _manifest()
    receipt = build_restore_receipt(
        {
            "verifiedFileCount": 1,
            "verifiedBytes": 4,
            "signedRegistryVerified": True,
            "verifiedArtifactCount": 0,
            "verifiedSignedReleaseCount": 0,
            "elapsedMilliseconds": 12.5,
            "sourceTreeMutated": False,
            "restoredTreeMutated": False,
        },
        manifest,
        restore_target_id="disposable-restore-001",
        restore_scheduler_sha256="3" * 64,
        verified_on="2026-08-31",
    )

    assert verify_restore_receipt(receipt, manifest)["cleanDisposableTarget"] is True
    tampered = deepcopy(receipt)
    tampered["verifiedFileCount"] = 0
    with pytest.raises(ContinuityError, match="identity or checksum|clean verified"):
        verify_restore_receipt(tampered, manifest)


def test_rotation_receipts_prove_old_rejection_new_acceptance_and_rollback() -> None:
    receipt = _rotation_receipt("service-token")

    assert verify_rotation_receipt(receipt, secret_kind="service-token")["rollbackTested"] is True
    incomplete = deepcopy(receipt)
    incomplete["oldCredentialRejected"] = False
    incomplete["receiptSha256"] = _digest(incomplete, "receiptSha256")
    with pytest.raises(ContinuityError, match="does not prove"):
        verify_rotation_receipt(incomplete, secret_kind="service-token")


def test_readiness_binds_backup_receipt_to_provider_destination_and_scheduler(
    tmp_path, monkeypatch
) -> None:
    definition = tmp_path / "backup-schedule.json"
    definition.write_text("{}\n", encoding="utf-8")
    manifest = _manifest()
    receipt = _backup_receipt(manifest)
    receipt["schedulerDefinitionSha256"] = hashlib.sha256(definition.read_bytes()).hexdigest()
    receipt["receiptSha256"] = _digest(receipt, "receiptSha256")
    receipt_path = tmp_path / "backup-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    policy = {
        "backup": {
            "encryptedBackupProvider": "encrypted-store-001",
            "encryptedDestinationId": "external-destination-001",
            "schedulerDefinitionPath": definition.name,
            "backupReceiptPath": receipt_path.name,
        },
        "restore": {},
        "secretRotation": {},
    }
    monkeypatch.setattr(continuity_readiness, "AI_ROOT", tmp_path)

    evidence = continuity_readiness._receipt_evidence(policy, manifest)

    assert evidence["backup"]["valid"] is True
    assert evidence["backup"]["configurationBindingsValid"] is True
    policy["backup"]["encryptedDestinationId"] = "different-destination"
    mismatched = continuity_readiness._receipt_evidence(policy, manifest)
    assert mismatched["backup"]["valid"] is False
    assert mismatched["backup"]["configurationBindingsValid"] is False


def _report_inputs(**overrides: bool) -> dict[str, object]:
    inputs: dict[str, object] = {
        "deploymentTargetAssigned": False,
        "ownerObjectivesApproved": False,
        "serviceManagerAssigned": False,
        "serviceDefinitionReady": False,
        "privateNetworkPolicyReady": False,
        "resourceLimitPolicyReady": False,
        "healthProbePolicyReady": False,
        "leastPrivilegeAccountApproved": False,
        "encryptedBackupProviderAssigned": False,
        "encryptedDestinationAssigned": False,
        "backupSchedulerReady": False,
        "disposableRestoreTargetAssigned": False,
        "restoreSchedulerReady": False,
        "secretManagerProviderAssigned": False,
        "serviceTokenRotationDefinitionReady": False,
        "signingTrustRotationDefinitionReady": False,
    }
    inputs.update(overrides)
    return inputs


def _readiness_report(
    *,
    source_ready: bool = True,
    inputs: dict[str, object] | None = None,
    backup_valid: bool = False,
    restore_valid: bool = False,
    token_valid: bool = False,
    trust_valid: bool = False,
) -> dict[str, object]:
    return build_report(
        master_backlog={"status": "complete", "items": [{"id": "VFAI-035", "status": "done"}]},
        followup_backlog={"items": [{"id": "VFAI-FU-010", "status": "accepted_for_later"}]},
        policy=json.loads(POLICY_PATH.read_text(encoding="utf-8")),
        inventory={
            "checksumValid": source_ready,
            "matchesCurrentSignedRegistry": source_ready,
            "contentFree": source_ready,
        },
        registry={"signatureValid": True},
        docker={"approvedDeploymentTarget": False},
        inputs=inputs or _report_inputs(),
        receipts={
            "backup": {"valid": backup_valid},
            "restore": {"valid": restore_valid},
            "serviceTokenRotation": {"valid": token_valid},
            "signingTrustRotation": {"valid": trust_valid},
        },
        generated_on="2026-08-31",
        source_sha256={"policy": "source"},
    )


def test_target_and_owner_objectives_are_first_external_gate() -> None:
    report = _readiness_report()

    assert report["decision"] == "await-deployment-target-and-owner-objectives"
    assert report["backupDataCopiedByEvaluator"] is False
    assert report["secretManagerAccessedByEvaluator"] is False
    validate_report(report)


def test_continuity_gates_advance_in_operational_order() -> None:
    target = _report_inputs(deploymentTargetAssigned=True, ownerObjectivesApproved=True)
    assert _readiness_report(inputs=target)["decision"] == "implement-least-privilege-service-target"
    service = _report_inputs(
        deploymentTargetAssigned=True,
        ownerObjectivesApproved=True,
        serviceManagerAssigned=True,
        serviceDefinitionReady=True,
        privateNetworkPolicyReady=True,
        resourceLimitPolicyReady=True,
        healthProbePolicyReady=True,
        leastPrivilegeAccountApproved=True,
    )
    assert _readiness_report(inputs=service)["decision"] == "implement-encrypted-backup-schedule"
    backup = deepcopy(service)
    backup.update({
        "encryptedBackupProviderAssigned": True,
        "encryptedDestinationAssigned": True,
        "backupSchedulerReady": True,
    })
    assert _readiness_report(inputs=backup, backup_valid=True)["decision"] == "run-clean-restore-test"
    restore = deepcopy(backup)
    restore.update({"disposableRestoreTargetAssigned": True, "restoreSchedulerReady": True})
    assert _readiness_report(inputs=restore, backup_valid=True, restore_valid=True)["decision"] == "implement-secret-manager-rotation"


def test_complete_external_evidence_advances_only_to_independent_review() -> None:
    inputs = _report_inputs(**{
        key: True
        for key in _report_inputs()
    })
    report = _readiness_report(
        inputs=inputs,
        backup_valid=True,
        restore_valid=True,
        token_valid=True,
        trust_valid=True,
    )

    assert report["decision"] == "continuity-results-ready-for-independent-operations-review"
    assert report["completionClaimed"] is False
    assert report["credentialsRotatedByEvaluator"] is False
    validate_report(report)


def test_tampered_readiness_receipt_fails_closed() -> None:
    report = _readiness_report()
    report["completionClaimed"] = True

    with pytest.raises(DeploymentContinuityReadinessError, match="checksum"):
        validate_report(report)
