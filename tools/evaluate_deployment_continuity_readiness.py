"""Generate or verify the VFAI-FU-010 deployment-continuity readiness receipt.

The evaluator is content-free and read-only. It does not create a backup,
restore files, access a secret manager, rotate a credential, or schedule work.
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
from operations.continuity import (  # noqa: E402
    ContinuityError,
    validate_inventory,
    verify_backup_receipt,
    verify_owner_objectives_receipt,
    verify_restore_receipt,
    verify_rotation_receipt,
)


MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "operations/deployment-continuity-policy.v1.json"
INVENTORY_PATH = AI_ROOT / "evaluation/reports/continuity-source-inventory-v1.json"
REGISTRY_PATH = AI_ROOT / "model/registry/active_model.json"
TRUST_STORE_PATH = AI_ROOT / "model/registry/trust/trusted-keys.json"
CONTINUITY_PATH = AI_ROOT / "operations/continuity.py"
CLI_PATH = AI_ROOT / "tools/manage_continuity.py"
DOCKERFILE_PATH = AI_ROOT / "Dockerfile"
DEFAULT_REPORT_PATH = AI_ROOT / "evaluation/reports/deployment-continuity-readiness-v1.json"
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class DeploymentContinuityReadinessError(RuntimeError):
    """VFAI-FU-010 readiness evidence is invalid or stale."""


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


def _report_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentContinuityReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeploymentContinuityReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _backlog_item(backlog: Mapping[str, Any]) -> Mapping[str, Any]:
    items = backlog.get("items")
    matches = [
        item
        for item in items if isinstance(item, Mapping) and item.get("id") == "VFAI-FU-010"
    ] if isinstance(items, list) else []
    if len(matches) != 1:
        raise DeploymentContinuityReadinessError("expected exactly one VFAI-FU-010 item")
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


def _workspace_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise DeploymentContinuityReadinessError("continuity evidence path escapes the workspace") from exc
    return path


def _existing_file(value: Any) -> bool:
    path = _workspace_path(value)
    return path is not None and path.is_file()


def _inventory_evidence(
    inventory: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        validate_inventory(inventory)
        valid = True
    except ContinuityError:
        valid = False
    registered = registry.get("artifacts")
    verified = inventory.get("verifiedArtifacts")
    source_matches = bool(
        valid
        and inventory.get("registry", {}).get("revision") == registry.get("revision")
        and inventory.get("registry", {}).get("registrySha256")
        == registry.get("registrySha256")
        and isinstance(registered, list)
        and isinstance(verified, list)
        and len(verified) == len(registered)
    )
    return {
        "path": INVENTORY_PATH.relative_to(AI_ROOT).as_posix(),
        "manifestSha256": inventory.get("manifestSha256"),
        "checksumValid": valid,
        "matchesCurrentSignedRegistry": source_matches,
        "entryCount": inventory.get("entryCount"),
        "totalBytes": inventory.get("totalBytes"),
        "verifiedArtifactCount": len(verified) if isinstance(verified, list) else 0,
        "registryHistoryEntryCount": inventory.get("registryHistoryEntryCount"),
        "signedReleaseEntryCount": inventory.get("signedReleaseEntryCount"),
        "backupCreated": inventory.get("backupCreated"),
        "encryptionApplied": inventory.get("encryptionApplied"),
        "restoreTestExecuted": inventory.get("restoreTestExecuted"),
        "contentFree": all(
            inventory.get(field) is False
            for field in (
                "environmentFileIncluded",
                "privateSigningKeyIncluded",
                "sourceRootStored",
                "rawSecretStored",
            )
        ),
    }


def _docker_evidence(source: str) -> dict[str, Any]:
    return {
        "path": DOCKERFILE_PATH.relative_to(AI_ROOT).as_posix(),
        "baseImage": "python:3.11-slim-bookworm"
        if "FROM python:3.11-slim-bookworm" in source
        else None,
        "nonRootUserDeclared": "\nUSER " in f"\n{source}",
        "healthcheckDeclared": "\nHEALTHCHECK " in f"\n{source}",
        "bindsAllInterfaces": "VOLTFORGE_AI_HOST=0.0.0.0" in source,
        "approvedDeploymentTarget": False,
        "classification": "development-container-template-not-approved-deployment",
    }


def _deployment_inputs(policy: Mapping[str, Any]) -> dict[str, Any]:
    deployment = policy.get("deployment", {})
    objectives = policy.get("ownerObjectives", {})
    backup = policy.get("backup", {})
    restore = policy.get("restore", {})
    rotation = policy.get("secretRotation", {})
    approval_path = _workspace_path(objectives.get("approvalReceiptPath"))
    try:
        approval = (
            verify_owner_objectives_receipt(
                _read_json(approval_path, "owner-objectives receipt")
            )
            if approval_path is not None and approval_path.is_file()
            else {}
        )
    except ContinuityError:
        approval = {}
    objective_keys = (
        "recoveryPointObjectiveSeconds",
        "recoveryTimeObjectiveSeconds",
        "availabilityObjectivePercent",
        "retentionDays",
    )
    owner_ready = bool(
        approval
        and all(approval.get(key) == objectives.get(key) for key in objective_keys)
        and approval.get("dataOwnerApproved") is objectives.get("dataOwnerApproved") is True
        and approval.get("operationsOwnerApproved")
        is objectives.get("operationsOwnerApproved")
        is True
    )
    return {
        "deploymentTargetId": deployment.get("targetId"),
        "deploymentTargetAssigned": bool(deployment.get("targetId")),
        "ownerObjectivesApproved": owner_ready,
        "ownerObjectivesReceiptSha256": approval.get("receiptSha256"),
        "serviceManagerAssigned": bool(deployment.get("serviceManager")),
        "serviceDefinitionReady": _existing_file(deployment.get("serviceDefinitionPath")),
        "privateNetworkPolicyReady": _existing_file(deployment.get("privateNetworkPolicyPath")),
        "resourceLimitPolicyReady": _existing_file(deployment.get("resourceLimitPolicyPath")),
        "healthProbePolicyReady": _existing_file(deployment.get("healthProbePolicyPath")),
        "leastPrivilegeAccountApproved": deployment.get("leastPrivilegeAccountApproved") is True,
        "encryptedBackupProviderAssigned": bool(backup.get("encryptedBackupProvider")),
        "encryptedDestinationAssigned": bool(backup.get("encryptedDestinationId")),
        "backupSchedulerReady": _existing_file(backup.get("schedulerDefinitionPath")),
        "backupReceiptPath": backup.get("backupReceiptPath"),
        "disposableRestoreTargetAssigned": bool(restore.get("disposableRestoreTargetId")),
        "restoreSchedulerReady": _existing_file(restore.get("restoreSchedulerDefinitionPath")),
        "restoreReceiptPath": restore.get("restoreReceiptPath"),
        "secretManagerProviderAssigned": bool(rotation.get("secretManagerProvider")),
        "serviceTokenRotationDefinitionReady": _existing_file(rotation.get("serviceTokenRotationDefinitionPath")),
        "signingTrustRotationDefinitionReady": _existing_file(rotation.get("signingTrustRotationDefinitionPath")),
        "serviceTokenRotationReceiptPath": rotation.get("serviceTokenRotationReceiptPath"),
        "signingTrustRotationReceiptPath": rotation.get("signingTrustRotationReceiptPath"),
    }


def _receipt_evidence(
    policy: Mapping[str, Any], inventory: Mapping[str, Any]
) -> dict[str, Any]:
    backup = policy.get("backup", {})
    restore = policy.get("restore", {})
    rotation = policy.get("secretRotation", {})

    def verify_optional(path_value: Any, verifier: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        path = _workspace_path(path_value)
        if path is None:
            return {"path": None, "assigned": False, "valid": False, "receiptSha256": None}
        if not path.is_file():
            return {"path": path_value, "assigned": True, "valid": False, "receiptSha256": None}
        receipt = _read_json(path, "continuity receipt")
        try:
            verified = verifier(receipt, *args, **kwargs)
            valid = True
        except ContinuityError:
            verified = {}
            valid = False
        return {
            "path": path_value,
            "assigned": True,
            "valid": valid,
            "receiptSha256": receipt.get("receiptSha256"),
            "verified": verified,
        }

    def definition_sha256(path_value: Any) -> str | None:
        path = _workspace_path(path_value)
        return _sha256_file(path) if path is not None and path.is_file() else None

    def bind(receipt: dict[str, Any], bindings_valid: bool) -> dict[str, Any]:
        receipt["configurationBindingsValid"] = bindings_valid
        receipt["valid"] = receipt.get("valid") is True and bindings_valid
        return receipt

    backup_receipt = verify_optional(
        backup.get("backupReceiptPath"), verify_backup_receipt, inventory
    )
    backup_verified = backup_receipt.get("verified", {})
    bind(
        backup_receipt,
        bool(
            backup_receipt.get("valid") is True
            and backup_verified.get("providerId") == backup.get("encryptedBackupProvider")
            and backup_verified.get("destinationId") == backup.get("encryptedDestinationId")
            and backup_verified.get("schedulerDefinitionSha256")
            == definition_sha256(backup.get("schedulerDefinitionPath"))
        ),
    )
    restore_receipt = verify_optional(
        restore.get("restoreReceiptPath"), verify_restore_receipt, inventory
    )
    restore_verified = restore_receipt.get("verified", {})
    bind(
        restore_receipt,
        bool(
            restore_receipt.get("valid") is True
            and restore_verified.get("restoreTargetId")
            == restore.get("disposableRestoreTargetId")
            and restore_verified.get("restoreSchedulerDefinitionSha256")
            == definition_sha256(restore.get("restoreSchedulerDefinitionPath"))
        ),
    )

    def rotation_receipt(kind: str, receipt_key: str, definition_key: str) -> dict[str, Any]:
        evidence = verify_optional(
            rotation.get(receipt_key), verify_rotation_receipt, secret_kind=kind
        )
        verified = evidence.get("verified", {})
        return bind(
            evidence,
            bool(
                evidence.get("valid") is True
                and verified.get("secretManagerProvider")
                == rotation.get("secretManagerProvider")
                and verified.get("rotationDefinitionSha256")
                == definition_sha256(rotation.get(definition_key))
            ),
        )

    return {
        "backup": backup_receipt,
        "restore": restore_receipt,
        "serviceTokenRotation": rotation_receipt(
            "service-token",
            "serviceTokenRotationReceiptPath",
            "serviceTokenRotationDefinitionPath",
        ),
        "signingTrustRotation": rotation_receipt(
            "signing-trust",
            "signingTrustRotationReceiptPath",
            "signingTrustRotationDefinitionPath",
        ),
    }


def _expected_decision(gates: Mapping[str, Any]) -> str:
    if gates.get("masterBacklogComplete") is not True:
        return "await-master-backlog-completion"
    if gates.get("sourceInventoryReady") is not True:
        return "repair-source-inventory"
    if gates.get("deploymentTargetAndOwnerObjectivesReady") is not True:
        return "await-deployment-target-and-owner-objectives"
    if gates.get("leastPrivilegeServiceTargetReady") is not True:
        return "implement-least-privilege-service-target"
    if gates.get("encryptedBackupScheduleAndReceiptReady") is not True:
        return "implement-encrypted-backup-schedule"
    if gates.get("cleanRestoreScheduleAndReceiptReady") is not True:
        return "run-clean-restore-test"
    if gates.get("secretManagerRotationDefinitionsReady") is not True:
        return "implement-secret-manager-rotation"
    if gates.get("tokenAndTrustRotationReceiptsReady") is not True:
        return "run-token-and-trust-rotation-tests"
    return "continuity-results-ready-for-independent-operations-review"


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    inventory: Mapping[str, Any],
    registry: Mapping[str, Any],
    docker: Mapping[str, Any],
    inputs: Mapping[str, Any],
    receipts: Mapping[str, Any],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-010":
        raise DeploymentContinuityReadinessError("policy is not bound to VFAI-FU-010")
    fu010 = _backlog_item(followup_backlog)
    source_ready = bool(
        inventory.get("checksumValid")
        and inventory.get("matchesCurrentSignedRegistry")
        and inventory.get("contentFree")
    )
    target_ready = bool(
        inputs.get("deploymentTargetAssigned") and inputs.get("ownerObjectivesApproved")
    )
    service_ready = all(
        inputs.get(key) is True
        for key in (
            "serviceManagerAssigned",
            "serviceDefinitionReady",
            "privateNetworkPolicyReady",
            "resourceLimitPolicyReady",
            "healthProbePolicyReady",
            "leastPrivilegeAccountApproved",
        )
    )
    backup_ready = all(
        inputs.get(key) is True
        for key in (
            "encryptedBackupProviderAssigned",
            "encryptedDestinationAssigned",
            "backupSchedulerReady",
        )
    ) and receipts.get("backup", {}).get("valid") is True
    restore_ready = all(
        inputs.get(key) is True
        for key in ("disposableRestoreTargetAssigned", "restoreSchedulerReady")
    ) and receipts.get("restore", {}).get("valid") is True
    rotation_definitions_ready = all(
        inputs.get(key) is True
        for key in (
            "secretManagerProviderAssigned",
            "serviceTokenRotationDefinitionReady",
            "signingTrustRotationDefinitionReady",
        )
    )
    rotation_receipts_ready = all(
        receipts.get(key, {}).get("valid") is True
        for key in ("serviceTokenRotation", "signingTrustRotation")
    )
    gates = {
        "masterBacklogComplete": _master_complete(master_backlog),
        "sourceInventoryReady": source_ready,
        "deploymentTargetAndOwnerObjectivesReady": target_ready,
        "leastPrivilegeServiceTargetReady": service_ready,
        "encryptedBackupScheduleAndReceiptReady": backup_ready,
        "cleanRestoreScheduleAndReceiptReady": restore_ready,
        "secretManagerRotationDefinitionsReady": rotation_definitions_ready,
        "tokenAndTrustRotationReceiptsReady": rotation_receipts_ready,
    }
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-010-deployment-continuity-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": fu010.get("id"), "recordedStatus": fu010.get("status")},
        "policy": {"policyId": policy.get("policyId"), "status": policy.get("status")},
        "sourceInventory": dict(inventory),
        "registry": dict(registry),
        "deploymentTemplate": dict(docker),
        "assignedInputs": dict(inputs),
        "receipts": dict(receipts),
        "gates": gates,
        "decision": _expected_decision(gates),
        "completionClaimed": False,
        "backupDataCopiedByEvaluator": False,
        "encryptionPerformedByEvaluator": False,
        "restoreExecutedByEvaluator": False,
        "secretManagerAccessedByEvaluator": False,
        "credentialsRotatedByEvaluator": False,
        "hostTaskScheduledByEvaluator": False,
        "liveStateMutated": False,
        "networkAccessed": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _report_digest(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _report_digest(report):
        raise DeploymentContinuityReadinessError("VFAI-FU-010 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or report.get("decision") != _expected_decision(gates):
        raise DeploymentContinuityReadinessError("VFAI-FU-010 receipt decision is inconsistent")
    if any(
        report.get(field) is not False
        for field in (
            "completionClaimed",
            "backupDataCopiedByEvaluator",
            "encryptionPerformedByEvaluator",
            "restoreExecutedByEvaluator",
            "secretManagerAccessedByEvaluator",
            "credentialsRotatedByEvaluator",
            "hostTaskScheduledByEvaluator",
            "liveStateMutated",
            "networkAccessed",
        )
    ):
        raise DeploymentContinuityReadinessError("VFAI-FU-010 receipt overclaims operations")


def _current_inputs() -> dict[str, Any]:
    master = _read_json(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json(POLICY_PATH, "deployment continuity policy")
    source_inventory = _read_json(INVENTORY_PATH, "continuity source inventory")
    try:
        registry = verify_registry(REGISTRY_PATH, trust_store_path=TRUST_STORE_PATH)
    except RegistryManagerError as exc:
        raise DeploymentContinuityReadinessError("signed registry verification failed") from exc
    inventory = _inventory_evidence(source_inventory, registry)
    inputs = _deployment_inputs(policy)
    receipts = _receipt_evidence(policy, source_inventory)
    source_paths = [
        MASTER_BACKLOG_PATH,
        FOLLOWUP_BACKLOG_PATH,
        POLICY_PATH,
        INVENTORY_PATH,
        REGISTRY_PATH,
        TRUST_STORE_PATH,
        CONTINUITY_PATH,
        CLI_PATH,
        DOCKERFILE_PATH,
        Path(__file__),
    ]
    for section in ("deployment", "ownerObjectives", "backup", "restore", "secretRotation"):
        for value in policy.get(section, {}).values():
            path = _workspace_path(value)
            if path is not None and path.is_file() and path not in source_paths:
                source_paths.append(path)
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "inventory": inventory,
        "registry": {
            "signatureValid": True,
            "revision": registry.get("revision"),
            "registrySha256": registry.get("registrySha256"),
            "activeArtifactId": registry.get("activeArtifactId"),
            "registeredArtifactCount": len(registry.get("artifacts", [])),
        },
        "docker": _docker_evidence(DOCKERFILE_PATH.read_text(encoding="utf-8")),
        "inputs": inputs,
        "receipts": receipts,
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


def verify(path: Path) -> dict[str, Any]:
    report = _read_json(path, "VFAI-FU-010 readiness receipt")
    validate_report(report)
    if report != _build_current_report(str(report.get("generatedOn"))):
        raise DeploymentContinuityReadinessError("VFAI-FU-010 readiness receipt is stale")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--generated-on", required=True)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = (
        evaluate(args.output.resolve(), args.generated_on)
        if args.command == "evaluate"
        else verify(args.input.resolve())
    )
    print(json.dumps({
        "ok": True,
        "command": args.command,
        "reportId": report["reportId"],
        "decision": report["decision"],
        "sourceInventoryReady": report["gates"]["sourceInventoryReady"],
        "deploymentTargetAssigned": report["assignedInputs"]["deploymentTargetAssigned"],
        "reportSha256": report["reportSha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
