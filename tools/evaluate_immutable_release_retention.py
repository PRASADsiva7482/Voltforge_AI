"""Generate or verify the content-free VFAI-FU-015 readiness receipt.

The evaluator hashes retained bytes and lineage metadata. It never copies raw
shards, creates an external backup, restores a backup, deletes a release, or
rewrites tokenizer/model history.
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

from data_governance.governance import analyze_source_removal, sha256_file  # noqa: E402
from synthetic_data.immutable_release import (  # noqa: E402
    ImmutableReleaseError,
    resolve_tokenizer_shard,
    verify_all_releases,
    verify_current_release,
)
from synthetic_data.pipeline import PIPELINE_VERSION  # noqa: E402


MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "synthetic_data/retention-policy.v1.json"
TOKENIZER_V1_1_PATH = (
    AI_ROOT / "model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json"
)
TOKENIZER_V1_0_PATH = (
    AI_ROOT / "model/tokenizers/vfdlm-byte-bpe-v1.0.0/tokenizer_manifest.json"
)
DEFAULT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/immutable-release-retention-readiness-v1.json"
)
SOURCE_PATHS = (
    MASTER_BACKLOG_PATH,
    FOLLOWUP_BACKLOG_PATH,
    POLICY_PATH,
    TOKENIZER_V1_1_PATH,
    TOKENIZER_V1_0_PATH,
    AI_ROOT / "synthetic_data/immutable_release.py",
    AI_ROOT / "synthetic_data/pipeline.py",
    AI_ROOT / "synthetic_data/pipeline-lock.v1.json",
    AI_ROOT / "data_governance/governance.py",
    AI_ROOT / "gen1_training/data.py",
    AI_ROOT / "tools/build_tokenizer.py",
    Path(__file__),
)
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class ImmutableRetentionReadinessError(RuntimeError):
    """VFAI-FU-015 readiness evidence is malformed, stale, or overclaimed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_digest(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutableRetentionReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ImmutableRetentionReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _self_digest_valid(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        return False
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == declared


def _backlog_item(backlog: Mapping[str, Any]) -> Mapping[str, Any]:
    items = backlog.get("items")
    matches = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("id") == "VFAI-FU-015"
    ] if isinstance(items, list) else []
    if len(matches) != 1:
        raise ImmutableRetentionReadinessError("expected exactly one VFAI-FU-015 item")
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


def _resolve_optional_file(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ImmutableRetentionReadinessError(f"{label} must be null or a path")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise ImmutableRetentionReadinessError(f"{label} escapes the AI workspace") from exc
    return path if path.is_file() else None


def _optional_receipt(value: Any, label: str, kind: str) -> dict[str, Any]:
    path = _resolve_optional_file(value, label)
    if path is None:
        return {"ready": False, "path": None, "receipt": {}}
    receipt = _read_json(path, label)
    return {
        "ready": receipt.get("manifestKind") == kind
        and _self_digest_valid(receipt, "reportSha256"),
        "path": path.relative_to(AI_ROOT).as_posix(),
        "receipt": receipt,
    }


def _release_evidence(policy: Mapping[str, Any]) -> dict[str, Any]:
    try:
        releases = verify_all_releases()
        current = verify_current_release()
    except ImmutableReleaseError as exc:
        return {"ready": False, "error": str(exc), "releases": []}
    contract = policy.get("immutableReleaseContract", {})
    ready = bool(
        len(releases) >= contract.get("minimumRetainedReleaseCount", 1)
        and contract.get("exactFileSetRequired") is True
        and contract.get("contentKeyIncludesEveryRetainedFile") is True
        and contract.get("existingReleaseMayBeOverwrittenOrRepaired") is False
        and contract.get("mutableAliasesMayMoveOnlyAfterPriorReleaseRetention") is True
        and contract.get("pipelineVersion") == PIPELINE_VERSION
        and current.get("contentKey") in {item.get("contentKey") for item in releases}
    )
    return {
        "ready": ready,
        "retainedReleaseCount": len(releases),
        "retainedReleaseIds": [item["releaseId"] for item in releases],
        "retainedReleaseManifestSha256": {
            item["releaseId"]: item["releaseManifestSha256"] for item in releases
        },
        "retainedFileCount": sum(int(item["fileCount"]) for item in releases),
        "retainedBytes": sum(int(item["totalBytes"]) for item in releases),
        "currentReleaseId": current.get("releaseId"),
        "currentContentKey": current.get("contentKey"),
        "externalBackupVerifiedByReleaseManifests": all(
            item.get("externalBackupVerified") is True for item in releases
        ),
    }


def _tokenizer_evidence(policy: Mapping[str, Any]) -> dict[str, Any]:
    contract = policy.get("currentTokenizerBinding", {})
    manifest = _read_json(TOKENIZER_V1_1_PATH, "tokenizer v1.1 manifest")
    shards = manifest.get("lineage", {}).get("shards", [])
    resolved: list[dict[str, Any]] = []
    failures: list[str] = []
    for descriptor in shards if isinstance(shards, list) else []:
        try:
            retained = resolve_tokenizer_shard(descriptor)
            resolved.append(retained)
        except ImmutableReleaseError as exc:
            failures.append(str(exc))
    release_ids = {item["releaseId"] for item in resolved}
    ready = bool(
        manifest.get("version") == contract.get("tokenizerVersion")
        and _self_digest_valid(manifest, "artifactSha256")
        and len(shards) == contract.get("expectedShardCount")
        and len(resolved) == len(shards)
        and len(release_ids) == 1
        and not failures
        and contract.get("approvedTokenizerManifestMayBeRewritten") is False
        and contract.get("trainingMustResolveImmutableReleaseBytes") is True
    )
    return {
        "ready": ready,
        "tokenizerVersion": manifest.get("version"),
        "tokenizerArtifactSha256": manifest.get("artifactSha256"),
        "declaredShardCount": len(shards) if isinstance(shards, list) else 0,
        "resolvedShardCount": len(resolved),
        "immutableReleaseIds": sorted(release_ids),
        "failures": failures,
        "approvedManifestRewrittenByEvaluator": False,
    }


def _historical_evidence(policy: Mapping[str, Any]) -> dict[str, Any]:
    contract = policy.get("historicalLineage", {})
    manifest = _read_json(TOKENIZER_V1_0_PATH, "tokenizer v1.0 manifest")
    shards = manifest.get("lineage", {}).get("shards", [])
    expected_hashes = {
        str(item.get("sha256"))
        for item in shards
        if isinstance(item, Mapping) and isinstance(item.get("sha256"), str)
    } if isinstance(shards, list) else set()
    search_roots = (
        AI_ROOT / "synthetic_data",
        AI_ROOT / "model/artifacts",
    )
    candidates = sorted(
        {
            path
            for root in search_roots
            if root.is_dir()
            for path in root.rglob("*.jsonl")
            if path.is_file()
        }
    )
    recovered = {
        sha256_file(path): path.relative_to(AI_ROOT).as_posix()
        for path in candidates
        if sha256_file(path) in expected_hashes
    }
    recovery = _optional_receipt(
        contract.get("trustedRecoveryReceiptPath"),
        "historical trusted recovery receipt",
        "vfai-fu-015-historical-recovery-v1",
    )
    recovery_value = recovery["receipt"]
    exact_recovery_ready = bool(
        recovery.get("ready") is True
        and recovery_value.get("status") == "recovered-exactly"
        and set(recovery_value.get("recoveredShardSha256", [])) == expected_hashes
        and recovery_value.get("historicalManifestRewritten") is False
    )
    unavailable_classification_ready = bool(
        not recovered
        and recovery.get("path") is None
        and contract.get("unavailableExactBytesClassification")
        == "non-reconstructable-from-retained-trusted-bytes"
        and contract.get("currentDataMaySubstituteForHistoricalBytes") is False
        and contract.get("historicalManifestMayBeRewritten") is False
        and contract.get("inactiveHistoricalArtifactsMayBeRelabeled") is False
    )
    return {
        "ready": bool(
            manifest.get("version") == contract.get("tokenizerVersion")
            and len(shards) == contract.get("expectedShardCount")
            and sum(int(item.get("recordCount", 0)) for item in shards)
            == contract.get("expectedRecordCount")
            and (exact_recovery_ready or unavailable_classification_ready)
        ),
        "tokenizerVersion": manifest.get("version"),
        "expectedShardCount": len(shards) if isinstance(shards, list) else 0,
        "expectedRecordCount": sum(int(item.get("recordCount", 0)) for item in shards),
        "expectedShardSha256": sorted(expected_hashes),
        "searchedJsonlFileCount": len(candidates),
        "recoveredExactShardCount": len(recovered),
        "recoveredPaths": recovered,
        "classification": (
            "recovered-exactly"
            if exact_recovery_ready
            else "non-reconstructable-from-retained-trusted-bytes"
            if unavailable_classification_ready
            else "unresolved"
        ),
        "historicalManifestRewrittenByEvaluator": False,
        "inactiveArtifactsRelabeledByEvaluator": False,
    }


def _source_removal_evidence(
    policy: Mapping[str, Any], releases: Mapping[str, Any]
) -> dict[str, Any]:
    source_ids = policy.get("sourceRemovalContract", {}).get("requiredSourceIds", [])
    reports = [analyze_source_removal(source_id) for source_id in source_ids]
    retained_counts = {
        report["sourceId"]: len(
            [item for item in report["affectedShards"] if item["retainedImmutableRelease"]]
        )
        for report in reports
    }
    expected_retained_shards = int(releases.get("retainedReleaseCount", 0)) * 4
    return {
        "ready": bool(
            source_ids
            and expected_retained_shards > 0
            and all(
                retained_counts.get(source_id, 0) == expected_retained_shards
                for source_id in source_ids
            )
            and policy.get("sourceRemovalContract", {}).get(
                "immutableReleasePathsMustBeReported"
            )
            is True
            and policy.get("sourceRemovalContract", {}).get(
                "removalImpactIsDeletionAuthorization"
            )
            is False
        ),
        "requiredSourceIds": list(source_ids),
        "expectedRetainedAffectedShardCountPerSource": expected_retained_shards,
        "retainedAffectedShardCounts": retained_counts,
        "deletionAuthorizedByEvaluator": False,
    }


def _backup_evidence(
    policy: Mapping[str, Any], releases: Mapping[str, Any]
) -> dict[str, Any]:
    owner_contract = policy.get("ownerBackupPolicy", {})
    owner = _optional_receipt(
        owner_contract.get("approvalReceiptPath"),
        "owner backup policy receipt",
        "vfai-fu-015-owner-backup-retention-policy-v1",
    )
    owner_value = owner["receipt"]
    owner_ready = bool(
        owner.get("ready") is True
        and owner_value.get("status") == "approved-by-owner"
        and all(
            isinstance(owner_value.get(field), (int, float))
            and not isinstance(owner_value.get(field), bool)
            and owner_value[field] > 0
            for field in owner_contract.get("requiredPositiveFields", [])
        )
        and all(
            owner_value.get(field) is True
            for field in owner_contract.get("requiredTrueFields", [])
        )
        and isinstance(owner_value.get("storageOwnerId"), str)
        and bool(owner_value.get("storageOwnerId"))
    )
    external = policy.get("externalBackup", {})
    backup = _optional_receipt(
        external.get("backupReceiptPath"),
        "external backup receipt",
        "vfai-fu-015-external-backup-v1",
    )
    expected_ids = set(releases.get("retainedReleaseIds", []))
    backup_value = backup["receipt"]
    backup_ready = bool(
        backup.get("ready") is True
        and backup_value.get("status") == "pass"
        and set(backup_value.get("retainedReleaseIds", [])) == expected_ids
        and backup_value.get("encryptedAtRest") is True
        and backup_value.get("checksumInventoryVerified") is True
        and backup_value.get("independentCopyCount", 0)
        >= owner_value.get("minimumIndependentCopies", 1)
    )
    restore = _optional_receipt(
        external.get("restoreReceiptPath"),
        "external restore receipt",
        "vfai-fu-015-external-restore-v1",
    )
    restore_value = restore["receipt"]
    restore_ready = bool(
        restore.get("ready") is True
        and restore_value.get("status") == "pass"
        and set(restore_value.get("restoredReleaseIds", [])) == expected_ids
        and restore_value.get("exactFileInventoryVerified") is True
        and restore_value.get("sourceRemovalImpactVerified") is True
        and isinstance(restore_value.get("restoreMinutes"), (int, float))
        and not isinstance(restore_value.get("restoreMinutes"), bool)
        and restore_value.get("restoreMinutes", float("inf"))
        <= owner_value.get("maximumRestoreMinutes", -1)
    )
    return {
        "ownerBackupPolicyReady": owner_ready,
        "externalBackupReady": backup_ready,
        "externalRestoreReady": restore_ready,
        "configuredPaths": {
            "ownerBackupPolicy": owner.get("path"),
            "externalBackup": backup.get("path"),
            "externalRestore": restore.get("path"),
        },
    }


def _expected_decision(gates: Mapping[str, Any]) -> str:
    if gates.get("masterBacklogComplete") is not True:
        return "await-master-backlog-completion"
    if gates.get("immutableReleasesReady") is not True:
        return "repair-immutable-release-retention"
    if gates.get("currentTokenizerBindingReady") is not True:
        return "repair-current-tokenizer-immutable-binding"
    if gates.get("pipelineNonOverwriteReady") is not True:
        return "repair-future-release-non-overwrite-enforcement"
    if gates.get("sourceRemovalCoverageReady") is not True:
        return "repair-immutable-source-removal-impact"
    if gates.get("historicalClassificationReady") is not True:
        return "review-historical-lineage-recovery"
    if gates.get("ownerBackupPolicyReady") is not True:
        return "await-owner-backup-retention-policy"
    if gates.get("externalBackupReady") is not True:
        return "create-approved-external-release-backup"
    if gates.get("externalRestoreReady") is not True:
        return "await-external-backup-restore-verification"
    return "retention-results-ready-for-independent-review"


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    releases: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    historical: Mapping[str, Any],
    source_removal: Mapping[str, Any],
    backup: Mapping[str, Any],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-015":
        raise ImmutableRetentionReadinessError("policy is not bound to VFAI-FU-015")
    item = _backlog_item(followup_backlog)
    pipeline_ready = bool(
        policy.get("immutableReleaseContract", {}).get(
            "existingReleaseMayBeOverwrittenOrRepaired"
        )
        is False
        and policy.get("immutableReleaseContract", {}).get(
            "mutableAliasesMayMoveOnlyAfterPriorReleaseRetention"
        )
        is True
        and PIPELINE_VERSION == "1.3.0"
    )
    gates = {
        "masterBacklogComplete": _master_complete(master_backlog),
        "immutableReleasesReady": releases.get("ready") is True,
        "currentTokenizerBindingReady": tokenizer.get("ready") is True,
        "pipelineNonOverwriteReady": pipeline_ready,
        "sourceRemovalCoverageReady": source_removal.get("ready") is True,
        "historicalClassificationReady": historical.get("ready") is True,
        "ownerBackupPolicyReady": backup.get("ownerBackupPolicyReady") is True,
        "externalBackupReady": backup.get("externalBackupReady") is True,
        "externalRestoreReady": backup.get("externalRestoreReady") is True,
    }
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-015-immutable-release-retention-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": item.get("id"), "recordedStatus": item.get("status")},
        "policy": {"policyId": policy.get("policyId"), "status": policy.get("status")},
        "immutableReleases": dict(releases),
        "currentTokenizerBinding": dict(tokenizer),
        "historicalTokenizerLineage": dict(historical),
        "sourceRemovalImpact": dict(source_removal),
        "backupAndRestore": dict(backup),
        "gates": gates,
        "decision": _expected_decision(gates),
        "completionClaimed": False,
        "rawShardBytesCopiedByEvaluator": 0,
        "externalBackupsCreatedByEvaluator": 0,
        "restoreDrillsExecutedByEvaluator": 0,
        "releasesDeletedOrRewrittenByEvaluator": 0,
        "tokenizerOrModelManifestsRewrittenByEvaluator": 0,
        "historicalArtifactsRelabeledByEvaluator": 0,
        "networkAccessed": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _report_digest(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _report_digest(report):
        raise ImmutableRetentionReadinessError("VFAI-FU-015 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or report.get("decision") != _expected_decision(gates):
        raise ImmutableRetentionReadinessError("VFAI-FU-015 receipt decision is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("rawShardBytesCopiedByEvaluator") != 0,
            report.get("externalBackupsCreatedByEvaluator") != 0,
            report.get("restoreDrillsExecutedByEvaluator") != 0,
            report.get("releasesDeletedOrRewrittenByEvaluator") != 0,
            report.get("tokenizerOrModelManifestsRewrittenByEvaluator") != 0,
            report.get("historicalArtifactsRelabeledByEvaluator") != 0,
            report.get("networkAccessed") is not False,
        )
    ):
        raise ImmutableRetentionReadinessError("VFAI-FU-015 receipt overclaims mutation")


def _current_inputs() -> dict[str, Any]:
    policy = _read_json(POLICY_PATH, "immutable retention policy")
    releases = _release_evidence(policy)
    tokenizer = _tokenizer_evidence(policy)
    historical = _historical_evidence(policy)
    source_removal = _source_removal_evidence(policy, releases)
    backup = _backup_evidence(policy, releases)
    release_manifests = sorted(
        (AI_ROOT / "synthetic_data/releases").glob("*/*/release-manifest.json")
    )
    source_paths = (*SOURCE_PATHS, *release_manifests)
    return {
        "master_backlog": _read_json(MASTER_BACKLOG_PATH, "master backlog"),
        "followup_backlog": _read_json(FOLLOWUP_BACKLOG_PATH, "follow-up backlog"),
        "policy": policy,
        "releases": releases,
        "tokenizer": tokenizer,
        "historical": historical,
        "source_removal": source_removal,
        "backup": backup,
        "source_sha256": {
            path.relative_to(AI_ROOT).as_posix(): sha256_file(path) for path in source_paths
        },
    }


def _build_current_report(generated_on: str) -> dict[str, Any]:
    return build_report(generated_on=generated_on, **_current_inputs())


def evaluate(path: Path, generated_on: str) -> dict[str, Any]:
    report = _build_current_report(generated_on)
    _write_json(path, report)
    return report


def verify(path: Path) -> dict[str, Any]:
    report = _read_json(path, "VFAI-FU-015 readiness receipt")
    validate_report(report)
    if report != _build_current_report(str(report.get("generatedOn"))):
        raise ImmutableRetentionReadinessError("VFAI-FU-015 readiness receipt is stale")
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
    print(
        json.dumps(
            {
                "ok": True,
                "command": args.command,
                "reportId": report["reportId"],
                "decision": report["decision"],
                "retainedReleaseCount": report["immutableReleases"].get(
                    "retainedReleaseCount"
                ),
                "retainedBytes": report["immutableReleases"].get("retainedBytes"),
                "currentTokenizerResolvedShardCount": report[
                    "currentTokenizerBinding"
                ].get("resolvedShardCount"),
                "historicalClassification": report["historicalTokenizerLineage"].get(
                    "classification"
                ),
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
