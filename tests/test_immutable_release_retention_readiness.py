from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.evaluate_immutable_release_retention import (
    ImmutableRetentionReadinessError,
    POLICY_PATH,
    _historical_evidence,
    build_report,
    validate_report,
)


def _policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _report(
    *,
    owner: bool = False,
    backup: bool = False,
    restore: bool = False,
) -> dict[str, object]:
    return build_report(
        master_backlog={
            "status": "completed",
            "items": [{"id": "VFAI-035", "status": "done"}],
        },
        followup_backlog={
            "items": [{"id": "VFAI-FU-015", "status": "accepted_for_later"}]
        },
        policy=_policy(),
        releases={"ready": True},
        tokenizer={"ready": True},
        historical={"ready": True},
        source_removal={"ready": True},
        backup={
            "ownerBackupPolicyReady": owner,
            "externalBackupReady": backup,
            "externalRestoreReady": restore,
        },
        generated_on="2026-09-01",
        source_sha256={"policy": "source"},
    )


def test_current_checkpoint_waits_for_owner_backup_policy() -> None:
    report = _report()

    assert report["decision"] == "await-owner-backup-retention-policy"
    assert report["rawShardBytesCopiedByEvaluator"] == 0
    assert report["externalBackupsCreatedByEvaluator"] == 0
    assert report["restoreDrillsExecutedByEvaluator"] == 0
    validate_report(report)


def test_backup_and_restore_gates_advance_sequentially() -> None:
    assert _report(owner=True)["decision"] == "create-approved-external-release-backup"
    assert _report(owner=True, backup=True)["decision"] == (
        "await-external-backup-restore-verification"
    )
    complete = _report(owner=True, backup=True, restore=True)
    assert complete["decision"] == "retention-results-ready-for-independent-review"
    assert complete["completionClaimed"] is False
    validate_report(complete)


def test_policy_prohibits_overwrite_relabel_and_current_substitution() -> None:
    policy = _policy()

    assert policy["immutableReleaseContract"][  # type: ignore[index]
        "existingReleaseMayBeOverwrittenOrRepaired"
    ] is False
    assert policy["currentTokenizerBinding"][  # type: ignore[index]
        "approvedTokenizerManifestMayBeRewritten"
    ] is False
    assert policy["historicalLineage"][  # type: ignore[index]
        "currentDataMaySubstituteForHistoricalBytes"
    ] is False
    assert policy["ownerBackupPolicy"]["approvalReceiptPath"] is None  # type: ignore[index]
    assert "overwrite-content-addressed-release" in policy["forbidden"]  # type: ignore[operator]


def test_historical_lineage_is_explicitly_non_reconstructable() -> None:
    evidence = _historical_evidence(_policy())

    assert evidence["ready"] is True
    assert evidence["expectedShardCount"] == 4
    assert evidence["expectedRecordCount"] == 154
    assert evidence["recoveredExactShardCount"] == 0
    assert evidence["classification"] == (
        "non-reconstructable-from-retained-trusted-bytes"
    )
    assert evidence["historicalManifestRewrittenByEvaluator"] is False


def test_tampered_readiness_receipt_fails_closed() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["historicalArtifactsRelabeledByEvaluator"] = 1

    with pytest.raises(ImmutableRetentionReadinessError, match="checksum"):
        validate_report(tampered)
