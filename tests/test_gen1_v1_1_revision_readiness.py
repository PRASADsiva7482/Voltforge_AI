from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.evaluate_gen1_v1_1_revision_readiness import (
    Gen1V11RevisionReadinessError,
    POLICY_PATH,
    _corpus_evidence,
    build_report,
    validate_report,
)


def _master() -> dict[str, object]:
    return {"status": "completed", "items": [{"id": "VFAI-035", "status": "done"}]}


def _followups(*, immutable: bool = False) -> dict[str, object]:
    return {
        "items": [
            {"id": "VFAI-FU-013", "status": "accepted_for_later"},
            {"id": "VFAI-FU-015", "status": "done" if immutable else "proposed"},
        ]
    }


def _policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _report(
    *,
    immutable: bool = False,
    budget: bool = False,
    stage: bool = False,
    plan: bool = False,
    runs: bool = False,
    evaluation: bool = False,
) -> dict[str, object]:
    return build_report(
        master_backlog=_master(),
        followup_backlog=_followups(immutable=immutable),
        policy=_policy(),
        corpus={
            "receiptValid": True,
            "stageReady": stage,
            "currentApprovedStage": "edge-stage-1-1m" if stage else None,
            "currentUniqueTrainingPredictedTokens": 155690,
            "immutableReleaseRetentionReady": immutable,
            "ownerStorageAndComputeBudgetApproved": budget,
        },
        tokenizer={"ready": True, "version": "1.1.0"},
        historical={
            "ready": True,
            "activeArtifactId": None,
            "allExistingArtifactPackagesBindTokenizerV1": True,
        },
        completion_inputs={
            "immutableCorpusReleaseReceiptReady": immutable,
            "ownerComputeBudgetReceiptReady": budget,
            "immutableTrainingPlanReady": plan,
            "allTrainingRunReceiptsReady": runs,
            "evaluationScorecardReady": evaluation,
            "rollbackAndCanaryReceiptReady": evaluation,
        },
        generated_on="2026-09-01",
        source_sha256={"policy": "source"},
    )


def test_current_stop_gates_block_allocation() -> None:
    report = _report()

    assert report["decision"] == "await-immutable-retention-and-owner-budget"
    assert report["trainingRunsAllocatedByEvaluator"] == 0
    assert report["optimizerStepsExecutedByEvaluator"] == 0
    assert report["candidateArtifactsCreatedByEvaluator"] == 0
    validate_report(report)


def test_immutable_retention_and_budget_are_independent_gates() -> None:
    assert _report(immutable=True)["decision"] == "await-owner-compute-budget"
    assert _report(budget=True)["decision"] == "await-immutable-release-retention"


def test_governed_stage_precedes_plan_and_training() -> None:
    assert _report(immutable=True, budget=True)["decision"] == "await-governed-corpus-stage"
    assert _report(immutable=True, budget=True, stage=True)["decision"] == (
        "create-immutable-tokenizer-v1.1-training-plan"
    )


def test_plan_and_run_receipts_advance_sequentially() -> None:
    ready = dict(immutable=True, budget=True, stage=True)
    assert _report(**ready, plan=True)["decision"] == (
        "ready-to-run-approved-inactive-training-matrix"
    )
    assert _report(**ready, plan=True, runs=True)["decision"] == (
        "complete-inactive-artifact-evaluation-scorecard"
    )
    complete = _report(**ready, plan=True, runs=True, evaluation=True)
    assert complete["decision"] == "inactive-results-ready-for-independent-release-review"
    assert complete["releaseOrActivationApproved"] is False
    assert complete["completionClaimed"] is False
    validate_report(complete)


def test_policy_prohibits_relabeling_and_automatic_activation() -> None:
    policy = _policy()

    assert policy["lineageContract"]["tokenizerVersion"] == "1.1.0"
    assert policy["minimumStartConditions"]["minimumIndependentSeeds"] == 3
    assert policy["historicalBoundary"]["existingArtifactsMayBeRelabeled"] is False
    assert policy["releaseBoundary"]["automaticRegistryActivationAllowed"] is False
    assert "train-from-mutable-shard-aliases" in policy["forbidden"]


def test_corpus_authorization_flags_come_from_common_gates() -> None:
    evidence = _corpus_evidence(
        {
            "commonGates": {
                "immutableReleaseRetentionReady": False,
                "ownerStorageAndComputeBudgetApproved": False,
            }
        },
        "edge-stage-1-1m",
    )

    assert evidence["immutableReleaseRetentionReady"] is False
    assert evidence["ownerStorageAndComputeBudgetApproved"] is False


def test_tampered_receipt_fails_closed() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["trainingRunsAllocatedByEvaluator"] = 1

    with pytest.raises(Gen1V11RevisionReadinessError, match="checksum"):
        validate_report(tampered)
