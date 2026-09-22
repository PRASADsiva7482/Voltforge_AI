from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.evaluate_context_revision_readiness import (
    ContextRevisionReadinessError,
    POLICY_PATH,
    build_report,
    validate_report,
)


def _master() -> dict[str, object]:
    return {"status": "completed", "items": [{"id": "VFAI-035", "status": "done"}]}


def _followups() -> dict[str, object]:
    return {"items": [{"id": "VFAI-FU-008", "status": "accepted_for_later"}]}


def _policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _evidence(ready: bool, **values: object) -> dict[str, object]:
    return {"ready": ready, **values}


def _report(
    *,
    pilot_stage: bool = False,
    selection_stage: bool = False,
    release_stage: bool = False,
    budget: bool = False,
    migration: bool = False,
    assets: bool = False,
    results: bool = False,
) -> dict[str, object]:
    return build_report(
        master_backlog=_master(),
        followup_backlog=_followups(),
        policy=_policy(),
        corpus={
            "receiptValid": True,
            "currentApprovedStage": None,
            "currentUniqueTrainingPredictedTokens": 155690,
            "pilotStageReady": pilot_stage,
            "selectionStageReady": selection_stage,
            "releaseStageReady": release_stage,
        },
        historical_context=_evidence(True, tokenizerVersion="1.0.0"),
        tokenizer=_evidence(True, version="1.1.0"),
        artifacts=_evidence(True, allExistingArtifactsBelowMinimum=True),
        migration=_evidence(migration, migrationRequired=True),
        owner_budget=_evidence(budget),
        candidate_assets=_evidence(assets),
        results=_evidence(results),
        generated_on="2026-08-31",
        source_sha256={"policy": "source"},
    )


def test_current_corpus_blocks_training_before_allocation() -> None:
    report = _report()

    assert report["decision"] == "await-governed-corpus-stage"
    assert report["gates"]["pilotCorpusStageReady"] is False
    assert report["trainingRunsExecutedByEvaluator"] == 0
    assert report["completionClaimed"] is False
    validate_report(report)


def test_owner_budget_is_required_after_pilot_corpus_stage() -> None:
    report = _report(pilot_stage=True)

    assert report["decision"] == "await-owner-compute-budget"
    assert report["gates"]["ownerComputeBudgetApproved"] is False
    validate_report(report)


def test_tokenizer_migration_is_a_separate_gate() -> None:
    report = _report(pilot_stage=True, budget=True)

    assert report["decision"] == "implement-tokenizer-bound-context-compiler-revision"
    assert report["historicalContext"]["tokenizerVersion"] == "1.0.0"
    assert report["approvedTrainingTokenizer"]["version"] == "1.1.0"
    validate_report(report)


def test_candidate_and_packing_manifests_precede_training() -> None:
    report = _report(pilot_stage=True, budget=True, migration=True)

    assert report["decision"] == "implement-long-context-candidate-assets"
    assert report["gates"]["allCandidateAndPackingManifestsReady"] is False
    validate_report(report)


def test_release_stage_precedes_final_candidate_matrix() -> None:
    report = _report(
        pilot_stage=True,
        selection_stage=True,
        budget=True,
        migration=True,
        assets=True,
    )

    assert report["decision"] == "await-release-corpus-stage"
    validate_report(report)


def test_accepted_results_advance_only_to_independent_release_review() -> None:
    report = _report(
        pilot_stage=True,
        selection_stage=True,
        release_stage=True,
        budget=True,
        migration=True,
        assets=True,
        results=True,
    )

    assert report["decision"] == "context-results-ready-for-independent-release-review"
    assert report["completionClaimed"] is False
    assert report["releaseOrActivationApproved"] is False
    validate_report(report)


def test_policy_freezes_four_contexts_and_prohibits_relabeling() -> None:
    policy = _policy()

    assert [item["contextLength"] for item in policy["candidates"]] == [
        1024,
        2048,
        4096,
        8192,
    ]
    assert policy["immutableBoundaries"]["existingArtifactsMayBeRelabeled"] is False
    assert policy["experimentProtocol"]["minimumIndependentSeeds"] == 3
    assert "relabel-128-token-artifact" in policy["forbidden"]


def test_tampered_readiness_receipt_fails_closed() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["completionClaimed"] = True

    with pytest.raises(ContextRevisionReadinessError, match="checksum"):
        validate_report(tampered)
