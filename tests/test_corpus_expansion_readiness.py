from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.evaluate_corpus_expansion_readiness import (
    CorpusExpansionReadinessError,
    POLICY_PATH,
    build_report,
    validate_report,
)


def _master() -> dict[str, object]:
    return {"status": "completed", "items": [{"id": "VFAI-035", "status": "done"}]}


def _followups(fu015_status: str = "proposed") -> dict[str, object]:
    return {
        "items": [
            {"id": "VFAI-FU-006", "status": "accepted_for_later"},
            {"id": "VFAI-FU-015", "status": fu015_status},
        ]
    }


def _policy(*, budget_approved: bool = False) -> dict[str, object]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if budget_approved:
        policy["budgetApproval"] = {
            "status": "approved-by-owner",
            "maximumRetainedBytes": 1_000_000_000,
            "maximumGenerationCpuHours": 100,
            "maximumCompilerCpuHours": 100,
            "maximumTrainingGpuHours": 100,
            "approvedHardwareBoundary": "owner-controlled-local-hardware",
        }
    return policy


def _current(*, unique_tokens: int = 155_690, quality_ready: bool = False) -> dict[str, object]:
    current: dict[str, object] = {
        "datasetId": "approved-corpus",
        "tokenizerVersion": "1.1.0",
        "packedCorpusFingerprint": "corpus",
        "trainingRecordCount": 227,
        "modelValidationRecords": 23,
        "uniqueTrainingPredictedTokens": unique_tokens,
        "approvedSourceFamilies": 2,
        "approvedSourceIds": ["source-a", "source-b"],
        "sourceAdmissionReady": True,
        "exactBoardVariants": 11,
        "exactComponentVariants": 1,
        "compilerToolchains": 5,
        "globalHeldOutCases": 26,
        "retrievalHeldOutCases": 27,
        "adversarialSafetyCases": 2,
        "manualAuditRecords": 0,
        "acceptedExactDuplicateCount": 0,
        "trainingHeldOutCollisionCount": 0,
        "heldOutCollisionCandidatesRejected": 0,
        "deterministicLabelReceiptCoveragePercent": 100.0,
        "compilerLabelReceiptCoveragePercent": 100.0,
        "semanticDeduplicationEnforced": False,
        "unsupportedClaimUncertaintyCoveragePercent": None,
        "criticalManualAuditErrorCount": None,
        "forbiddenInputsDisabled": True,
        "privateConsentWorkflowImplemented": False,
        "electronicsKnowledgeRecords": 61,
        "taskPercent": {
            "circuit_validation": 13.2,
            "compiler_repair": 8.8,
            "domain_chat": 17.6,
            "firmware_generation": 17.6,
            "pin_routing": 8.8,
            "refusal": 4.4,
            "structured_output": 8.8,
            "uncertainty": 3.2,
            "wiring": 17.6,
        },
    }
    if quality_ready:
        current.update(
            {
                "modelValidationRecords": 500,
                "approvedSourceFamilies": 3,
                "exactComponentVariants": 10,
                "globalHeldOutCases": 260,
                "retrievalHeldOutCases": 100,
                "adversarialSafetyCases": 50,
                "manualAuditRecords": 200,
                "semanticDeduplicationEnforced": True,
                "unsupportedClaimUncertaintyCoveragePercent": 100.0,
                "criticalManualAuditErrorCount": 0,
                "taskPercent": {
                    "circuit_validation": 20.0,
                    "firmware_generation": 20.0,
                    "domain_chat": 14.0,
                    "wiring": 10.0,
                    "pin_routing": 4.0,
                    "structured_output": 8.0,
                    "compiler_repair": 8.0,
                    "refusal": 8.0,
                    "uncertainty": 8.0,
                },
            }
        )
    return current


def _report(
    *,
    current: dict[str, object],
    fu015_status: str = "proposed",
    budget_approved: bool = False,
) -> dict[str, object]:
    return build_report(
        master_backlog=_master(),
        followup_backlog=_followups(fu015_status),
        policy=_policy(budget_approved=budget_approved),
        current=current,
        generated_on="2026-08-31",
        source_sha256={"policy": "source"},
    )


def test_current_corpus_records_stage_one_gap_without_count_inflation() -> None:
    report = _report(current=_current())

    assert report["decision"] == "await-immutable-retention-and-owner-budget"
    assert report["currentApprovedStage"] is None
    assert report["nextStage"] == "edge-stage-1-1m"
    assert report["nextStageTokenCoveragePercent"] == 15.569
    assert report["nextStageTokenGap"] == 844_310
    assert report["completionClaimed"] is False
    assert report["corpusRecordsGeneratedByEvaluator"] == 0
    validate_report(report)


def test_token_threshold_alone_cannot_approve_a_stage() -> None:
    report = _report(current=_current(unique_tokens=1_000_000))

    first_stage = report["stages"][0]
    assert first_stage["metrics"]["uniqueTrainingPredictedTokens"]["passed"] is True
    assert first_stage["metrics"]["exactComponentVariants"]["passed"] is False
    assert first_stage["commonGates"]["semanticDeduplicationEnforced"] is False
    assert first_stage["readyForThreeSeedLearningCurve"] is False
    validate_report(report)


def test_all_stage_one_gates_only_authorize_three_seed_learning_curve() -> None:
    current = _current(unique_tokens=1_000_000, quality_ready=True)
    report = _report(
        current=current,
        fu015_status="done",
        budget_approved=True,
    )

    assert report["decision"] == "edge-stage-1-1m-ready-for-three-seed-learning-curve"
    assert report["currentApprovedStage"] == "edge-stage-1-1m"
    assert report["stages"][0]["readyForThreeSeedLearningCurve"] is True
    assert report["stages"][1]["readyForThreeSeedLearningCurve"] is False
    assert report["completionClaimed"] is False
    assert report["modelRunsExecuted"] == 0
    validate_report(report)


def test_tampered_readiness_receipt_fails_closed() -> None:
    report = _report(current=_current())
    tampered = deepcopy(report)
    tampered["completionClaimed"] = True

    with pytest.raises(CorpusExpansionReadinessError, match="checksum"):
        validate_report(tampered)
