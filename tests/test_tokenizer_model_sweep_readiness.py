from __future__ import annotations

import pytest

from tools.evaluate_tokenizer_model_sweep_readiness import (
    SweepReadinessError,
    build_report,
    validate_report,
)


EDGE_MINIMUM = 100_000_000


def _master() -> dict[str, object]:
    return {"status": "completed", "items": [{"id": "VFAI-035", "status": "done"}]}


def _followups(fu006_status: str = "proposed") -> dict[str, object]:
    return {
        "items": [
            {"id": "VFAI-FU-005", "status": "accepted_for_later"},
            {"id": "VFAI-FU-006", "status": fu006_status},
        ]
    }


def _policy() -> dict[str, object]:
    return {
        "policyId": "vfai-fu-005-joint-tokenizer-model-sweep-v1",
        "status": "readiness-contract-only",
        "sourceFollowup": "VFAI-FU-005",
        "dataDependency": {
            "minimumUniqueTrainingPredictedTokens": EDGE_MINIMUM,
        },
        "candidateAdmission": {
            "requiredApprovedCandidateCount": 3,
            "sameApprovedNormalizedRecordIds": True,
            "sameTrainingAndValidationSplitHashes": True,
            "sameRawUtf8TrainingAndValidationBytes": True,
        },
        "fairExposure": {
            "tokenBudgetEquivalenceAllowed": False,
            "sameRecordOrderAndSeedSchedule": True,
            "sameRawContentExposurePerSeed": True,
            "requiredAccounting": [f"metric-{index}" for index in range(10)],
        },
        "modelSweep": {
            "minimumSeeds": 3,
            "releaseActivationAllowed": False,
        },
        "selection": {
            "method": "measured-pareto-frontier",
            "requiresMaterialImprovementOverBaseline": True,
            "automaticPromotionAllowed": False,
        },
        "forbiddenInputsAndServices": [f"forbidden-{index}" for index in range(7)],
    }


def _corpus(unique_predictions: int) -> dict[str, object]:
    return {
        "datasetId": "approved-corpus",
        "approvalStatus": "approved",
        "trainingRecordCount": 227,
        "validationRecordCount": 23,
        "trainingCorpusSha256": "training",
        "validationCorpusSha256": "validation",
        "training": {"predictedTokenCount": unique_predictions},
        "tokenizer": {"version": "1.1.0", "artifactSha256": "tokenizer"},
    }


def _candidates(approved_count: int) -> list[dict[str, object]]:
    return [
        {
            "candidateId": f"candidate-{index}",
            "vocabSize": size,
            "approved": index < approved_count,
        }
        for index, size in enumerate((2048, 3072, 4096))
    ]


def _report(
    *,
    unique_predictions: int,
    fu006_status: str,
    approved_count: int,
) -> dict[str, object]:
    return build_report(
        master_backlog=_master(),
        followup_backlog=_followups(fu006_status),
        policy=_policy(),
        corpus_manifest=_corpus(unique_predictions),
        corpus_fingerprint="packed-corpus",
        candidates=_candidates(approved_count),
        generated_on="2026-08-31",
        source_sha256={"policy": "source"},
    )


def test_current_scale_records_exact_gap_without_claiming_a_sweep() -> None:
    report = _report(
        unique_predictions=155_690,
        fu006_status="proposed",
        approved_count=1,
    )

    assert report["decision"] == "await-governed-corpus-expansion"
    assert report["corpus"]["edgeCoveragePercent"] == 0.15569
    assert report["corpus"]["edgeTokenGap"] == 99_844_310
    assert report["gates"]["approvedTokenizerCandidates"] == 1
    assert report["gates"]["jointSweepExecutionAllowed"] is False
    assert report["completionClaimed"] is False
    assert report["modelSweepExecuted"] is False
    validate_report(report)


def test_edge_ready_corpus_advances_only_to_candidate_training() -> None:
    report = _report(
        unique_predictions=EDGE_MINIMUM,
        fu006_status="done",
        approved_count=1,
    )

    assert report["decision"] == "ready-to-train-tokenizer-candidates"
    assert report["gates"]["approvedTokenizerCandidateGate"] is False
    assert report["gates"]["jointSweepExecutionAllowed"] is False
    validate_report(report)


def test_all_readiness_gates_authorize_but_do_not_execute_the_sweep() -> None:
    report = _report(
        unique_predictions=EDGE_MINIMUM,
        fu006_status="complete",
        approved_count=3,
    )

    assert report["decision"] == "ready-for-joint-controlled-sweep"
    assert report["gates"]["jointSweepExecutionAllowed"] is True
    assert report["completionClaimed"] is False
    assert report["tokenizerCandidatesTrainedByEvaluator"] is False
    validate_report(report)


def test_tampered_readiness_receipt_fails_closed() -> None:
    report = _report(
        unique_predictions=155_690,
        fu006_status="proposed",
        approved_count=1,
    )
    report["completionClaimed"] = True

    with pytest.raises(SweepReadinessError, match="checksum"):
        validate_report(report)
