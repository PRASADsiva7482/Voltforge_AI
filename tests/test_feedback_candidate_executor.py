from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from feedback_governance import FeedbackGovernanceError, FeedbackStore
from feedback_training import FeedbackCandidateExecutorError, inspect_scheduled_candidate
from tools.evaluate_feedback_candidate_executor_readiness import (
    FeedbackExecutorReadinessError,
    POLICY_PATH,
    build_report,
    validate_report,
)


def _master() -> dict[str, object]:
    return {"status": "completed", "items": [{"id": "VFAI-035", "status": "done"}]}


def _followups() -> dict[str, object]:
    return {"items": [{"id": "VFAI-FU-009", "status": "accepted_for_later"}]}


def _policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _report(
    *,
    context_ready: bool = False,
    base_ready: bool = False,
    implementation_ready: bool = True,
    run_assigned: bool = False,
    database_assigned: bool = False,
    adapter_assigned: bool = False,
    result_assigned: bool = False,
) -> dict[str, object]:
    return build_report(
        master_backlog=_master(),
        followup_backlog=_followups(),
        policy=_policy(),
        context={
            "receiptValid": True,
            "contextCapableBaseRevisionReady": context_ready,
            "decision": "ready" if context_ready else "await-governed-corpus-stage",
        },
        registry={"activeStableContextCapableBaseReady": base_ready},
        implementation={"preAllocationAdmissionImplemented": implementation_ready},
        inputs={
            "scheduledRunManifestAssigned": run_assigned,
            "feedbackDatabaseAssigned": database_assigned,
            "trainingAdapterAssigned": adapter_assigned,
            "candidateResultReceiptAssigned": result_assigned,
        },
        generated_on="2026-08-31",
        source_sha256={"policy": "source"},
    )


def _approved_feedback_run(tmp_path) -> tuple[FeedbackStore, dict[str, object], object]:
    database = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(database, artifact_directory=tmp_path / "artifacts")
    pending = store.submit(
        user_id="user-feedback-executor",
        project_id="project-feedback-executor",
        project_revision="client:feedback-executor",
        request_id="request-feedback-executor",
        response_record_id="vf-task-v1-0123456789abcdef01234567",
        artifact_id="vfdlm-g1-edge-v1.0.0-test",
        registry_revision=7,
        feedback_kind="incorrect",
        rating=2,
        evidence_approved=True,
        training_consent=True,
        evidence="A reviewed relay answer omitted the isolation boundary.",
        expected_behavior="Explain the isolated driver boundary.",
    )
    store.review(
        pending["feedbackId"],
        decision="approve-heldout",
        reviewer_id="reviewer-feedback-executor",
    )
    store.approve_training(
        pending["feedbackId"],
        training_example={
            "message": "How should an AVR schedule bounded sensor work?",
            "expectedBehavior": "Use a bounded cooperative schedule and preserve deadlines.",
        },
        reviewer_id="reviewer-feedback-executor",
        expected_version=2,
    )
    scheduled = store.schedule_retraining(
        [pending["feedbackId"]],
        base_artifact_id="vfdlm-g1-edge-v1.0.0-test",
        base_registry_revision=7,
        output_directory=tmp_path / "runs",
    )
    run_path = tmp_path / "runs" / f"{scheduled['runId']}.json"
    return store, json.loads(run_path.read_text(encoding="utf-8")), run_path


def test_current_context_dependency_blocks_before_database_or_run_access(tmp_path) -> None:
    missing_database = tmp_path / "must-not-be-created.sqlite3"

    with pytest.raises(FeedbackCandidateExecutorError) as raised:
        inspect_scheduled_candidate(tmp_path / "missing-run.json", missing_database)

    assert raised.value.code == "FEEDBACK_EXECUTOR_CONTEXT_BASE_NOT_READY"
    assert missing_database.exists() is False


def test_feedback_store_revalidates_consent_lineage_and_all_heldout_read_only(tmp_path) -> None:
    store, run, run_path = _approved_feedback_run(tmp_path)
    database = tmp_path / "feedback.sqlite3"

    admitted = store.verify_execution_admission(run, run_path=run_path)

    assert admitted["consentAndReviewReverified"] is True
    assert admitted["currentHeldoutLeakageRejected"] is True
    assert admitted["stateMutated"] is False
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE feedback_records SET training_consent = 0")
        connection.commit()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with pytest.raises(
        FeedbackGovernanceError, match="consent, review state, or base lineage"
    ):
        store.verify_execution_admission(run, run_path=run_path)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_context_revision_is_the_first_readiness_gate() -> None:
    report = _report()

    assert report["decision"] == "await-context-capable-base-revision"
    assert report["trainingAllocationsExecutedByEvaluator"] == 0
    assert report["liveStateMutated"] is False
    validate_report(report)


def test_active_stable_base_is_required_after_context_revision() -> None:
    report = _report(context_ready=True)

    assert report["decision"] == "await-active-stable-base-artifact"
    validate_report(report)


def test_governed_run_database_and_adapter_are_sequential_gates() -> None:
    assert _report(context_ready=True, base_ready=True)["decision"] == "await-governed-feedback-run"
    assert (
        _report(context_ready=True, base_ready=True, run_assigned=True)["decision"]
        == "await-feedback-database-binding"
    )
    assert (
        _report(
            context_ready=True,
            base_ready=True,
            run_assigned=True,
            database_assigned=True,
        )["decision"]
        == "implement-context-training-adapter"
    )


def test_complete_external_evidence_advances_only_to_release_review() -> None:
    report = _report(
        context_ready=True,
        base_ready=True,
        run_assigned=True,
        database_assigned=True,
        adapter_assigned=True,
        result_assigned=True,
    )

    assert report["decision"] == "candidate-results-ready-for-independent-release-review"
    assert report["completionClaimed"] is False
    assert report["releaseOrActivationApproved"] is False
    validate_report(report)


def test_policy_matches_vfai034_release_gates_and_prohibits_mutation() -> None:
    policy = _policy()

    assert policy["scheduledRun"]["requiredFormat"] == "vfai034-retraining-run-v1"
    assert policy["executionAdmission"]["failedValidationMayMutateState"] is False
    assert policy["isolation"]["trainingAdapterPath"] is None
    assert "live-runtime-weight-mutation" in policy["forbidden"]


def test_tampered_readiness_receipt_fails_closed() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["completionClaimed"] = True

    with pytest.raises(FeedbackExecutorReadinessError, match="checksum"):
        validate_report(tampered)
