from __future__ import annotations

import pytest

from feedback_governance import FeedbackGovernanceError, FeedbackStore, verify_retraining_run


def _submit(store: FeedbackStore, *, request_id: str, training_consent: bool = True) -> dict[str, object]:
    return store.submit(
        user_id="user-feedback-test",
        project_id="project-feedback-test",
        project_revision="client:revision-feedback-test",
        request_id=request_id,
        response_record_id="vf-task-v1-0123456789abcdef01234567",
        artifact_id="vfdlm-g1-edge-v1.0.0-test",
        registry_revision=7,
        feedback_kind="incorrect",
        rating=2,
        evidence="A reviewed relay explanation omitted the isolation requirement.",
        expected_behavior="Explain the isolation requirement and the safe driver boundary.",
        evidence_approved=True,
        training_consent=training_consent,
    )


def test_feedback_is_project_scoped_bounded_and_content_safe(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3", artifact_directory=tmp_path / "artifacts")
    result = _submit(store, request_id="request-feedback-scope-1", training_consent=False)
    assert result["state"] == "pending-review"
    assert result["rawContentStored"] is False
    assert result["trainingEligible"] is False
    database_bytes = (tmp_path / "feedback.sqlite3").read_bytes()
    assert b"user-feedback-test" not in database_bytes
    assert b"project-feedback-test" not in database_bytes

    with pytest.raises(FeedbackGovernanceError, match="approval"):
        store.submit(
            user_id="user-feedback-test",
            project_id="project-feedback-test",
            project_revision="client:revision-feedback-test",
            request_id="request-feedback-scope-2",
            response_record_id="vf-task-v1-0123456789abcdef01234567",
            artifact_id="vfdlm-g1-edge-v1.0.0-test",
            registry_revision=7,
            feedback_kind="useful",
            rating=5,
            evidence="A bounded useful observation.",
            expected_behavior=None,
            evidence_approved=False,
            training_consent=False,
        )


def test_heldout_precedes_training_and_training_requires_consent(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3", artifact_directory=tmp_path / "artifacts")
    no_consent = _submit(store, request_id="request-feedback-no-consent", training_consent=False)
    heldout = store.review(no_consent["feedbackId"], decision="approve-heldout", reviewer_id="reviewer-feedback-v1")
    assert heldout["state"] == "held-out"
    with pytest.raises(FeedbackGovernanceError, match="consent"):
        store.approve_training(
            no_consent["feedbackId"],
            training_example={
                "message": "How should a logic output control an isolated relay input?",
                "expectedBehavior": "Use the approved isolated driver boundary.",
            },
            reviewer_id="reviewer-feedback-v1",
            expected_version=2,
        )

    pending = _submit(store, request_id="request-feedback-consented", training_consent=True)
    with pytest.raises(FeedbackGovernanceError, match="held-out"):
        store.approve_training(
            pending["feedbackId"],
            training_example={
                "message": "How should a timer interrupt be bounded on an AVR?",
                "expectedBehavior": "Use a bounded interrupt workload and measure the deadline.",
            },
            reviewer_id="reviewer-feedback-v1",
        )


def test_approved_feedback_becomes_heldout_then_nonleaking_retraining_run(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3", artifact_directory=tmp_path / "artifacts")
    pending = _submit(store, request_id="request-feedback-training")
    heldout = store.review(pending["feedbackId"], decision="approve-heldout", reviewer_id="reviewer-feedback-v1")
    approved = store.approve_training(
        pending["feedbackId"],
        training_example={
            "message": "What is a safe way to budget an AVR loop that reads a sensor?",
            "expectedBehavior": "Keep the loop bounded, schedule work cooperatively, and preserve the deadline.",
        },
        reviewer_id="reviewer-feedback-v1",
        expected_version=2,
    )
    assert approved["state"] == "training-approved"
    run = store.schedule_retraining(
        [pending["feedbackId"]],
        base_artifact_id="vfdlm-g1-edge-v1.0.0-test",
        base_registry_revision=7,
        seed=1234,
        hyperparameters={"learningRate": 0.0001, "epochs": 1},
        output_directory=tmp_path / "runs",
    )
    assert run["liveChatWeightMutation"] is False
    assert run["trainingRecordCount"] == 1
    run_path = tmp_path / "runs" / f"{run['runId']}.json"
    verified = verify_retraining_run(run_path)
    assert verified["verified"] is True
    assert verified["runSha256"] == run["runSha256"]
    assert heldout["heldoutSha256"]


def test_training_example_cannot_copy_heldout_regression(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3", artifact_directory=tmp_path / "artifacts")
    pending = _submit(store, request_id="request-feedback-leakage")
    store.review(pending["feedbackId"], decision="approve-heldout", reviewer_id="reviewer-feedback-v1")
    with pytest.raises(FeedbackGovernanceError, match="overlaps"):
        store.approve_training(
            pending["feedbackId"],
            training_example={
                "message": "Use this reviewed VoltForge regression evidence: A reviewed relay explanation omitted the isolation requirement.",
                "expectedBehavior": "Explain the isolation requirement and the safe driver boundary.",
            },
            reviewer_id="reviewer-feedback-v1",
            expected_version=2,
        )
