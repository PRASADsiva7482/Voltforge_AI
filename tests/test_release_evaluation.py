import json
import hashlib
from pathlib import Path

import pytest

from evaluation.leakage import (
    HeldOutRegistry,
    build_manifest_data,
    exclude_held_out_records,
    load_cases,
    protected_texts,
    scan_corpora,
    verify_frozen_suite,
)
from evaluation.release_gate import ADAPTERS, _run_case
from benchmarking.runtime_baseline import validate_schema_contract


AI_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_suite_has_two_cases_for_every_required_task() -> None:
    manifest = verify_frozen_suite()
    cases = load_cases()
    required_tasks = {
        "domain_chat",
        "circuit_validation",
        "wiring",
        "board_pins",
        "firmware_review",
        "firmware_generation",
        "compiler_repair",
        "simulation_interpretation",
        "search_grounding",
        "memory_isolation",
        "refusal",
        "malformed_input",
        "adversarial_safety",
    }
    task_counts = {task: sum(case["task"] == task for case in cases) for task in required_tasks}

    assert manifest["freezeStatus"] == "locked"
    assert manifest["caseCount"] == 26
    assert set(case["task"] for case in cases) == required_tasks
    assert task_counts == {task: 2 for task in required_tasks}
    assert manifest == build_manifest_data()
    schema_path = AI_ROOT / "evaluation" / "evaluation-case.schema.json"
    for case in cases:
        validate_schema_contract(case, schema_path)


def test_each_metric_has_governance_and_owned_cases() -> None:
    policy = json.loads((AI_ROOT / "evaluation" / "metrics.v1.json").read_text(encoding="utf-8"))
    cases = load_cases()
    metric_ids = {metric["id"] for metric in policy["metrics"]}

    assert len(policy["metrics"]) == 13
    assert all(metric["owner"] for metric in policy["metrics"])
    assert all(metric["scoringMethod"] for metric in policy["metrics"])
    assert all(0 < metric["minimumScore"] <= 1 for metric in policy["metrics"])
    assert all(metric["criticalFailureRule"] for metric in policy["metrics"])
    assert {case["metricId"] for case in cases} == metric_ids


def test_exact_and_semantic_duplicates_are_excluded() -> None:
    cases = load_cases()
    registry = HeldOutRegistry(cases, semantic_threshold=0.86)
    protected = protected_texts(cases[0])[0]
    semantic_variant = protected + " Please answer carefully."

    exact = registry.match_text(protected)
    semantic = registry.match_text(semantic_variant)
    accepted, rejected = exclude_held_out_records(
        [{"prompt": "A separate approved training example."}, {"prompt": protected}], registry
    )

    assert exact is not None and exact.match_type == "exact"
    assert semantic is not None and semantic.match_type == "semantic"
    assert accepted == [{"prompt": "A separate approved training example."}]
    assert rejected[0]["caseId"] == cases[0]["id"]


def test_current_training_and_retrieval_corpora_have_no_held_out_collision() -> None:
    report = scan_corpora()
    assert report["status"] == "pass"
    assert report["exactAndSemanticCollisions"] == 0
    assert report["recordsScanned"] == 17_814
    assert report["filesScanned"] == 12


def test_fallback_and_unsupported_results_receive_zero_credit(monkeypatch) -> None:
    monkeypatch.setitem(
        ADAPTERS,
        "test-fallback",
        lambda _input: {
            "status": "completed",
            "executionClass": "test",
            "fallbackUsed": True,
            "output": {"reply": "expected"},
        },
    )
    fallback = _run_case(
        {
            "id": "test-fallback",
            "task": "domain_chat",
            "metricId": "domain_chat_correctness",
            "adapter": "test-fallback",
            "critical": False,
            "input": {},
            "assertions": [{"type": "path_text_contains", "path": "reply", "value": "expected"}],
        }
    )
    unsupported = _run_case(
        {
            "id": "test-unsupported",
            "task": "compiler_repair",
            "metricId": "compiler_repair_correctness",
            "adapter": "compiler_repair",
            "critical": False,
            "input": {},
            "assertions": [{"type": "path_not_empty", "path": "$"}],
        }
    )

    assert fallback["rawAssertionScore"] == 1.0
    assert fallback["creditedScore"] == 0.0
    assert fallback["passed"] is False
    assert unsupported["status"] == "unsupported"
    assert unsupported["creditedScore"] == 0.0


def test_checked_baseline_is_truthful_and_release_blocked() -> None:
    baseline = json.loads(
        (AI_ROOT / "evaluation" / "reports" / "current-system-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = verify_frozen_suite()

    assert baseline["suite"]["sha256"] == manifest["suiteSha256"]
    assert baseline["suite"]["evaluatorSha256"] == hashlib.sha256(
        (AI_ROOT / "evaluation" / "release_gate.py").read_bytes()
    ).hexdigest()
    assert baseline["target"]["networkAccessAllowed"] is False
    assert baseline["target"]["neuralModelReady"] is False
    assert baseline["target"]["missingNeuralMetricsReceiveCredit"] is False
    assert baseline["summary"]["releaseDecision"] == "blocked"
    assert baseline["summary"]["overallScore"] == pytest.approx(0.673077)
    assert baseline["summary"]["passedCases"] == 17
    assert baseline["summary"]["fallbackCases"] == 6
    assert baseline["summary"]["unsupportedCases"] == 2
    assert baseline["summary"]["errorCases"] == 0
    assert baseline["leakage"]["exactAndSemanticCollisions"] == 0
    validate_schema_contract(
        baseline, AI_ROOT / "evaluation" / "evaluation-report.schema.json"
    )


def test_every_training_and_runtime_corpus_path_uses_or_checks_exclusion_gate() -> None:
    paths = [
        AI_ROOT / "model" / "generate_dataset.py",
        AI_ROOT / "model" / "generate_domain_corpus.py",
        AI_ROOT / "model" / "train_chunks.py",
        AI_ROOT / "engine" / "reasoning.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "held_out" in source.lower(), path
        assert "evaluation.leakage" in source, path


def test_legacy_success_banner_is_removed_from_model_evaluator() -> None:
    source = (AI_ROOT / "model" / "evaluate.py").read_text(encoding="utf-8")
    assert "All checks passed" not in source
    assert "evaluation.release_gate" in source
