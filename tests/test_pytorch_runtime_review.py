from __future__ import annotations

import json

import pytest

from tools.evaluate_pytorch_runtime_review import build_report, verify


def _probe(version: str, passed: int) -> dict[str, object]:
    attempted = 20
    return {
        "pythonVersion": "3.12.13",
        "implementation": "CPython",
        "architecture": "AMD64",
        "torchDistributionVersion": version,
        "interpreter": ".toolchains/fixture/Scripts/python.exe",
        "cleanProcessImports": {
            "attempted": attempted,
            "passed": passed,
            "failed": attempted - passed,
            "observedImportedVersions": [version] if passed else [],
            "failureCounts": (
                {} if passed else {"WINERROR_1114_C10_DLL_INITIALIZATION_FAILED": 20}
            ),
        },
    }


def test_failed_candidate_import_gate_retains_pinned_runtime() -> None:
    report = build_report(
        baseline=_probe("2.8.0+cpu", 20),
        candidate=_probe("2.13.0+cpu", 0),
        upstream_issue_status="open",
        generated_on="2026-08-31",
        master_backlog_complete=True,
        host_operating_system="Windows",
        minimum_patched_candidate=_probe("2.10.0+cpu", 0),
    )

    assert report["decision"] == "retain-pinned-runtime-with-serving-blocked"
    assert report["upgradeApplied"] is False
    assert report["gates"]["baselineFreshImportGatePassed"] is True
    assert report["gates"]["candidateFreshImportGatePassed"] is False
    assert (
        report["gates"]["candidateArchitectureAndCheckpointSuite"]
        == "not-run-candidate-import-gate-failed"
    )
    assert report["security"]["baselineAffected"] is True
    assert report["security"]["minimumPatchedCandidateImportGatePassed"] is False
    assert report["security"]["checkpointBackedServingContainmentImplemented"] is True


def test_candidate_only_advances_when_upstream_and_import_gates_pass() -> None:
    open_issue = build_report(
        baseline=_probe("2.8.0+cpu", 20),
        candidate=_probe("2.13.0+cpu", 20),
        upstream_issue_status="open",
        generated_on="2026-08-31",
        master_backlog_complete=True,
        host_operating_system="Windows",
    )
    resolved_issue = build_report(
        baseline=_probe("2.8.0+cpu", 20),
        candidate=_probe("2.13.0+cpu", 20),
        upstream_issue_status="resolved",
        generated_on="2026-08-31",
        master_backlog_complete=True,
        host_operating_system="Windows",
    )

    assert open_issue["decision"] == "retain-pinned-runtime-with-serving-blocked"
    assert resolved_issue["decision"] == "candidate-ready-for-downstream-evaluation"
    assert (
        resolved_issue["gates"]["candidateFullAiSuite"]
        == "pending-candidate-import-gate-passed"
    )


def test_verify_rejects_a_tampered_review_receipt(tmp_path) -> None:
    report = build_report(
        baseline=_probe("2.8.0+cpu", 20),
        candidate=_probe("2.13.0+cpu", 0),
        upstream_issue_status="open",
        generated_on="2026-08-31",
        master_backlog_complete=True,
        host_operating_system="Windows",
    )
    path = tmp_path / "review.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert verify(path)["decision"] == "retain-pinned-runtime-with-serving-blocked"

    report["decision"] = "candidate-ready-for-downstream-evaluation"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        verify(path)
