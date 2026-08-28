from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json

import pytest

try:
    installed_torch = version("torch")
except PackageNotFoundError:
    pytest.skip("Gen1 pinned PyTorch runtime is not installed", allow_module_level=True)
if installed_torch.partition("+")[0] != "2.8.0":
    pytest.skip(
        f"Gen1 decision tests require pinned PyTorch 2.8.0, found {installed_torch}",
        allow_module_level=True,
    )

from gen1_decision import (  # noqa: E402
    ScaleDecisionContractError,
    check_scale_decision,
    evaluate_larger_profile_eligibility,
    load_scale_policy,
)
from gen1_decision.decision import DEFAULT_REPORT_PATH  # noqa: E402


def eligible_observation(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "dataCoveragePercent": 100.0,
        "independentSeeds": 3,
        "criticalDomainCompositeImprovementPoints": 2.0,
        "confidenceLowerBoundImprovementPoints": 0.1,
        "bootstrapProxyLossImprovementPercent": 1.0,
        "criticalMetricRegressions": 0,
        "smallerProfileAtDataTarget": True,
        "smallerProfileOptimizationPlateau": True,
        "targetHardwareMeasured": True,
        "candidateGlobalMetricsPassing": 13,
        "candidateCriticalCaseFailures": 0,
        "generationRuntimeApproved": True,
    }
    values.update(overrides)
    return values


def test_policy_binds_profiles_release_gates_and_owned_evidence() -> None:
    policy = load_scale_policy()
    assert [item["name"] for item in policy["profiles"]] == ["edge", "core", "server"]
    assert [item["minimumApprovedTrainingTokens"] for item in policy["profiles"]] == [
        100_000_000,
        500_000_000,
        1_200_000_000,
    ]
    assert policy["releaseGates"]["requiredGlobalMetricsPassing"] == 13
    assert policy["largerProfileGates"]["minimumIndependentSeeds"] == 3
    assert policy["fallbackPolicy"]["selectedWhenAnyReleaseGateFails"] == (
        "no-neural-release"
    )


def test_larger_profile_requires_data_replication_critical_gain_and_runtime() -> None:
    gates = load_scale_policy()["largerProfileGates"]
    result = evaluate_larger_profile_eligibility(
        eligible_observation(
            dataCoveragePercent=0.014,
            independentSeeds=1,
            criticalDomainCompositeImprovementPoints=0.0,
            confidenceLowerBoundImprovementPoints=0.0,
            generationRuntimeApproved=False,
        ),
        gates,
    )
    assert result["eligible"] is False
    assert {
        "INSUFFICIENT_APPROVED_DATA",
        "INSUFFICIENT_INDEPENDENT_SEEDS",
        "CRITICAL_DOMAIN_GAIN_NOT_MATERIAL",
        "GAIN_NOT_ABOVE_NOISE",
        "GENERATION_RUNTIME_UNAPPROVED",
    } <= set(result["failedGates"])


def test_larger_profile_can_pass_only_when_every_frozen_gate_passes() -> None:
    gates = load_scale_policy()["largerProfileGates"]
    result = evaluate_larger_profile_eligibility(eligible_observation(), gates)
    assert result["eligible"] is True
    assert result["failedGates"] == []


def test_committed_decision_selects_no_release_and_classifies_data_bottleneck() -> None:
    result = check_scale_decision()
    assert result == {
        "decision": "pass",
        "reportId": "vfai015-gen1-scale-decision-v1",
        "selectedOption": "no-neural-release",
        "primaryBottleneck": "data-limited",
        "parameterIncreaseApproved": False,
        "activeArtifactAtDecision": None,
        "nextGate": "VFAI-016",
    }
    report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    assert report["hypothesisAssessment"]["dataLimited"]["verdict"] == (
        "supported-primary-bottleneck"
    )
    assert report["hypothesisAssessment"]["architectureLimited"]["verdict"] == (
        "not-supported"
    )
    assert report["hypothesisAssessment"]["capacityLimited"]["verdict"].startswith(
        "not-supported"
    )
    assert report["safeOperatingBoundary"]["experimentalCheckpointMayServeUsers"] is False


def test_decision_report_is_tamper_evident(tmp_path) -> None:
    report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    report["decision"]["parameterIncreaseApproved"] = True
    path = tmp_path / "tampered-decision.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ScaleDecisionContractError, match="checksum mismatch"):
        check_scale_decision(report_path=path)
