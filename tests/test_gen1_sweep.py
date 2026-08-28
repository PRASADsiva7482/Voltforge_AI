from __future__ import annotations

from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
import json

import pytest

try:
    installed_torch = version("torch")
except PackageNotFoundError:
    pytest.skip("Gen1 pinned PyTorch runtime is not installed", allow_module_level=True)
if installed_torch.partition("+")[0] != "2.8.0":
    pytest.skip(
        f"Gen1 sweep tests require pinned PyTorch 2.8.0, found {installed_torch}",
        allow_module_level=True,
    )

from gen1_sweep import (  # noqa: E402
    SweepContractError,
    check_scorecard,
    load_sweep_plan,
    pareto_frontier,
    select_candidate,
)
from gen1_sweep.sweep import (  # noqa: E402
    DEFAULT_REPORT_PATH,
    DEFAULT_SELECTED_CONFIG_PATH,
)
from gen1_training import load_approved_corpus  # noqa: E402
from model.gen1 import Gen1Config  # noqa: E402


def measured_candidate(
    candidate_id: str,
    *,
    profile: str = "edge",
    loss: float = 2.0,
    accuracy: float = 0.25,
    wall: float = 10.0,
    first: float = 10.0,
    decode: float = 20.0,
    memory: float = 500.0,
    artifact: int = 10_000,
    cpu: float = 20.0,
    parameters: int = 10_000_000,
) -> dict[str, object]:
    return {
        "candidateId": candidate_id,
        "deploymentProfile": profile,
        "parameterCount": parameters,
        "status": "completed",
        "quality": {
            "final": {"nextTokenLoss": loss, "top1TokenAccuracy": accuracy}
        },
        "training": {
            "wallSeconds": wall,
            "processCpuSeconds": cpu,
            "peakMemory": {"peakWorkingSetMiB": memory},
        },
        "inference": {
            "firstTokenLatencyMs": {"median": first},
            "cachedDecodeTokensPerSecond": {"median": decode},
        },
        "artifact": {"checkpointBytes": artifact},
    }


def test_plan_covers_required_dimensions_and_parameter_profiles() -> None:
    plan = load_sweep_plan()
    candidates = plan["candidates"]
    assert len(candidates) == 6
    assert plan["controls"]["expectedPredictedTokens"] == 8_128
    assert plan["controls"]["packingBlockSize"] == 128
    dimensions = {value for item in candidates for value in item["dimensions"]}
    assert {
        "depth",
        "width",
        "query-heads",
        "kv-heads",
        "ffn-ratio",
        "context",
        "dropout",
        "weight-decay",
    } <= dimensions
    for candidate in candidates:
        count = Gen1Config.from_dict(candidate["model"]).parameter_count()
        lower, upper = (
            (8_000_000, 25_000_000)
            if candidate["deploymentProfile"] == "edge"
            else (25_000_000, 60_000_000)
        )
        assert lower <= count <= upper
    assert plan["nonExecutableControls"][0]["dimension"] == "vocabulary"
    assert plan["nonExecutableControls"][0]["status"] == "rejected-before-training"


def test_controlled_packing_has_one_dataset_identity_across_context_sizes() -> None:
    base = Gen1Config()
    context_128 = replace(base, max_sequence_length=128)
    context_256 = replace(base, max_sequence_length=256)
    first = load_approved_corpus(context_128, packing_block_size=128)
    second = load_approved_corpus(context_256, packing_block_size=128)
    assert first.fingerprint == second.fingerprint
    assert first.train.stream_sha256 == second.train.stream_sha256
    assert first.validation.stream_sha256 == second.validation.stream_sha256


def test_pareto_frontier_removes_only_measurably_dominated_candidate() -> None:
    efficient = measured_candidate("efficient")
    dominated = measured_candidate(
        "dominated",
        loss=2.1,
        accuracy=0.20,
        wall=11.0,
        first=11.0,
        decode=19.0,
        memory=510.0,
        artifact=10_100,
        cpu=21.0,
    )
    quality = measured_candidate("quality", loss=1.9, wall=14.0)
    assert pareto_frontier([efficient, dominated, quality]) == ["efficient", "quality"]


def test_selection_does_not_reward_parameter_count() -> None:
    plan = load_sweep_plan()
    smaller_slower = measured_candidate(
        "smaller-slower", parameters=9_000_000, wall=11.0, first=11.0, cpu=21.0
    )
    larger_efficient = measured_candidate(
        "larger-efficient", parameters=20_000_000
    )
    decision = select_candidate(
        [smaller_slower, larger_efficient], plan["selectionPolicy"]
    )
    assert decision["selectedCandidateId"] == "larger-efficient"
    assert decision["parameterCountRewarded"] is False


def test_core_requires_material_proxy_quality_improvement() -> None:
    plan = load_sweep_plan()
    edge = measured_candidate("edge", loss=2.0)
    core = measured_candidate(
        "core",
        profile="core",
        loss=1.99,
        wall=20.0,
        first=20.0,
        memory=700.0,
        artifact=20_000,
        cpu=40.0,
        parameters=30_000_000,
    )
    decision = select_candidate([edge, core], plan["selectionPolicy"])
    assert decision["selectedCandidateId"] == "edge"
    assert decision["coreMaterialThresholdMet"] is False


def test_committed_scorecard_and_selected_config_are_self_verifying(tmp_path) -> None:
    result = check_scorecard()
    assert result["decision"] == "pass"
    assert result["candidateCount"] == 6

    tampered = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    tampered["selection"]["selectedCandidateId"] = "tampered"
    tampered_path = tmp_path / "tampered-report.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(SweepContractError, match="checksum mismatch"):
        check_scorecard(
            report_path=tampered_path,
            selected_config_path=DEFAULT_SELECTED_CONFIG_PATH,
        )
