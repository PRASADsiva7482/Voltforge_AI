from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from gen1_optimization.benchmark import (
    build_candidate_matrix,
    build_policy,
    build_report,
    validate_report,
)
from model.gen1 import Gen1Config, VoltForgeGen1
from model.gen1.optimization import (
    InferenceOptimizationProfile,
    OptimizationProfileError,
    REFERENCE_PROFILE,
    VFAI017_SAFE_PROFILE,
    apply_model_optimizations,
    model_tensor_bytes,
    profiles_from_policy,
)
from model.registry_manager import canonical_json_bytes, sha256_bytes


def tiny_config() -> Gen1Config:
    return Gen1Config(
        vocab_size=32,
        max_sequence_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
        seed=18018,
    )


def test_profile_and_signed_policy_contract_fail_closed() -> None:
    selected = VFAI017_SAFE_PROFILE.as_dict()
    fallback = VFAI017_SAFE_PROFILE.as_dict()
    policy = {
        "schemaVersion": 1,
        "selectedProfileId": selected["profileId"],
        "fallbackProfileId": fallback["profileId"],
        "profiles": [selected],
    }

    parsed_selected, parsed_fallback = profiles_from_policy(policy)
    assert parsed_selected == VFAI017_SAFE_PROFILE
    assert parsed_fallback == VFAI017_SAFE_PROFILE

    malformed = deepcopy(selected)
    malformed["remoteDevice"] = True
    with pytest.raises(OptimizationProfileError) as raised:
        InferenceOptimizationProfile.from_mapping(malformed)
    assert raised.value.code == "MODEL_OPTIMIZATION_PROFILE_INVALID"

    unsafe_fallback = deepcopy(policy)
    unsafe_fallback["profiles"][0]["quantization"] = "int8-weight-only"
    with pytest.raises(OptimizationProfileError) as raised:
        profiles_from_policy(unsafe_fallback)
    assert raised.value.code == "MODEL_OPTIMIZATION_POLICY_INVALID"


def test_sdpa_matches_manual_attention_and_cache() -> None:
    model = VoltForgeGen1.for_training(tiny_config()).eval()
    tokens = torch.tensor([[2, 5, 8, 11], [2, 7, 9, 13]], dtype=torch.long)
    manual = model(tokens, use_cache=True)

    model.set_attention_backend("sdpa")
    optimized = model(tokens, use_cache=True)

    assert torch.allclose(manual.logits, optimized.logits, atol=1e-5, rtol=1e-5)
    assert torch.equal(manual.logits.argmax(dim=-1), optimized.logits.argmax(dim=-1))
    assert optimized.past_key_values is not None
    assert manual.past_key_values is not None
    for manual_layer, optimized_layer in zip(
        manual.past_key_values, optimized.past_key_values, strict=True
    ):
        assert torch.allclose(manual_layer[0], optimized_layer[0], atol=1e-5, rtol=1e-5)
        assert torch.allclose(manual_layer[1], optimized_layer[1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("quantization", ["int8-weight-only", "int4-weight-only"])
def test_weight_only_quantization_is_local_finite_and_smaller(quantization: str) -> None:
    model = VoltForgeGen1.for_training(tiny_config()).eval()
    before = model_tensor_bytes(model)
    profile = InferenceOptimizationProfile(
        profile_id=f"test-{quantization}",
        kv_cache_enabled=True,
        attention_backend="manual",
        torch_threads=1,
        mmap_weights=False,
        quantization=quantization,
        dynamic_batch_max_size=1,
    )

    result = apply_model_optimizations(model, profile, torch.device("cpu"))
    output = model(torch.tensor([[2, 4, 6, 8]], dtype=torch.long))

    assert result["quantizedLinearCount"] > 0
    assert result["residentTensorBytes"] < before
    assert torch.all(torch.isfinite(output.logits))


def test_mmap_checkpoint_is_output_equivalent(tmp_path: Path) -> None:
    original = VoltForgeGen1.for_training(tiny_config()).eval()
    original.save_checkpoint(tmp_path / "checkpoint")
    tokens = torch.tensor([[2, 5, 8, 3]], dtype=torch.long)

    regular = VoltForgeGen1.from_checkpoint(tmp_path / "checkpoint", mmap_weights=False)
    mapped = VoltForgeGen1.from_checkpoint(tmp_path / "checkpoint", mmap_weights=True)

    assert torch.equal(regular(tokens).logits, mapped(tokens).logits)
    assert torch.equal(original(tokens).logits, mapped(tokens).logits)


def _fake_measurements() -> list[dict[str, object]]:
    throughput = {
        "reference": 10.0,
        "kv-cache": 20.0,
        "sdpa": 25.0,
        "memory-map": 20.5,
        "threads-1": 17.0,
        "threads-2": 22.0,
        "threads-8": 19.0,
        "dynamic-batch-4": 40.0,
        "int8": 14.0,
        "int4": 8.0,
    }
    measurements = []
    for candidate in build_candidate_matrix():
        candidate_id = candidate["candidateId"]
        quantized = candidate_id in {"int8", "int4"}
        measurements.append(
            {
                **deepcopy(candidate),
                "status": "measured",
                "artifact": {"manifestSha256": "a" * 64},
                "quality": {
                    "nextTokenLoss": 6.0 if not quantized else 6.2,
                    "top1Accuracy": 0.02 if not quantized else 0.01,
                },
                "generation": {
                    "throughputTokensPerSecond": throughput[candidate_id],
                    "outputTokenIdsSha256": "b" * 64 if not quantized else "c" * 64,
                    "latencyMs": {"median": 1_000 / throughput[candidate_id]},
                    "firstTokenLatencyMs": {"median": 10.0},
                },
                "memory": {
                    "residentTensorBytes": 1000 if not quantized else 400,
                },
            }
        )
    return measurements


def test_report_keeps_reference_selects_measured_candidate_and_binds_policy() -> None:
    report = build_report(_fake_measurements())
    validate_report(report)
    policy = build_policy(report)

    assert report["selection"]["selectedCandidateId"] == "sdpa"
    assert report["selection"]["modelReleaseApproved"] is False
    assert report["portability"]["quantizedFormatsReleased"] == []
    assert policy["benchmarkReportSha256"] == report["reportSha256"]
    assert policy["fallbackProfileId"] == VFAI017_SAFE_PROFILE.profile_id
    assert json.dumps(report, allow_nan=False)


def test_candidate_matrix_retains_explicit_unoptimized_reference() -> None:
    matrix = build_candidate_matrix()
    assert matrix[0]["profile"] == REFERENCE_PROFILE.as_dict()
    assert {item["technique"] for item in matrix} == {
        "reference",
        "kv-cache",
        "attention-kernel",
        "memory-mapping",
        "thread-setting",
        "dynamic-batching",
        "quantization-int8",
        "quantization-int4",
    }


def test_checked_in_optimization_evidence_is_bound_and_privacy_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (root / "benchmarks/reports/gen1-inference-optimization-v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads(
        (root / "model/gen1/optimization-policy.v1.json").read_text(encoding="utf-8")
    )
    package = json.loads(
        (
            root
            / "model/registry/reports/vfdlm-g1-edge-v0.1.2-optimized-package.json"
        ).read_text(encoding="utf-8")
    )
    smoke = json.loads(
        (root / "evaluation/reports/gen1-optimization-smoke-v1.json").read_text(
            encoding="utf-8"
        )
    )

    validate_report(report)
    assert policy == build_policy(report)
    unsigned_package = dict(package)
    package_digest = unsigned_package.pop("reportSha256")
    unsigned_smoke = dict(smoke)
    smoke_digest = unsigned_smoke.pop("reportSha256")
    assert package_digest == sha256_bytes(canonical_json_bytes(unsigned_package))
    assert smoke_digest == sha256_bytes(canonical_json_bytes(unsigned_smoke))
    assert package["benchmarkReportSha256"] == report["reportSha256"]
    assert package["optimizationPolicySha256"] == policy["policySha256"]
    assert package["selectedProfileId"] == report["selection"]["selectedProfile"][
        "profileId"
    ]
    assert package["quantizedFormatReleased"] is False
    assert package["registry"]["activeArtifactId"] is None
    assert all(smoke["checks"].values())
    assert smoke["generation"]["rawPromptStored"] is False
    assert smoke["generation"]["rawOutputStored"] is False
