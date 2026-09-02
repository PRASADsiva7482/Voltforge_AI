from __future__ import annotations

import json

import pytest

from tools.evaluate_cuda_mixed_precision_resume import build_report, verify


TRAINER_SOURCE = """
MAXIMUM_OVERFLOW_RETRIES_PER_STEP = 8
mixedPrecisionOverflowRetries
self.stream.load_state_dict(stream_state)
_restore_rng_state(rng_state)
"""


def _capabilities(cuda: bool) -> dict[str, object]:
    return {
        "platform": "Windows-11",
        "machine": "AMD64",
        "pythonVersion": "3.12.13",
        "torchVersion": "2.8.0+cpu" if not cuda else "2.8.0+cu128",
        "torchCudaBuild": None if not cuda else "12.8",
        "cudaAvailable": cuda,
        "cudaDeviceCount": 1 if cuda else 0,
        "cpuCapability": "AVX2",
        "cpuAvx512Bf16": False,
        "cudaDevices": [] if not cuda else [{"index": 0, "bf16Supported": True}],
    }


def _smokes(passed: bool) -> list[dict[str, object]]:
    return [
        {"precision": "bfloat16", "executed": passed, "passed": passed},
        {"precision": "float16", "executed": passed, "passed": passed},
    ]


def test_cpu_only_host_records_unmeasured_cuda_gates_without_completion() -> None:
    report = build_report(
        capabilities=_capabilities(False),
        precision_smokes=_smokes(False),
        nvidia_smi={"available": False, "devices": []},
        generated_on="2026-08-31",
    )

    assert report["decision"] == "await-owned-cuda-hardware"
    assert report["completionClaimed"] is False
    assert report["gates"]["ownedCudaVisibleToPinnedRuntime"] is False
    assert report["gates"]["safePointOverflowRecoveryImplemented"] is True
    assert (
        report["gates"]["cudaResumeEquivalence"]
        == "not-run-owned-cuda-unavailable"
    )


def test_supported_cuda_smokes_only_advance_to_controlled_scorecard() -> None:
    report = build_report(
        capabilities=_capabilities(True),
        precision_smokes=_smokes(True),
        nvidia_smi={"available": True, "querySucceeded": True, "devices": [{}]},
        generated_on="2026-08-31",
        trainer_source=TRAINER_SOURCE,
    )

    assert report["decision"] == "ready-for-controlled-owned-cuda-scorecard"
    assert report["completionClaimed"] is False
    assert (
        report["gates"]["controlledFp32Bf16Fp16Scorecard"]
        == "pending-controlled-owned-cuda-runs"
    )


def test_verify_rejects_tampered_capability_receipt(tmp_path) -> None:
    report = build_report(
        capabilities=_capabilities(False),
        precision_smokes=_smokes(False),
        nvidia_smi={"available": False, "devices": []},
        generated_on="2026-08-31",
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert verify(path)["decision"] == "await-owned-cuda-hardware"

    report["completionClaimed"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        verify(path)
