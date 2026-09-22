from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.evaluate_accelerator_inference_readiness import (
    AcceleratorReadinessError,
    POLICY_PATH,
    build_report,
    validate_report,
)


def _master() -> dict[str, object]:
    return {"status": "completed", "items": [{"id": "VFAI-035", "status": "done"}]}


def _followups() -> dict[str, object]:
    return {"items": [{"id": "VFAI-FU-007", "status": "accepted_for_later"}]}


def _policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _capabilities(*, cuda: bool = False, mps: bool = False) -> dict[str, object]:
    return {
        "platform": "test-platform",
        "machine": "arm64" if mps else "AMD64",
        "processor": "test-processor",
        "pythonVersion": "3.12.13",
        "torchVersion": "2.8.0+cpu",
        "torchCudaBuild": "12.8" if cuda else None,
        "cudaAvailable": cuda,
        "cudaDeviceCount": 1 if cuda else 0,
        "cudaDevices": [{}] if cuda else [],
        "mpsBuilt": mps,
        "mpsAvailable": mps,
        "appleSiliconHost": mps,
        "nvidiaSmi": {"available": cuda, "querySucceeded": cuda, "devices": []},
        "serialNumbersStored": False,
        "userOrHostNamesStored": False,
    }


def _historical_harness() -> dict[str, object]:
    return {
        "path": "gen1_optimization/benchmark.py",
        "classification": "historical-cpu-only-evidence",
        "cpuBoundMarkers": {"checkpointLoadsOnCpu": True},
        "deviceAware": False,
        "mayBeRewrittenInPlace": False,
    }


def _accelerator_harness(ready: bool) -> dict[str, object]:
    return {
        "path": "accelerator.py" if ready else None,
        "assigned": ready,
        "exists": ready,
        "deviceAwareContractVerified": ready,
        "sha256": "harness" if ready else None,
    }


def _artifact() -> dict[str, object]:
    return {
        "artifactId": "vfdlm-g1-edge-v0.1.2-optimized",
        "manifestSha256": "manifest",
        "cpuBenchmarkReportSha256": "cpu",
        "optimizationPolicySha256": "optimization",
        "gates": {"all": True},
        "ready": True,
    }


def _tier_results(*, accepted: set[str] | None = None) -> list[dict[str, object]]:
    accepted = accepted or set()
    return [
        {
            "tierId": tier,
            "path": f"{tier}.json" if tier in accepted else None,
            "assigned": tier in accepted,
            "valid": tier in accepted,
            "allAcceptanceGatesPassed": tier in accepted,
            "reportSha256": f"{tier}-report" if tier in accepted else None,
        }
        for tier in ("cuda", "mps")
    ]


def _report(
    *,
    cuda: bool = False,
    mps: bool = False,
    harness_ready: bool = False,
    accepted: set[str] | None = None,
) -> dict[str, object]:
    return build_report(
        master_backlog=_master(),
        followup_backlog=_followups(),
        policy=_policy(),
        capabilities=_capabilities(cuda=cuda, mps=mps),
        historical_harness=_historical_harness(),
        accelerator_harness=_accelerator_harness(harness_ready),
        artifact_evidence=_artifact(),
        tier_results=_tier_results(accepted=accepted),
        generated_on="2026-08-31",
        source_sha256={"policy": "source"},
    )


def test_cpu_only_host_records_both_accelerator_tiers_as_unmeasured() -> None:
    report = _report()

    assert report["decision"] == "await-owned-accelerator-hardware"
    assert report["gates"]["cudaVisible"] is False
    assert report["gates"]["mpsVisible"] is False
    assert report["gates"]["acceptedAcceleratorTiers"] == []
    assert report["completionClaimed"] is False
    assert report["benchmarkMeasurementsExecuted"] == 0
    validate_report(report)


def test_visible_cuda_cannot_reuse_cpu_bound_historical_harness() -> None:
    report = _report(cuda=True)

    assert report["decision"] == "implement-device-aware-accelerator-harness"
    assert report["historicalHarness"]["deviceAware"] is False
    assert report["gates"]["deviceAwareAcceleratorHarnessReady"] is False
    validate_report(report)


def test_device_aware_harness_advances_only_to_owned_matrix_execution() -> None:
    report = _report(cuda=True, harness_ready=True)

    assert report["decision"] == "ready-to-run-owned-accelerator-matrix"
    assert report["tiers"][0]["accepted"] is False
    assert report["releaseOrActivationApproved"] is False
    validate_report(report)


def test_both_external_tier_receipts_only_advance_to_package_review() -> None:
    report = _report(
        cuda=True,
        mps=True,
        harness_ready=True,
        accepted={"cuda", "mps"},
    )

    assert report["decision"] == "accelerator-results-ready-for-independent-package-review"
    assert report["gates"]["bothAcceleratorTiersAccepted"] is True
    assert report["completionClaimed"] is False
    assert report["releaseOrActivationApproved"] is False
    validate_report(report)


def test_tampered_readiness_receipt_fails_closed() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["completionClaimed"] = True

    with pytest.raises(AcceleratorReadinessError, match="checksum"):
        validate_report(tampered)
