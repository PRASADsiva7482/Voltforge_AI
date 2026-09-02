"""Generate or verify the VFAI-FU-007 owned-accelerator readiness receipt.

This tool performs capability and evidence checks only. It does not load model
weights or governed data, benchmark an unavailable device, access the network,
or infer CUDA/MPS support from CPU measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "gen1_optimization/accelerator-policy.v1.json"
CPU_BENCHMARK_PATH = AI_ROOT / "benchmarks/reports/gen1-inference-optimization-v1.json"
OPTIMIZATION_POLICY_PATH = AI_ROOT / "model/gen1/optimization-policy.v1.json"
ARTIFACT_MANIFEST_PATH = (
    AI_ROOT / "model/registry/manifests/vfdlm-g1-edge-v0.1.2-optimized.json"
)
ARTIFACT_PACKAGE_REPORT_PATH = (
    AI_ROOT / "model/registry/reports/vfdlm-g1-edge-v0.1.2-optimized-package.json"
)
ARTIFACT_DIRECTORY = (
    AI_ROOT / "model/registry/artifacts/vfdlm-g1-edge-v0.1.2-optimized"
)
HISTORICAL_HARNESS_PATH = AI_ROOT / "gen1_optimization/benchmark.py"
DEFAULT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/accelerator-inference-readiness-v1.json"
)
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class AcceleratorReadinessError(RuntimeError):
    """The accelerator readiness evidence is invalid or stale."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceleratorReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcceleratorReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_workspace_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AcceleratorReadinessError(f"{label} path is missing")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise AcceleratorReadinessError(f"{label} path escapes the AI workspace") from exc
    return path


def _backlog_item(backlog: Mapping[str, Any], item_id: str) -> Mapping[str, Any]:
    items = backlog.get("items")
    if not isinstance(items, list):
        raise AcceleratorReadinessError("follow-up backlog items are invalid")
    matches = [item for item in items if isinstance(item, Mapping) and item.get("id") == item_id]
    if len(matches) != 1:
        raise AcceleratorReadinessError(f"expected exactly one {item_id} backlog item")
    return matches[0]


def _master_complete(master: Mapping[str, Any]) -> bool:
    items = master.get("items")
    return bool(
        master.get("status") in COMPLETE_STATUSES
        and isinstance(items, list)
        and items
        and all(
            isinstance(item, Mapping) and item.get("status") in COMPLETE_STATUSES
            for item in items
        )
    )


def _nvidia_smi_inventory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "querySucceeded": False, "devices": []}
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return {"available": True, "querySucceeded": False, "devices": []}
    devices = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            continue
        devices.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "computeCapability": values[2],
                "memoryMiB": int(values[3]),
                "driverVersion": values[4],
            }
        )
    return {"available": True, "querySucceeded": True, "devices": devices}


def probe_capabilities() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    cuda_devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "computeCapability": list(torch.cuda.get_device_capability(index)),
                    "totalMemoryBytes": int(properties.total_memory),
                    "bf16Supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pythonVersion": platform.python_version(),
        "torchVersion": torch.__version__,
        "torchCudaBuild": torch.version.cuda,
        "cudaAvailable": cuda_available,
        "cudaDeviceCount": int(torch.cuda.device_count()),
        "cudaDevices": cuda_devices,
        "mpsBuilt": mps_built,
        "mpsAvailable": mps_available,
        "appleSiliconHost": platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"},
        "nvidiaSmi": _nvidia_smi_inventory(),
        "serialNumbersStored": False,
        "userOrHostNamesStored": False,
    }


def _historical_harness_evidence(source: str) -> dict[str, Any]:
    cpu_markers = {
        "checkpointLoadsOnCpu": 'device="cpu"' in source,
        "optimizationAppliesToCpu": 'torch.device("cpu")' in source,
        "validationTensorsLackDevice": "input_ids = corpus.validation.input_ids[start:end]"
        in source,
        "generationTensorsLackDevice": "input_ids = torch.tensor(padded, dtype=torch.long)"
        in source,
    }
    return {
        "path": HISTORICAL_HARNESS_PATH.relative_to(AI_ROOT).as_posix(),
        "classification": "historical-cpu-only-evidence",
        "cpuBoundMarkers": cpu_markers,
        "deviceAware": not any(cpu_markers.values()),
        "mayBeRewrittenInPlace": False,
    }


def _artifact_evidence(
    policy: Mapping[str, Any],
    cpu_report: Mapping[str, Any],
    optimization_policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    package_report: Mapping[str, Any],
) -> dict[str, Any]:
    source = policy.get("sourceEvidence")
    if not isinstance(source, Mapping):
        raise AcceleratorReadinessError("accelerator source evidence policy is invalid")
    cpu_digest = cpu_report.get("reportSha256")
    optimization_digest = optimization_policy.get("policySha256")
    cpu_unsigned = dict(cpu_report)
    cpu_unsigned.pop("reportSha256", None)
    optimization_unsigned = dict(optimization_policy)
    optimization_unsigned.pop("policySha256", None)
    declared_tiers = {
        item.get("tier"): item.get("status")
        for item in cpu_report.get("portability", {}).get("declaredTiers", [])
        if isinstance(item, Mapping)
    }
    gates = {
        "optimizedArtifactIdentityMatches": manifest.get("artifactId")
        == source.get("optimizedArtifactId")
        == package_report.get("artifactId"),
        "artifactIsImmutable": manifest.get("immutable") is True,
        "packageBindsManifest": package_report.get("manifestSha256")
        == manifest.get("manifestSha256"),
        "cpuBenchmarkChecksumValid": cpu_digest
        == hashlib.sha256(_canonical_json(cpu_unsigned)).hexdigest(),
        "optimizationPolicyChecksumValid": optimization_digest
        == hashlib.sha256(_canonical_json(optimization_unsigned)).hexdigest(),
        "optimizationPolicyBindsCpuBenchmark": optimization_policy.get(
            "benchmarkReportSha256"
        )
        == cpu_digest,
        "cpuReportKeepsAcceleratorsUnmeasured": {
            tier: declared_tiers.get(tier) for tier in ("cuda", "mps")
        }
        == {"cuda": "conditional-unmeasured", "mps": "conditional-unmeasured"},
        "localArtifactDirectoryPresent": ARTIFACT_DIRECTORY.is_dir(),
        "weightsUnchangedForAcceleratorBenchmark": source.get(
            "weightsMayChangeForBenchmark"
        )
        is False,
        "tokenizerUnchangedForAcceleratorBenchmark": source.get(
            "tokenizerMayChangeForBenchmark"
        )
        is False,
    }
    return {
        "artifactId": manifest.get("artifactId"),
        "manifestSha256": manifest.get("manifestSha256"),
        "cpuBenchmarkReportSha256": cpu_digest,
        "optimizationPolicySha256": optimization_digest,
        "gates": gates,
        "ready": all(gates.values()),
    }


def _accelerator_harness_evidence(policy: Mapping[str, Any]) -> dict[str, Any]:
    source = policy.get("sourceEvidence")
    path_value = (
        source.get("acceleratorBenchmarkHarnessPath")
        if isinstance(source, Mapping)
        else None
    )
    if path_value is None:
        return {
            "path": None,
            "assigned": False,
            "exists": False,
            "deviceAwareContractVerified": False,
            "sha256": None,
        }
    path = _resolve_workspace_path(path_value, "accelerator benchmark harness")
    exists = path.is_file()
    source_text = path.read_text(encoding="utf-8") if exists else ""
    required_markers = (
        "requested_device",
        "synchronize_before_and_after_timing",
        "peak_device_memory",
        "thermal_cycle",
        "complete_frozen_validation",
    )
    return {
        "path": path_value,
        "assigned": True,
        "exists": exists,
        "deviceAwareContractVerified": exists
        and all(marker in source_text for marker in required_markers),
        "sha256": _sha256_file(path) if exists else None,
    }


def _tier_result_evidence(tier: Mapping[str, Any]) -> dict[str, Any]:
    tier_id = tier.get("tierId")
    path_value = tier.get("resultReportPath")
    if path_value is None:
        return {
            "tierId": tier_id,
            "path": None,
            "assigned": False,
            "valid": False,
            "allAcceptanceGatesPassed": False,
            "reportSha256": None,
        }
    path = _resolve_workspace_path(path_value, f"{tier_id} result report")
    if not path.is_file():
        return {
            "tierId": tier_id,
            "path": path_value,
            "assigned": True,
            "valid": False,
            "allAcceptanceGatesPassed": False,
            "reportSha256": None,
        }
    report = _read_json_object(path, f"{tier_id} result report")
    digest = report.get("reportSha256")
    valid = digest == _receipt_digest(report) and report.get("tierId") == tier_id
    required_gates = report.get("acceptanceGates")
    all_passed = bool(
        valid
        and isinstance(required_gates, Mapping)
        and required_gates
        and all(value is True for value in required_gates.values())
        and report.get("releaseOrActivationApproved") is False
        and report.get("networkAccessed") is False
        and report.get("governedDataExported") is False
    )
    return {
        "tierId": tier_id,
        "path": path_value,
        "assigned": True,
        "valid": valid,
        "allAcceptanceGatesPassed": all_passed,
        "reportSha256": digest,
    }


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    historical_harness: Mapping[str, Any],
    accelerator_harness: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any],
    tier_results: Sequence[Mapping[str, Any]],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-007":
        raise AcceleratorReadinessError("accelerator policy is not bound to VFAI-FU-007")
    fu007 = _backlog_item(followup_backlog, "VFAI-FU-007")
    target_tiers = policy.get("targetTiers")
    if not isinstance(target_tiers, list) or {item.get("tierId") for item in target_tiers} != {
        "cuda",
        "mps",
    }:
        raise AcceleratorReadinessError("accelerator target tiers are invalid")
    results_by_tier = {item.get("tierId"): item for item in tier_results}
    if set(results_by_tier) != {"cuda", "mps"}:
        raise AcceleratorReadinessError("accelerator tier result evidence is incomplete")
    hardware = {
        "cuda": bool(capabilities.get("cudaAvailable"))
        and int(capabilities.get("cudaDeviceCount", 0)) > 0,
        "mps": bool(capabilities.get("mpsAvailable"))
        and bool(capabilities.get("appleSiliconHost")),
    }
    harness_ready = accelerator_harness.get("deviceAwareContractVerified") is True
    artifact_ready = artifact_evidence.get("ready") is True
    tier_gates = []
    for tier in target_tiers:
        tier_id = str(tier["tierId"])
        result = results_by_tier.get(tier_id, {})
        gates = {
            "ownedHardwareVisible": hardware[tier_id],
            "immutableArtifactEvidenceReady": artifact_ready,
            "deviceAwareBenchmarkHarnessReady": harness_ready,
            "resultReportAssigned": result.get("assigned") is True,
            "resultReportValid": result.get("valid") is True,
            "allAcceptanceGatesPassed": result.get("allAcceptanceGatesPassed") is True,
        }
        tier_gates.append(
            {
                "tierId": tier_id,
                "requiredProfiles": tier.get("requiredProfiles"),
                "conditionalPrecisionProfiles": tier.get("conditionalPrecisionProfiles"),
                "gates": gates,
                "accepted": all(gates.values()),
                "result": dict(result),
            }
        )
    master_ready = _master_complete(master_backlog)
    visible_tiers = [tier for tier, visible in hardware.items() if visible]
    accepted_tiers = [item["tierId"] for item in tier_gates if item["accepted"]]
    if not master_ready:
        decision = "await-master-backlog-completion"
    elif not visible_tiers:
        decision = "await-owned-accelerator-hardware"
    elif not artifact_ready:
        decision = "repair-immutable-artifact-evidence"
    elif not harness_ready:
        decision = "implement-device-aware-accelerator-harness"
    elif not any(results_by_tier[tier].get("assigned") for tier in visible_tiers):
        decision = "ready-to-run-owned-accelerator-matrix"
    elif len(accepted_tiers) < 2:
        decision = "await-remaining-owned-accelerator-tier"
    else:
        decision = "accelerator-results-ready-for-independent-package-review"
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-007-owned-accelerator-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": fu007.get("id"), "recordedStatus": fu007.get("status")},
        "policy": {
            "policyId": policy.get("policyId"),
            "status": policy.get("status"),
            "networkAllowed": policy.get("measurementProtocol", {}).get("networkAllowed"),
            "governedDataMayLeaveDevice": policy.get("measurementProtocol", {}).get(
                "governedDataMayLeaveDevice"
            ),
        },
        "capabilities": dict(capabilities),
        "historicalHarness": dict(historical_harness),
        "acceleratorHarness": dict(accelerator_harness),
        "artifactEvidence": dict(artifact_evidence),
        "tiers": tier_gates,
        "gates": {
            "masterBacklogComplete": master_ready,
            "anyOwnedAcceleratorVisible": bool(visible_tiers),
            "cudaVisible": hardware["cuda"],
            "mpsVisible": hardware["mps"],
            "immutableArtifactEvidenceReady": artifact_ready,
            "deviceAwareAcceleratorHarnessReady": harness_ready,
            "acceptedAcceleratorTiers": accepted_tiers,
            "bothAcceleratorTiersAccepted": set(accepted_tiers) == {"cuda", "mps"},
        },
        "decision": decision,
        "completionClaimed": False,
        "modelWeightsLoaded": False,
        "governedCorpusLoaded": False,
        "benchmarkMeasurementsExecuted": 0,
        "governedDataExported": False,
        "networkAccessed": False,
        "releaseOrActivationApproved": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def _current_inputs() -> dict[str, Any]:
    master = _read_json_object(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json_object(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json_object(POLICY_PATH, "accelerator policy")
    cpu_report = _read_json_object(CPU_BENCHMARK_PATH, "CPU benchmark report")
    optimization_policy = _read_json_object(
        OPTIMIZATION_POLICY_PATH, "optimization policy"
    )
    manifest = _read_json_object(ARTIFACT_MANIFEST_PATH, "optimized artifact manifest")
    package_report = _read_json_object(
        ARTIFACT_PACKAGE_REPORT_PATH, "optimized artifact package report"
    )
    historical_source = HISTORICAL_HARNESS_PATH.read_text(encoding="utf-8")
    target_tiers = policy.get("targetTiers")
    if not isinstance(target_tiers, list):
        raise AcceleratorReadinessError("accelerator target tiers are invalid")
    source_paths = (
        MASTER_BACKLOG_PATH,
        FOLLOWUP_BACKLOG_PATH,
        POLICY_PATH,
        CPU_BENCHMARK_PATH,
        OPTIMIZATION_POLICY_PATH,
        ARTIFACT_MANIFEST_PATH,
        ARTIFACT_PACKAGE_REPORT_PATH,
        HISTORICAL_HARNESS_PATH,
        Path(__file__),
    )
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "capabilities": probe_capabilities(),
        "historical_harness": _historical_harness_evidence(historical_source),
        "accelerator_harness": _accelerator_harness_evidence(policy),
        "artifact_evidence": _artifact_evidence(
            policy, cpu_report, optimization_policy, manifest, package_report
        ),
        "tier_results": [_tier_result_evidence(tier) for tier in target_tiers],
        "source_sha256": {
            path.relative_to(AI_ROOT).as_posix(): _sha256_file(path)
            for path in source_paths
        },
    }


def _build_current_report(generated_on: str) -> dict[str, Any]:
    return build_report(generated_on=generated_on, **_current_inputs())


def evaluate(path: Path, generated_on: str) -> dict[str, Any]:
    report = _build_current_report(generated_on)
    _write_json(report, path)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _receipt_digest(report):
        raise AcceleratorReadinessError("VFAI-FU-007 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        raise AcceleratorReadinessError("VFAI-FU-007 readiness gates are invalid")
    tiers = report.get("tiers")
    if not isinstance(tiers, list) or len(tiers) != 2:
        raise AcceleratorReadinessError("VFAI-FU-007 tier evidence is invalid")
    for tier in tiers:
        tier_gates = tier.get("gates") if isinstance(tier, Mapping) else None
        if not isinstance(tier_gates, Mapping):
            raise AcceleratorReadinessError("VFAI-FU-007 tier gates are invalid")
        if tier.get("accepted") is not all(value is True for value in tier_gates.values()):
            raise AcceleratorReadinessError("VFAI-FU-007 tier decision is inconsistent")
    if gates.get("masterBacklogComplete") is not True:
        expected = "await-master-backlog-completion"
    elif gates.get("anyOwnedAcceleratorVisible") is not True:
        expected = "await-owned-accelerator-hardware"
    elif gates.get("immutableArtifactEvidenceReady") is not True:
        expected = "repair-immutable-artifact-evidence"
    elif gates.get("deviceAwareAcceleratorHarnessReady") is not True:
        expected = "implement-device-aware-accelerator-harness"
    else:
        visible = {
            item.get("tierId")
            for item in tiers
            if item.get("gates", {}).get("ownedHardwareVisible") is True
        }
        assigned_visible = any(
            item.get("tierId") in visible
            and item.get("gates", {}).get("resultReportAssigned") is True
            for item in tiers
        )
        accepted = {
            item.get("tierId") for item in tiers if item.get("accepted") is True
        }
        if not assigned_visible:
            expected = "ready-to-run-owned-accelerator-matrix"
        elif accepted != {"cuda", "mps"}:
            expected = "await-remaining-owned-accelerator-tier"
        else:
            expected = "accelerator-results-ready-for-independent-package-review"
    if report.get("decision") != expected:
        raise AcceleratorReadinessError("VFAI-FU-007 receipt decision is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("modelWeightsLoaded") is not False,
            report.get("governedCorpusLoaded") is not False,
            report.get("benchmarkMeasurementsExecuted") != 0,
            report.get("governedDataExported") is not False,
            report.get("networkAccessed") is not False,
            report.get("releaseOrActivationApproved") is not False,
        )
    ):
        raise AcceleratorReadinessError("VFAI-FU-007 receipt overclaims execution")


def verify(path: Path) -> dict[str, Any]:
    report = _read_json_object(path, "VFAI-FU-007 readiness receipt")
    validate_report(report)
    expected = _build_current_report(str(report.get("generatedOn")))
    if report != expected:
        raise AcceleratorReadinessError("VFAI-FU-007 readiness receipt is stale")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--generated-on", required=True)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = (
        evaluate(arguments.output.resolve(), arguments.generated_on)
        if arguments.command == "evaluate"
        else verify(arguments.input.resolve())
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": arguments.command,
                "reportId": report["reportId"],
                "decision": report["decision"],
                "cudaVisible": report["gates"]["cudaVisible"],
                "mpsVisible": report["gates"]["mpsVisible"],
                "deviceAwareAcceleratorHarnessReady": report["gates"][
                    "deviceAwareAcceleratorHarnessReady"
                ],
                "acceptedAcceleratorTiers": report["gates"][
                    "acceptedAcceleratorTiers"
                ],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
