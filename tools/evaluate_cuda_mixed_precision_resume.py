"""Generate or verify the VFAI-FU-004 owned-CUDA qualification receipt.

This evaluator performs only hardware and arithmetic capability probes. It does
not load the governed corpus or claim the controlled training/resume scorecard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import torch


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/cuda-mixed-precision-resume-v1.json"
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _nvidia_smi_inventory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "devices": []}
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,compute_cap,memory.total,driver_version",
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
        if len(values) == 4:
            devices.append(
                {
                    "name": values[0],
                    "computeCapability": values[1],
                    "memoryMiB": int(values[2]),
                    "driverVersion": values[3],
                }
            )
    return {"available": True, "querySucceeded": True, "devices": devices}


def _precision_smoke(name: str, dtype: torch.dtype) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "precision": name,
            "executed": False,
            "passed": False,
            "reason": "CUDA_UNAVAILABLE",
        }
    if name == "bfloat16" and not torch.cuda.is_bf16_supported():
        return {
            "precision": name,
            "executed": False,
            "passed": False,
            "reason": "CUDA_BF16_UNSUPPORTED",
        }
    device = torch.device("cuda", 0)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(4_004)
        torch.cuda.manual_seed_all(4_004)
        value = torch.linspace(-1.0, 1.0, 4096, device=device).reshape(64, 64)
        value.requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = (value @ value.transpose(0, 1)).square().mean()
        output.backward()
        torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(output) and torch.isfinite(value.grad).all())
        return {
            "precision": name,
            "executed": True,
            "passed": finite,
            "reason": None if finite else "NON_FINITE_ARITHMETIC",
            "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
        }
    except (RuntimeError, OSError) as exc:
        return {
            "precision": name,
            "executed": True,
            "passed": False,
            "reason": type(exc).__name__,
        }
    finally:
        torch.cuda.empty_cache()


def probe_capabilities() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    cuda_available = bool(torch.cuda.is_available())
    capability: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pythonVersion": platform.python_version(),
        "torchVersion": torch.__version__,
        "torchCudaBuild": torch.version.cuda,
        "cudaAvailable": cuda_available,
        "cudaDeviceCount": int(torch.cuda.device_count()),
        "cpuCapability": torch.backends.cpu.get_cpu_capability(),
        "cpuAvx512Bf16": bool(torch.cpu._is_avx512_bf16_supported()),
    }
    if cuda_available:
        capability["cudaDevices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "bf16Supported": bool(torch.cuda.is_bf16_supported()),
            }
            for index in range(torch.cuda.device_count())
        ]
    else:
        capability["cudaDevices"] = []
    smokes = [
        _precision_smoke("bfloat16", torch.bfloat16),
        _precision_smoke("float16", torch.float16),
    ]
    return capability, smokes, _nvidia_smi_inventory()


def build_report(
    *,
    capabilities: Mapping[str, Any],
    precision_smokes: Sequence[Mapping[str, Any]],
    nvidia_smi: Mapping[str, Any],
    generated_on: str,
    trainer_source: str | None = None,
) -> dict[str, Any]:
    source = trainer_source
    if source is None:
        source = (AI_ROOT / "gen1_training/trainer.py").read_text(encoding="utf-8")
    overflow_recovery_present = all(
        marker in source
        for marker in (
            "MAXIMUM_OVERFLOW_RETRIES_PER_STEP",
            "mixedPrecisionOverflowRetries",
            "self.stream.load_state_dict(stream_state)",
            "_restore_rng_state(rng_state)",
        )
    )
    cuda_visible = bool(capabilities.get("cudaAvailable")) and int(
        capabilities.get("cudaDeviceCount", 0)
    ) > 0
    smoke_by_precision = {
        str(item["precision"]): bool(item.get("passed")) for item in precision_smokes
    }
    capability_smokes_pass = all(
        smoke_by_precision.get(name, False) for name in ("bfloat16", "float16")
    )
    experiment_status = (
        "pending-controlled-owned-cuda-runs"
        if cuda_visible and capability_smokes_pass
        else "not-run-owned-cuda-unavailable"
    )
    decision = (
        "ready-for-controlled-owned-cuda-scorecard"
        if cuda_visible and capability_smokes_pass
        else "await-owned-cuda-hardware"
    )
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-004-cuda-mixed-precision-resume-v1",
        "generatedOn": generated_on,
        "capabilities": dict(capabilities),
        "nvidiaSmi": dict(nvidia_smi),
        "precisionSmokes": [dict(item) for item in precision_smokes],
        "gates": {
            "ownedCudaVisibleToPinnedRuntime": cuda_visible,
            "bfloat16CapabilitySmokePassed": smoke_by_precision.get("bfloat16", False),
            "float16CapabilitySmokePassed": smoke_by_precision.get("float16", False),
            "safePointOverflowRecoveryImplemented": overflow_recovery_present,
            "controlledFp32Bf16Fp16Scorecard": experiment_status,
            "cudaResumeEquivalence": experiment_status,
            "cudaThroughputPeakMemoryAndValidationComparison": experiment_status,
        },
        "decision": decision,
        "completionClaimed": False,
        "governedCorpusLoaded": False,
        "trainingDataTransferred": False,
        "networkAccessed": False,
        "trainerSourceSha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "evaluatorSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def evaluate(path: Path, generated_on: str) -> dict[str, Any]:
    capabilities, smokes, nvidia_smi = probe_capabilities()
    report = build_report(
        capabilities=capabilities,
        precision_smokes=smokes,
        nvidia_smi=nvidia_smi,
        generated_on=generated_on,
    )
    _write_json(report, path)
    return report


def verify(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("reportSha256") != _receipt_digest(report):
        raise RuntimeError("VFAI-FU-004 capability receipt checksum is invalid")
    trainer_source = (AI_ROOT / "gen1_training/trainer.py").read_bytes()
    if report.get("trainerSourceSha256") != hashlib.sha256(trainer_source).hexdigest():
        raise RuntimeError("VFAI-FU-004 capability receipt trainer source is stale")
    if report.get("evaluatorSha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("VFAI-FU-004 capability receipt evaluator source is stale")
    cuda_visible = report.get("gates", {}).get("ownedCudaVisibleToPinnedRuntime")
    expected = (
        "ready-for-controlled-owned-cuda-scorecard"
        if cuda_visible
        and report.get("gates", {}).get("bfloat16CapabilitySmokePassed")
        and report.get("gates", {}).get("float16CapabilitySmokePassed")
        else "await-owned-cuda-hardware"
    )
    if report.get("decision") != expected or report.get("completionClaimed") is not False:
        raise RuntimeError("VFAI-FU-004 capability receipt contradicts its gates")
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
                "cudaAvailable": report["capabilities"]["cudaAvailable"],
                "cudaDeviceCount": report["capabilities"]["cudaDeviceCount"],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
