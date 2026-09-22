"""Evaluate the first gate for the VFAI-FU-003 Windows PyTorch review.

The candidate is always probed from fresh child processes.  This evaluator does
not change the verified Gen1 environment, install packages, or upgrade project
pins.  A candidate that cannot import cleanly is rejected before model tests or
benchmarks are attempted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/pytorch-windows-runtime-review-v1.json"
)
UPSTREAM_ISSUE = "https://github.com/pytorch/pytorch/issues/166628"
OFFICIAL_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CHECKPOINT_RCE_ADVISORY = "GHSA-63cw-57p8-fm3p"
CHECKPOINT_RCE_ADVISORY_URL = (
    "https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p"
)
CHECKPOINT_RCE_FIXED_VERSION = (2, 10, 0)


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


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(AI_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _runtime_metadata(python: Path, timeout_seconds: float) -> dict[str, str]:
    program = (
        "import importlib.metadata,json,platform,sys;"
        "print(json.dumps({'pythonVersion':platform.python_version(),"
        "'implementation':platform.python_implementation(),"
        "'architecture':platform.machine(),"
        "'torchDistributionVersion':importlib.metadata.version('torch'),"
        "'executable':sys.executable}))"
    )
    result = subprocess.run(
        [str(python), "-c", program],
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not inspect runtime metadata for {_display_path(python)}"
        )
    metadata = json.loads(result.stdout)
    return {
        "pythonVersion": str(metadata["pythonVersion"]),
        "implementation": str(metadata["implementation"]),
        "architecture": str(metadata["architecture"]),
        "torchDistributionVersion": str(metadata["torchDistributionVersion"]),
        "interpreter": _display_path(python),
    }


def _classify_failure(stderr: str, return_code: int | None) -> str:
    if "WinError 1114" in stderr and "c10.dll" in stderr:
        return "WINERROR_1114_C10_DLL_INITIALIZATION_FAILED"
    if return_code is None:
        return "IMPORT_TIMEOUT"
    if return_code < 0 or return_code >= 0x80000000:
        return "NATIVE_PROCESS_FAILURE"
    return "TORCH_IMPORT_FAILED"


def _fresh_import(python: Path, timeout_seconds: float) -> tuple[bool, str]:
    program = "import torch; print(torch.__version__)"
    try:
        result = subprocess.run(
            [str(python), "-c", program],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, _classify_failure("", None)
    if result.returncode == 0:
        return True, result.stdout.strip()
    return False, _classify_failure(result.stderr, result.returncode)


def probe_runtime(
    python: Path,
    *,
    attempts: int,
    workers: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    metadata = _runtime_metadata(python, timeout_seconds)
    with ThreadPoolExecutor(max_workers=min(workers, attempts)) as executor:
        results = list(
            executor.map(
                lambda _: _fresh_import(python, timeout_seconds), range(attempts)
            )
        )
    passed = sum(success for success, _ in results)
    versions = sorted({detail for success, detail in results if success})
    failures = Counter(detail for success, detail in results if not success)
    return {
        **metadata,
        "cleanProcessImports": {
            "attempted": attempts,
            "passed": passed,
            "failed": attempts - passed,
            "observedImportedVersions": versions,
            "failureCounts": dict(sorted(failures.items())),
        },
    }


def _master_backlog_complete() -> bool:
    document = json.loads((AI_ROOT / "AI_MASTER_BACKLOG.json").read_text(encoding="utf-8"))
    terminal_statuses = {"complete", "done"}
    return document.get("status") == "completed" and all(
        item.get("status") in terminal_statuses for item in document.get("items", [])
    )


def _release_tuple(version: str) -> tuple[int, int, int]:
    base = version.partition("+")[0]
    parts = base.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"unsupported release version: {version}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def build_report(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    upstream_issue_status: str,
    generated_on: str,
    master_backlog_complete: bool,
    host_operating_system: str | None = None,
    minimum_patched_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_imports = baseline["cleanProcessImports"]
    candidate_imports = candidate["cleanProcessImports"]
    baseline_gate = (
        baseline_imports["attempted"] >= 20
        and baseline_imports["passed"] == baseline_imports["attempted"]
    )
    candidate_gate = (
        candidate_imports["attempted"] >= 20
        and candidate_imports["passed"] == candidate_imports["attempted"]
    )
    upstream_gate = upstream_issue_status == "resolved"
    operating_system = host_operating_system or platform.system()
    windows_gate = operating_system == "Windows"
    different_runtime_gate = (
        candidate["torchDistributionVersion"]
        != baseline["torchDistributionVersion"]
    )
    baseline_affected = (
        _release_tuple(str(baseline["torchDistributionVersion"]))
        < CHECKPOINT_RCE_FIXED_VERSION
    )
    minimum_patched_import_gate = False
    if minimum_patched_candidate is not None:
        minimum_imports = minimum_patched_candidate["cleanProcessImports"]
        minimum_patched_import_gate = (
            _release_tuple(str(minimum_patched_candidate["torchDistributionVersion"]))
            >= CHECKPOINT_RCE_FIXED_VERSION
            and minimum_imports["attempted"] >= 20
            and minimum_imports["passed"] == minimum_imports["attempted"]
        )
    runtime_service_source = (AI_ROOT / "model/runtime_service.py").read_text(
        encoding="utf-8"
    )
    serving_containment = all(
        marker in runtime_service_source
        for marker in (
            CHECKPOINT_RCE_ADVISORY,
            "MODEL_RUNTIME_CHECKPOINT_SECURITY_BLOCKED",
        )
    )
    ready_for_downstream = all(
        (
            master_backlog_complete,
            windows_gate,
            upstream_gate,
            baseline_gate,
            candidate_gate,
            different_runtime_gate,
        )
    )
    if ready_for_downstream:
        downstream_status = "pending-candidate-import-gate-passed"
    elif not candidate_gate:
        downstream_status = "not-run-candidate-import-gate-failed"
    else:
        downstream_status = "not-run-review-prerequisite-gate-failed"
    if ready_for_downstream:
        decision = "candidate-ready-for-downstream-evaluation"
    elif baseline_affected:
        decision = "retain-pinned-runtime-with-serving-blocked"
    else:
        decision = "retain-pinned-runtime"
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-003-pytorch-windows-runtime-review-v1",
        "generatedOn": generated_on,
        "host": {
            "operatingSystem": operating_system,
            "release": platform.release(),
            "architecture": platform.machine(),
        },
        "upstream": {
            "issue": UPSTREAM_ISSUE,
            "observedStatus": upstream_issue_status,
            "candidateDistributionSource": OFFICIAL_CPU_INDEX,
        },
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "security": {
            "advisory": CHECKPOINT_RCE_ADVISORY,
            "advisoryUrl": CHECKPOINT_RCE_ADVISORY_URL,
            "affectedVersions": "<=2.9.1",
            "patchedVersions": ">=2.10.0",
            "baselineAffected": baseline_affected,
            "minimumPatchedCandidate": (
                dict(minimum_patched_candidate)
                if minimum_patched_candidate is not None
                else None
            ),
            "minimumPatchedCandidateImportGatePassed": minimum_patched_import_gate,
            "checkpointBackedServingContainmentImplemented": serving_containment,
            "pinnedRuntimePermittedScope": "trusted-local-training-and-testing-only",
        },
        "gates": {
            "masterBacklogComplete": master_backlog_complete,
            "windowsHost": windows_gate,
            "upstreamWindowsIssueResolved": upstream_gate,
            "baselineFreshImportGatePassed": baseline_gate,
            "candidateFreshImportGatePassed": candidate_gate,
            "candidateDiffersFromBaseline": different_runtime_gate,
            "candidateArchitectureAndCheckpointSuite": downstream_status,
            "candidateFullAiSuite": downstream_status,
            "candidateCpuLatencyAndPeakMemoryComparison": downstream_status,
        },
        "decision": decision,
        "upgradeApplied": False,
        "verifiedGen1ToolchainModified": False,
        "pretrainedModelOrTokenizerAdded": False,
        "hostedInferenceProviderAdded": False,
        "externalGenerationApiAdded": False,
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def evaluate(
    *,
    baseline_python: Path,
    candidate_python: Path,
    minimum_patched_python: Path | None,
    output: Path,
    upstream_issue_status: str,
    generated_on: str,
    attempts: int,
    workers: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    baseline = probe_runtime(
        baseline_python,
        attempts=attempts,
        workers=workers,
        timeout_seconds=timeout_seconds,
    )
    candidate = probe_runtime(
        candidate_python,
        attempts=attempts,
        workers=workers,
        timeout_seconds=timeout_seconds,
    )
    minimum_patched_candidate = (
        probe_runtime(
            minimum_patched_python,
            attempts=attempts,
            workers=workers,
            timeout_seconds=timeout_seconds,
        )
        if minimum_patched_python is not None
        else None
    )
    report = build_report(
        baseline=baseline,
        candidate=candidate,
        upstream_issue_status=upstream_issue_status,
        generated_on=generated_on,
        master_backlog_complete=_master_backlog_complete(),
        minimum_patched_candidate=minimum_patched_candidate,
    )
    _write_json(report, output)
    return report


def verify(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("reportSha256") != _receipt_digest(report):
        raise RuntimeError("VFAI-FU-003 review report checksum is invalid")
    gates = report.get("gates", {})
    ready_for_downstream = all(
        gates.get(name)
        for name in (
            "masterBacklogComplete",
            "windowsHost",
            "upstreamWindowsIssueResolved",
            "baselineFreshImportGatePassed",
            "candidateFreshImportGatePassed",
            "candidateDiffersFromBaseline",
        )
    )
    if ready_for_downstream:
        expected_decision = "candidate-ready-for-downstream-evaluation"
    elif report.get("security", {}).get("baselineAffected"):
        expected_decision = "retain-pinned-runtime-with-serving-blocked"
    else:
        expected_decision = "retain-pinned-runtime"
    if report.get("decision") != expected_decision:
        raise RuntimeError("VFAI-FU-003 review report decision contradicts its gates")
    if report.get("upgradeApplied") is not False:
        raise RuntimeError("The review receipt must not claim that it applied an upgrade")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--baseline-python", type=Path, required=True)
    evaluate_parser.add_argument("--candidate-python", type=Path, required=True)
    evaluate_parser.add_argument("--minimum-patched-python", type=Path)
    evaluate_parser.add_argument(
        "--upstream-issue-status", choices=("open", "resolved"), required=True
    )
    evaluate_parser.add_argument("--generated-on", required=True)
    evaluate_parser.add_argument("--attempts", type=int, default=20)
    evaluate_parser.add_argument("--workers", type=int, default=4)
    evaluate_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.command == "evaluate":
        report = evaluate(
            baseline_python=arguments.baseline_python.resolve(),
            candidate_python=arguments.candidate_python.resolve(),
            minimum_patched_python=(
                arguments.minimum_patched_python.resolve()
                if arguments.minimum_patched_python is not None
                else None
            ),
            output=arguments.output.resolve(),
            upstream_issue_status=arguments.upstream_issue_status,
            generated_on=arguments.generated_on,
            attempts=arguments.attempts,
            workers=arguments.workers,
            timeout_seconds=arguments.timeout_seconds,
        )
    else:
        report = verify(arguments.input.resolve())
    print(
        json.dumps(
            {
                "ok": True,
                "command": arguments.command,
                "reportId": report["reportId"],
                "decision": report["decision"],
                "baselineImports": report["baseline"]["cleanProcessImports"],
                "candidateImports": report["candidate"]["cleanProcessImports"],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
