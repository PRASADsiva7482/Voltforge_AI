"""Run or verify the VFAI-018 offline Gen1 optimization benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from gen1_optimization.benchmark import (  # noqa: E402
    POLICY_PATH,
    REPORT_PATH,
    build_policy,
    build_report,
    run_all_workers,
    run_worker,
    validate_report,
    write_outputs,
)
from model.registry_manager import read_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("benchmark")
    subparsers.add_parser("verify")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--candidate-json", required=True)
    args = parser.parse_args()

    if args.command == "worker":
        print(json.dumps(run_worker(json.loads(args.candidate_json)), separators=(",", ":")))
        return 0
    if args.command == "benchmark":
        measurements = run_all_workers(Path(__file__).resolve())
        report = build_report(measurements)
        write_outputs(report)
    else:
        report = read_json(REPORT_PATH, "OPTIMIZATION_REPORT_INVALID")
        validate_report(report)
        policy = read_json(POLICY_PATH, "OPTIMIZATION_POLICY_INVALID")
        if policy != build_policy(report):
            raise ValueError("optimization policy does not match the measured report")
    print(
        json.dumps(
            {
                "ok": True,
                "reportPath": REPORT_PATH.relative_to(AI_ROOT).as_posix(),
                "policyPath": POLICY_PATH.relative_to(AI_ROOT).as_posix(),
                "reportSha256": report["reportSha256"],
                "selectedCandidateId": report["selection"]["selectedCandidateId"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
