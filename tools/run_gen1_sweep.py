#!/usr/bin/env python3
"""Run one controlled Gen1 architecture candidate or verify its scorecard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from gen1_sweep import (  # noqa: E402
    SweepContractError,
    assemble_scorecard,
    check_scorecard,
    load_sweep_plan,
    run_candidate,
)
from gen1_sweep.sweep import (  # noqa: E402
    DEFAULT_PLAN_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_SELECTED_CONFIG_PATH,
)


DEFAULT_WORK_ROOT = AI_ROOT / ".toolchains" / "gen1-sweep-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a single VFAI-013 candidate in a fresh process, assemble all "
            "candidate evidence, or validate the committed decision."
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--candidate", help="candidate ID to train and benchmark")
    action.add_argument("--assemble", action="store_true", help="assemble all results")
    action.add_argument("--check", action="store_true", help="verify committed evidence")
    action.add_argument("--list", action="store_true", help="list controlled candidates")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--selected-config", type=Path, default=DEFAULT_SELECTED_CONFIG_PATH
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.list:
            plan = load_sweep_plan(args.plan)
            payload = {
                "sweepId": plan["sweepId"],
                "planFingerprint": plan["planFingerprint"],
                "candidates": [item["id"] for item in plan["candidates"]],
                "nonExecutableControls": plan["nonExecutableControls"],
            }
        elif args.candidate:
            result = run_candidate(args.candidate, args.work_root, plan_path=args.plan)
            payload = {
                "decision": "candidate-complete",
                "candidateId": result["candidateId"],
                "parameterCount": result["parameterCount"],
                "validationLoss": result["quality"]["final"]["nextTokenLoss"],
                "trainingWallSeconds": result["training"]["wallSeconds"],
                "resultSha256": result["resultSha256"],
            }
        elif args.assemble:
            report = assemble_scorecard(
                args.work_root,
                plan_path=args.plan,
                report_path=args.report,
                selected_config_path=args.selected_config,
            )
            payload = {
                "decision": "scorecard-assembled",
                "reportId": report["reportId"],
                "selectedCandidateId": report["selection"]["selectedCandidateId"],
                "paretoFrontier": report["selection"]["paretoFrontier"],
                "reportSha256": report["reportSha256"],
            }
        else:
            payload = check_scorecard(
                plan_path=args.plan,
                report_path=args.report,
                selected_config_path=args.selected_config,
            )
    except SweepContractError as exc:
        print(json.dumps({"decision": "fail", "reason": str(exc)}, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
