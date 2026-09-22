#!/usr/bin/env python3
"""Execute or verify the governed VFAI-014 Gen1 bootstrap run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from gen1_bootstrap import (  # noqa: E402
    BootstrapContractError,
    check_bootstrap_scorecard,
    load_bootstrap_plan,
    run_bootstrap,
)
from gen1_bootstrap.bootstrap import (  # noqa: E402
    DEFAULT_BEST_CONFIG_PATH,
    DEFAULT_GLOBAL_REPORT_PATH,
    DEFAULT_PLAN_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_RUN_DIRECTORY,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the VFAI-014 bootstrap or verify its immutable evidence."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="execute the controlled run")
    action.add_argument("--show-plan", action="store_true", help="validate and summarize plan")
    action.add_argument("--check", action="store_true", help="verify committed evidence")
    action.add_argument(
        "--check-artifacts",
        action="store_true",
        help="verify evidence plus local best/last checkpoint bytes",
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--run-directory", type=Path, default=DEFAULT_RUN_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--global-report", type=Path, default=DEFAULT_GLOBAL_REPORT_PATH)
    parser.add_argument("--best-config", type=Path, default=DEFAULT_BEST_CONFIG_PATH)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.show_plan:
            plan = load_bootstrap_plan(args.plan)
            payload = {
                "decision": "plan-valid",
                "planId": plan["planId"],
                "planFingerprint": plan["planFingerprint"],
                "candidateId": plan["selectedConfig"]["candidateId"],
                "parameterCount": plan["selectedConfig"]["parameterCount"],
                "maxSteps": plan["controls"]["maxSteps"],
                "expectedTrainingPredictedTokens": plan["controls"][
                    "expectedTrainingPredictedTokens"
                ],
                "evaluationInterval": plan["controls"]["evaluationInterval"],
                "releaseApproved": False,
            }
        elif args.run:
            report = run_bootstrap(
                plan_path=args.plan,
                run_directory=args.run_directory,
                report_path=args.report,
                global_report_path=args.global_report,
                best_config_path=args.best_config,
            )
            payload = {
                "decision": "bootstrap-complete",
                "reportId": report["reportId"],
                "selectedStep": report["selection"]["selectedStep"],
                "lastStep": report["checkpointScorecard"][-1]["step"],
                "predictedTokens": report["run"]["predictedTokens"],
                "stopReason": report["stoppingDecision"]["reason"],
                "releaseApproved": False,
                "reportSha256": report["reportSha256"],
            }
        else:
            payload = check_bootstrap_scorecard(
                plan_path=args.plan,
                report_path=args.report,
                global_report_path=args.global_report,
                best_config_path=args.best_config,
                require_artifacts=args.check_artifacts,
            )
    except BootstrapContractError as exc:
        print(json.dumps({"decision": "fail", "reason": str(exc)}, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
