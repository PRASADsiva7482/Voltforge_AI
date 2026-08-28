#!/usr/bin/env python3
"""Generate or verify the VFAI-015 Gen1 scale decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from gen1_decision import (  # noqa: E402
    ScaleDecisionContractError,
    check_scale_decision,
    load_scale_policy,
    make_scale_decision,
)
from gen1_decision.decision import DEFAULT_POLICY_PATH, DEFAULT_REPORT_PATH  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make the evidence-bound Gen1 scale, revise, or stop decision."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--make", action="store_true", help="generate decision report")
    action.add_argument("--check", action="store_true", help="verify decision report")
    action.add_argument("--show-policy", action="store_true", help="validate policy")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.show_policy:
            policy = load_scale_policy(args.policy)
            payload = {
                "decision": "policy-valid",
                "policyId": policy["policyId"],
                "policyFingerprint": policy["policyFingerprint"],
                "profiles": [item["name"] for item in policy["profiles"]],
                "decisionOptions": policy["decisionOptions"],
            }
        elif args.make:
            report = make_scale_decision(policy_path=args.policy, report_path=args.report)
            payload = {
                "decision": "decision-recorded",
                "reportId": report["reportId"],
                "selectedOption": report["decision"]["selectedOption"],
                "primaryBottleneck": "data-limited",
                "parameterIncreaseApproved": report["decision"][
                    "parameterIncreaseApproved"
                ],
                "activeArtifactAtDecision": report["registryObservation"][
                    "activeArtifactId"
                ],
                "nextGate": report["decision"]["nextGate"],
                "reportSha256": report["reportSha256"],
            }
        else:
            payload = check_scale_decision(
                policy_path=args.policy,
                report_path=args.report,
            )
    except ScaleDecisionContractError as exc:
        print(json.dumps({"decision": "fail", "reason": str(exc)}, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
