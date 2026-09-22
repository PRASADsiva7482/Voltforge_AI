"""Review governed feedback and schedule reproducible retraining inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from feedback_governance import (  # noqa: E402
    FeedbackGovernanceError,
    feedback_health,
    get_feedback_store,
    verify_retraining_run,
)
from config import get_settings  # noqa: E402


def _default_database() -> Path:
    return Path(get_settings().runtime_directory) / "feedback-governance-v1" / "feedback.sqlite3"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FeedbackGovernanceError("FEEDBACK_INPUT_INVALID", f"Feedback input is invalid: {path.name}.") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--artifact-directory", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health")
    health.set_defaults(command="health")

    review = subparsers.add_parser("review")
    review.add_argument("feedback_id")
    review.add_argument("--decision", choices=("reject", "approve-heldout"), required=True)
    review.add_argument("--reviewer-id", required=True)
    review.add_argument("--note", default="")
    review.add_argument("--expected-version", type=int)

    training = subparsers.add_parser("approve-training")
    training.add_argument("feedback_id")
    training.add_argument("--example", type=Path, required=True)
    training.add_argument("--reviewer-id", required=True)
    training.add_argument("--note", default="")
    training.add_argument("--expected-version", type=int)

    schedule = subparsers.add_parser("schedule")
    schedule.add_argument("--feedback-id", action="append", required=True)
    schedule.add_argument("--base-artifact-id", required=True)
    schedule.add_argument("--base-registry-revision", type=int, required=True)
    schedule.add_argument("--seed", type=int, default=34034)
    schedule.add_argument("--hyperparameters", type=Path)
    schedule.add_argument("--output-directory", type=Path)

    verify = subparsers.add_parser("verify-run")
    verify.add_argument("run", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "health":
            result = feedback_health(args.database or _default_database())
        elif args.command == "review":
            result = get_feedback_store(args.database or _default_database(), artifact_directory=args.artifact_directory).review(
                args.feedback_id,
                decision=args.decision,
                reviewer_id=args.reviewer_id,
                review_note=args.note,
                expected_version=args.expected_version,
            )
        elif args.command == "approve-training":
            result = get_feedback_store(args.database or _default_database(), artifact_directory=args.artifact_directory).approve_training(
                args.feedback_id,
                training_example=_json(args.example),
                reviewer_id=args.reviewer_id,
                review_note=args.note,
                expected_version=args.expected_version,
            )
        elif args.command == "schedule":
            hyperparameters = _json(args.hyperparameters) if args.hyperparameters else {}
            result = get_feedback_store(args.database or _default_database(), artifact_directory=args.artifact_directory).schedule_retraining(
                args.feedback_id,
                base_artifact_id=args.base_artifact_id,
                base_registry_revision=args.base_registry_revision,
                seed=args.seed,
                hyperparameters=hyperparameters,
                output_directory=args.output_directory,
            )
        else:
            result = verify_retraining_run(args.run)
    except FeedbackGovernanceError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": error.message}, indent=2))
        return 1
    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
