"""Manage signed VoltForge model release decisions and atomic rollout pointers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from model.registry_manager import DEFAULT_REGISTRY_PATH, DEFAULT_TRUST_STORE_PATH  # noqa: E402
from model.release_lifecycle import (  # noqa: E402
    ReleaseLifecycleError,
    activate_stable_release,
    create_candidate_release,
    current_release_compatibility,
    promote_candidate_to_stable,
    reject_candidate,
    rollback_reasons,
    rollback_stable_release,
    verify_release_record,
)


DEFAULT_PRIVATE_KEY = AI_ROOT / ".toolchains" / "registry-signing" / "vf-local-registry-dev-2026-08.pem"
DEFAULT_KEY_ID = "vf-local-registry-dev-2026-08"


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseLifecycleError("MODEL_RELEASE_INPUT_INVALID", f"Release input is invalid: {path.name}.") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--trust-store", type=Path, default=DEFAULT_TRUST_STORE_PATH)
    parser.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)
    parser.add_argument("--key-id", default=DEFAULT_KEY_ID)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("artifact_id")
    candidate.add_argument("--scorecard", type=Path, required=True)
    candidate.add_argument("--output", type=Path)

    stable = subparsers.add_parser("stable")
    stable.add_argument("candidate", type=Path)
    stable.add_argument("--canary", type=Path, required=True)
    stable.add_argument("--output", type=Path)

    reject = subparsers.add_parser("reject")
    reject.add_argument("candidate", type=Path)
    reject.add_argument("--output", type=Path)

    activate = subparsers.add_parser("activate")
    activate.add_argument("stable", type=Path)
    activate.add_argument("--expected-revision", type=int)

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("stable", type=Path)
    rollback.add_argument("--expected-revision", type=int)

    verify = subparsers.add_parser("verify")
    verify.add_argument("release", type=Path)

    subparsers.add_parser("compatibility")

    reasons = subparsers.add_parser("rollback-reasons")
    reasons.add_argument("--metrics", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "candidate":
            result = create_candidate_release(
                args.artifact_id,
                scorecard=_json_file(args.scorecard),
                registry_path=args.registry,
                trust_store_path=args.trust_store,
                private_key_path=args.private_key,
                key_id=args.key_id,
                output_path=args.output,
            )
            summary = {"operation": "candidate", "releaseId": result["releaseId"], "state": result["state"], "artifactId": result["artifactId"]}
        elif args.command == "stable":
            candidate = _json_file(args.candidate)
            result = promote_candidate_to_stable(
                args.candidate,
                canary=_json_file(args.canary),
                registry_path=args.registry,
                trust_store_path=args.trust_store,
                private_key_path=args.private_key,
                key_id=args.key_id,
                output_path=args.output,
            )
            summary = {"operation": "stable", "releaseId": result["releaseId"], "state": result["state"], "artifactId": candidate["artifactId"]}
        elif args.command == "reject":
            result = reject_candidate(
                args.candidate,
                registry_path=args.registry,
                trust_store_path=args.trust_store,
                private_key_path=args.private_key,
                key_id=args.key_id,
                output_path=args.output,
            )
            summary = {"operation": "reject", "releaseId": result["releaseId"], "state": result["state"], "artifactId": result["artifactId"]}
        elif args.command == "activate":
            result = activate_stable_release(
                args.stable,
                registry_path=args.registry,
                trust_store_path=args.trust_store,
                private_key_path=args.private_key,
                key_id=args.key_id,
                expected_revision=args.expected_revision,
            )
            summary = {"operation": "activate", "registryRevision": result["revision"], "activeArtifactId": result["activeArtifactId"], "registrySha256": result["registrySha256"]}
        elif args.command == "rollback":
            result = rollback_stable_release(
                args.stable,
                registry_path=args.registry,
                trust_store_path=args.trust_store,
                private_key_path=args.private_key,
                key_id=args.key_id,
                expected_revision=args.expected_revision,
            )
            summary = {"operation": "rollback", "registryRevision": result["revision"], "activeArtifactId": result["activeArtifactId"], "registrySha256": result["registrySha256"]}
        elif args.command == "verify":
            result = verify_release_record(
                args.release,
                trust_store_path=args.trust_store,
                registry_path=args.registry,
            )
            summary = {"operation": "verify", "releaseId": result["releaseId"], "state": result["state"], "artifactId": result["artifactId"]}
        elif args.command == "compatibility":
            summary = {"operation": "compatibility", "compatibility": current_release_compatibility()}
        else:
            summary = {"operation": "rollback-reasons", "reasons": rollback_reasons(_json_file(args.metrics))}
    except ReleaseLifecycleError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": error.message}, indent=2))
        return 1
    print(json.dumps({"ok": True, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
