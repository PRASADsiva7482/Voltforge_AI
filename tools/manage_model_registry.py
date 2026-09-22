"""Verify and atomically manage the signed local VoltForge model registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.registry_manager import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    DEFAULT_TRUST_STORE_PATH,
    RegistryManagerError,
    activate_artifact,
    artifact_root_from_entry,
    recover_latest_registry_history,
    retire_artifact,
    rollback_registry,
    verify_artifact_directory,
    verify_registry,
)


DEFAULT_PRIVATE_KEY = PROJECT_ROOT / ".toolchains" / "registry-signing" / "vf-local-ed25519.pem"
DEFAULT_KEY_ID = "vf-local-registry-dev-2026-08"


def _status(registry_path: Path, trust_store_path: Path) -> dict:
    registry = verify_registry(registry_path, trust_store_path=trust_store_path)
    artifacts = []
    for entry in registry["artifacts"]:
        artifacts.append(
            {
                "artifactId": entry["artifactId"],
                "releaseStatus": entry["releaseStatus"],
                "activationEligible": entry["activationEligible"],
                "manifestSha256": entry["manifestSha256"],
                "active": entry["artifactId"] == registry["activeArtifactId"],
            }
        )
    return {
        "registryRevision": registry["revision"],
        "registrySha256": registry["registrySha256"],
        "activeArtifactId": registry["activeArtifactId"],
        "state": registry["state"],
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("status", "verify", "activate", "rollback", "retire", "recover")
    )
    parser.add_argument("artifact_id", nargs="?")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--trust-store", type=Path, default=DEFAULT_TRUST_STORE_PATH)
    parser.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)
    parser.add_argument("--key-id", default=DEFAULT_KEY_ID)
    parser.add_argument("--expected-revision", type=int)
    parser.add_argument("--reason")
    parser.add_argument("--operator-id")
    args = parser.parse_args()

    try:
        if args.command == "status":
            result = _status(args.registry, args.trust_store)
        elif args.command == "verify":
            registry = verify_registry(args.registry, trust_store_path=args.trust_store)
            entries = registry["artifacts"]
            if args.artifact_id:
                entries = [
                    item for item in entries if item.get("artifactId") == args.artifact_id
                ]
                if not entries:
                    raise RegistryManagerError(
                        "ACTIVE_MODEL_ARTIFACT_NOT_FOUND", "Requested artifact is not registered."
                    )
            verified = [
                verify_artifact_directory(
                    artifact_root_from_entry(args.registry, entry),
                    trust_store_path=args.trust_store,
                )
                for entry in entries
            ]
            result = {
                "registryRevision": registry["revision"],
                "artifacts": [
                    {
                        "artifactId": item.artifact_id,
                        "manifestSha256": item.manifest_sha256,
                        "releaseStatus": item.release_status,
                        "activationEligible": item.activation_eligible,
                        "verified": True,
                    }
                    for item in verified
                ],
            }
        elif args.command == "activate":
            if not args.artifact_id:
                parser.error("activate requires artifact_id")
            registry = activate_artifact(
                args.artifact_id,
                registry_path=args.registry,
                trust_store_path=args.trust_store,
                private_key_path=args.private_key,
                key_id=args.key_id,
                expected_revision=args.expected_revision,
            )
            result = {
                "transaction": "activate",
                "registryRevision": registry["revision"],
                "registrySha256": registry["registrySha256"],
                "activeArtifactId": registry["activeArtifactId"],
            }
        elif args.command == "rollback":
            registry = rollback_registry(
                registry_path=args.registry,
                trust_store_path=args.trust_store,
                private_key_path=args.private_key,
                key_id=args.key_id,
                expected_revision=args.expected_revision,
            )
            result = {
                "transaction": "rollback",
                "registryRevision": registry["revision"],
                "registrySha256": registry["registrySha256"],
                "activeArtifactId": registry["activeArtifactId"],
            }
        elif args.command == "retire":
            if not args.artifact_id:
                parser.error("retire requires artifact_id")
            if not args.reason or not args.operator_id:
                parser.error("retire requires --reason and --operator-id")
            registry = retire_artifact(
                args.artifact_id,
                reason=args.reason,
                operator_id=args.operator_id,
                registry_path=args.registry,
                trust_store_path=args.trust_store,
                private_key_path=args.private_key,
                key_id=args.key_id,
                expected_revision=args.expected_revision,
            )
            result = {
                "transaction": "retire",
                "registryRevision": registry["revision"],
                "registrySha256": registry["registrySha256"],
                "retiredArtifactId": args.artifact_id,
                "activeArtifactId": registry["activeArtifactId"],
            }
        else:
            registry = recover_latest_registry_history(
                registry_path=args.registry,
                trust_store_path=args.trust_store,
            )
            result = {
                "transaction": "recover",
                "registryRevision": registry["revision"],
                "registrySha256": registry["registrySha256"],
                "activeArtifactId": registry["activeArtifactId"],
            }
    except RegistryManagerError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": error.message}, indent=2))
        return 1

    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
