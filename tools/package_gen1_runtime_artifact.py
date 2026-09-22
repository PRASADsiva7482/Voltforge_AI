"""Derive the canonical VFAI-017 runtime package from the immutable bootstrap."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.registry_manager import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    DEFAULT_TRUST_STORE_PATH,
    RegistryManagerError,
    canonical_json_bytes,
    initialize_signing_key,
    json_file_bytes,
    read_json,
    register_artifact,
    sha256_bytes,
    sha256_file,
    sign_document,
    utc_now,
    verify_artifact_directory,
    verify_registry,
    write_json,
)


SOURCE_ARTIFACT_ID = "vfdlm-g1-edge-v0.1.0-bootstrap"
ARTIFACT_ID = "vfdlm-g1-edge-v0.1.1-runtime"
SIGNING_KEY_ID = "vf-local-registry-dev-2026-08"
DEFAULT_PRIVATE_KEY = PROJECT_ROOT / ".toolchains" / "registry-signing" / "vf-local-ed25519.pem"
REGISTRY_ROOT = PROJECT_ROOT / "model" / "registry"
ARTIFACTS_ROOT = REGISTRY_ROOT / "artifacts"
SOURCE_ROOT = ARTIFACTS_ROOT / SOURCE_ARTIFACT_ID
FINAL_ROOT = ARTIFACTS_ROOT / ARTIFACT_ID
MANIFEST_MIRROR = REGISTRY_ROOT / "manifests" / f"{ARTIFACT_ID}.json"
REPORT_PATH = REGISTRY_ROOT / "reports" / f"{ARTIFACT_ID}-package.json"


PATH_RENAMES = {
    "model/checkpoint-manifest.json": "model/manifest.json",
    "tokenizer/tokenizer-config.json": "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer-manifest.json": "tokenizer/tokenizer_manifest.json",
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _copy_payload(staging: Path, source_manifest: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for descriptor in source_manifest["files"]:
        source_relative = descriptor["path"]
        if source_relative in {"generation-policy.json", "provenance.json"}:
            continue
        destination_relative = PATH_RENAMES.get(source_relative, source_relative)
        source = SOURCE_ROOT / Path(*source_relative.split("/"))
        destination = staging / Path(*destination_relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        roles[destination_relative] = descriptor["role"]
    return roles


def _build(staging: Path, private_key_path: Path) -> None:
    source = verify_artifact_directory(
        SOURCE_ROOT,
        trust_store_path=DEFAULT_TRUST_STORE_PATH,
    )
    if source.release_status != "experimental" or source.activation_eligible:
        raise RegistryManagerError(
            "RUNTIME_PACKAGE_SOURCE_INVALID",
            "Runtime package source must remain experimental and activation-ineligible.",
        )
    source_manifest = dict(source.manifest)
    roles = _copy_payload(staging, source_manifest)

    policy = {
        "schemaVersion": 1,
        "policyId": "vfai017-experimental-runtime-policy-v1",
        "artifactId": ARTIFACT_ID,
        "releaseStatus": "experimental",
        "servingEnabled": False,
        "activationEligible": False,
        "offlineExperimentalInferenceEnabled": True,
        "contextLength": 128,
        "maximumInputTokens": 96,
        "maximumNewTokens": 32,
        "samplingEnabled": True,
        "streamingEnabled": True,
        "generationModes": ["greedy", "sample"],
        "temperatureRange": [0.05, 2.0],
        "maximumTopK": 3072,
        "minimumTopP": 0.05,
        "cancellationRequired": True,
        "deadlineRequiredForServiceServing": True,
        "deterministicFallbackRequired": True,
        "promptLoggingAllowed": False,
        "generationNetworkAccess": False,
        "reason": (
            "VFAI-017 permits signed offline runtime evaluation. VFAI-015 still denies "
            "activation and user serving for this undertrained checkpoint."
        ),
    }
    write_json(staging / "generation-policy.json", policy)
    roles["generation-policy.json"] = "generation-policy"

    source_provenance = read_json(SOURCE_ROOT / "provenance.json", "RUNTIME_PROVENANCE_INVALID")
    provenance = deepcopy(source_provenance)
    provenance.update(
        {
            "schemaVersion": 2,
            "artifactId": ARTIFACT_ID,
            "parentArtifact": {
                "artifactId": SOURCE_ARTIFACT_ID,
                "manifestSha256": source.manifest_sha256,
                "manifestFileSha256": source.manifest_file_sha256,
            },
            "runtimeRevision": {
                "item": "VFAI-017",
                "reason": (
                    "Canonicalize tokenizer/checkpoint filenames and authorize only bounded "
                    "offline experimental generation, streaming, cancellation, and deadlines."
                ),
                "weightsChanged": False,
                "tokenizerContentChanged": False,
                "releaseApprovalChanged": False,
            },
        }
    )
    write_json(staging / "provenance.json", provenance)
    roles["provenance.json"] = "package-provenance"

    files = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        files.append(
            {
                "path": relative,
                "role": roles[relative],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = deepcopy(source_manifest)
    manifest.pop("manifestSha256", None)
    manifest.pop("signature", None)
    manifest.update(
        {
            "artifactId": ARTIFACT_ID,
            "createdAtUtc": utc_now(),
            "packageSha256": sha256_bytes(canonical_json_bytes(files)),
            "files": files,
        }
    )
    manifest["model"] = deepcopy(source_manifest["model"])
    manifest["model"].update(
        {
            "checkpointManifestPath": "model/manifest.json",
            "configPath": "model/config.json",
            "weightsPath": "model/weights.pt",
        }
    )
    manifest["tokenizer"] = deepcopy(source_manifest["tokenizer"])
    manifest["tokenizer"]["manifestPath"] = "tokenizer/tokenizer_manifest.json"
    manifest["releaseState"] = {
        "releaseStatus": "experimental",
        "activationEligible": False,
        "servingAllowed": False,
        "offlineExperimentalInferenceAllowed": True,
        "decision": "runtime-ready-for-offline-evaluation-only",
        "blockingReason": (
            "Insufficient approved data and failed global model metrics; VFAI-019 output "
            "quality gates do not yet exist."
        ),
    }
    signed = sign_document(
        manifest,
        digest_field="manifestSha256",
        private_key_path=private_key_path,
        key_id=SIGNING_KEY_ID,
    )
    (staging / "artifact-manifest.json").write_bytes(json_file_bytes(signed))


def package(private_key_path: Path) -> dict[str, Any]:
    initialize_signing_key(
        private_key_path,
        DEFAULT_TRUST_STORE_PATH,
        key_id=SIGNING_KEY_ID,
    )
    if FINAL_ROOT.exists():
        artifact = verify_artifact_directory(
            FINAL_ROOT, trust_store_path=DEFAULT_TRUST_STORE_PATH
        )
    else:
        container = Path(
            tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.staging-", dir=ARTIFACTS_ROOT)
        )
        staging = container / ARTIFACT_ID
        staging.mkdir()
        try:
            _build(staging, private_key_path)
            verify_artifact_directory(staging, trust_store_path=DEFAULT_TRUST_STORE_PATH)
            os.replace(staging, FINAL_ROOT)
            container.rmdir()
        except Exception:
            if container.exists():
                shutil.rmtree(container)
            raise
        artifact = verify_artifact_directory(
            FINAL_ROOT, trust_store_path=DEFAULT_TRUST_STORE_PATH
        )

    registry = register_artifact(
        artifact,
        registry_path=DEFAULT_REGISTRY_PATH,
        trust_store_path=DEFAULT_TRUST_STORE_PATH,
        private_key_path=private_key_path,
        key_id=SIGNING_KEY_ID,
    )
    registry = verify_registry(DEFAULT_REGISTRY_PATH, trust_store_path=DEFAULT_TRUST_STORE_PATH)
    if registry["activeArtifactId"] is not None:
        raise RegistryManagerError(
            "RUNTIME_PACKAGE_ACTIVATION_BOUNDARY_VIOLATION",
            "VFAI-017 must leave activeArtifactId null.",
        )
    manifest_bytes = (FINAL_ROOT / "artifact-manifest.json").read_bytes()
    if MANIFEST_MIRROR.exists() and MANIFEST_MIRROR.read_bytes() != manifest_bytes:
        raise RegistryManagerError(
            "RUNTIME_PACKAGE_MANIFEST_CONFLICT",
            "Tracked runtime manifest differs from the immutable package.",
        )
    if not MANIFEST_MIRROR.exists():
        MANIFEST_MIRROR.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_MIRROR.write_bytes(manifest_bytes)
    entry = next(item for item in registry["artifacts"] if item["artifactId"] == ARTIFACT_ID)
    report = {
        "schemaVersion": 1,
        "reportId": "vfai017-runtime-package-v1",
        "generatedAtUtc": utc_now(),
        "artifactId": ARTIFACT_ID,
        "parentArtifactId": SOURCE_ARTIFACT_ID,
        "artifactRoot": _relative(FINAL_ROOT),
        "manifestMirrorPath": _relative(MANIFEST_MIRROR),
        "manifestSha256": artifact.manifest_sha256,
        "manifestFileSha256": artifact.manifest_file_sha256,
        "packageBytes": artifact.package_bytes,
        "fileCount": len(artifact.manifest["files"]) + 1,
        "weightsChanged": False,
        "tokenizerContentChanged": False,
        "runtimeContract": {
            "canonicalCheckpointManifest": "model/manifest.json",
            "canonicalTokenizerManifest": "tokenizer/tokenizer_manifest.json",
            "offlineExperimentalInferenceEnabled": True,
            "serviceServingEnabled": False,
            "activationEligible": False,
        },
        "verification": {
            "signatureVerified": True,
            "exactFileSetVerified": True,
            "allChecksumsVerified": True,
            "compatibilityVerified": True,
            "active": False,
        },
        "registry": {
            "revision": registry["revision"],
            "registrySha256": registry["registrySha256"],
            "activeArtifactId": registry["activeArtifactId"],
            "catalogEntry": entry,
        },
    }
    report["reportSha256"] = sha256_bytes(canonical_json_bytes(report))
    write_json(REPORT_PATH, report)
    return report


def verify() -> dict[str, Any]:
    artifact = verify_artifact_directory(
        FINAL_ROOT, trust_store_path=DEFAULT_TRUST_STORE_PATH
    )
    registry = verify_registry(DEFAULT_REGISTRY_PATH, trust_store_path=DEFAULT_TRUST_STORE_PATH)
    entry = next(
        (item for item in registry["artifacts"] if item.get("artifactId") == ARTIFACT_ID), None
    )
    if entry is None or entry["manifestSha256"] != artifact.manifest_sha256:
        raise RegistryManagerError(
            "MODEL_REGISTRY_ARTIFACT_MISMATCH", "Runtime package is not cataloged exactly."
        )
    return {
        "artifactId": artifact.artifact_id,
        "manifestSha256": artifact.manifest_sha256,
        "packageBytes": artifact.package_bytes,
        "releaseStatus": artifact.release_status,
        "activationEligible": artifact.activation_eligible,
        "registryRevision": registry["revision"],
        "registrySha256": registry["registrySha256"],
        "activeArtifactId": registry["activeArtifactId"],
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("package", "verify"), nargs="?", default="package")
    parser.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)
    args = parser.parse_args()
    try:
        result = package(args.private_key) if args.command == "package" else verify()
    except RegistryManagerError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": error.message}, indent=2))
        return 1
    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

