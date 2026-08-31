"""Package the VFAI-014 Gen1 checkpoint as a signed immutable artifact."""

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
    RuntimeContract,
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


ARTIFACT_ID = "vfdlm-g1-edge-v0.1.0-bootstrap"
SIGNING_KEY_ID = "vf-local-registry-dev-2026-08"
DEFAULT_PRIVATE_KEY = PROJECT_ROOT / ".toolchains" / "registry-signing" / "vf-local-ed25519.pem"
CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "training_runs"
    / "vfai014-gen1-bootstrap-v1"
    / "checkpoints"
    / "step-00000256"
)
TOKENIZER_ROOT = PROJECT_ROOT / "model" / "tokenizers" / "vfdlm-byte-bpe-v1.0.0"
SELECTION_PATH = PROJECT_ROOT / "model" / "gen1" / "configs" / "gen1-bootstrap-best-v1.json"
EVALUATION_PATH = PROJECT_ROOT / "evaluation" / "reports" / "gen1-bootstrap-heldout-v1.json"
DECISION_PATH = PROJECT_ROOT / "model" / "gen1" / "decisions" / "gen1-scale-decision-v1.json"
DECISION_POLICY_PATH = PROJECT_ROOT / "gen1_decision" / "policy.v1.json"
ARTIFACTS_ROOT = PROJECT_ROOT / "model" / "registry" / "artifacts"
REPORTS_ROOT = PROJECT_ROOT / "model" / "registry" / "reports"
MANIFESTS_ROOT = PROJECT_ROOT / "model" / "registry" / "manifests"


def _relative_project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _source_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": _relative_project_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _assert_source_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = read_json(SELECTION_PATH, "PACKAGE_SELECTION_INVALID")
    decision = read_json(DECISION_PATH, "PACKAGE_DECISION_INVALID")
    tokenizer = read_json(TOKENIZER_ROOT / "tokenizer_manifest.json", "PACKAGE_TOKENIZER_INVALID")
    checkpoint_manifest_path = CHECKPOINT_ROOT / "model" / "manifest.json"
    checkpoint_manifest = read_json(checkpoint_manifest_path, "PACKAGE_CHECKPOINT_INVALID")

    if selection.get("selectionStatus") != "best-experimental-bootstrap-checkpoint-not-release":
        raise RegistryManagerError(
            "PACKAGE_SELECTION_INVALID", "Selected checkpoint is not the experimental VFAI-014 result."
        )
    checkpoint = selection.get("checkpoint", {})
    if checkpoint.get("modelWeightsSha256") != sha256_file(CHECKPOINT_ROOT / "model" / "weights.pt"):
        raise RegistryManagerError(
            "PACKAGE_SOURCE_CHECKSUM_MISMATCH", "Checkpoint weights differ from VFAI-014 selection."
        )
    if checkpoint.get("checkpointManifestFileSha256") != sha256_file(
        CHECKPOINT_ROOT / "checkpoint-manifest.json"
    ):
        raise RegistryManagerError(
            "PACKAGE_SOURCE_CHECKSUM_MISMATCH", "Checkpoint manifest differs from VFAI-014 selection."
        )
    if checkpoint_manifest.get("parameterCount") != selection.get("parameterCount"):
        raise RegistryManagerError(
            "PACKAGE_SOURCE_IDENTITY_MISMATCH", "Checkpoint parameter identity is inconsistent."
        )
    if checkpoint_manifest.get("files", {}).get("weights.pt", {}).get("sha256") != checkpoint.get(
        "modelWeightsSha256"
    ):
        raise RegistryManagerError(
            "PACKAGE_SOURCE_CHECKSUM_MISMATCH", "Checkpoint internal weight checksum is inconsistent."
        )
    if selection.get("evidence", {}).get("reportFileSha256") != sha256_file(EVALUATION_PATH):
        raise RegistryManagerError(
            "PACKAGE_SOURCE_CHECKSUM_MISMATCH", "Held-out report differs from VFAI-014 selection."
        )
    if decision.get("evidenceSummary", {}).get("bestBootstrapManifestSha256") != selection.get(
        "manifestSha256"
    ):
        raise RegistryManagerError(
            "PACKAGE_DECISION_INVALID", "VFAI-015 decision is not bound to this checkpoint."
        )
    boundary = decision.get("safeOperatingBoundary", {})
    if (
        boundary.get("VFAI016PackagingPermission") != "experimental-inactive-only"
        or boundary.get("VFAI016ActivationPermission") is not False
        or boundary.get("activeArtifactIdMustRemainNullAtDecision") is not True
    ):
        raise RegistryManagerError(
            "PACKAGE_DECISION_INVALID", "VFAI-015 does not authorize this packaging boundary."
        )
    if (
        tokenizer.get("releaseStatus") != "approved"
        or tokenizer.get("tokenizerId") != "vfdlm-byte-bpe"
        or tokenizer.get("contractVersion") != "1.0.0"
        or tokenizer.get("vocabSize") != selection.get("modelConfig", {}).get("vocabSize")
    ):
        raise RegistryManagerError(
            "PACKAGE_TOKENIZER_INVALID", "Approved tokenizer is incompatible with the selected model."
        )
    return selection, decision, tokenizer


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RegistryManagerError("PACKAGE_SOURCE_MISSING", f"Package source is missing: {source.name}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _build_staging_artifact(
    staging: Path,
    *,
    private_key_path: Path,
) -> None:
    selection, decision, tokenizer = _assert_source_contracts()
    checkpoint_model = CHECKPOINT_ROOT / "model"
    sources = {
        "model/config.json": checkpoint_model / "config.json",
        "model/weights.pt": checkpoint_model / "weights.pt",
        "model/checkpoint-manifest.json": checkpoint_model / "manifest.json",
        "tokenizer/vocab.json": TOKENIZER_ROOT / "vocab.json",
        "tokenizer/merges.json": TOKENIZER_ROOT / "merges.json",
        "tokenizer/tokenizer-config.json": TOKENIZER_ROOT / "tokenizer_config.json",
        "tokenizer/tokenizer-manifest.json": TOKENIZER_ROOT / "tokenizer_manifest.json",
        "tokenizer/evaluation-report.json": TOKENIZER_ROOT / "evaluation-report.json",
        "evaluation/gen1-bootstrap-heldout-v1.json": EVALUATION_PATH,
        "evaluation/gen1-scale-decision-v1.json": DECISION_PATH,
        "provenance/gen1-bootstrap-best-v1.json": SELECTION_PATH,
        "provenance/gen1-scale-policy-v1.json": DECISION_POLICY_PATH,
    }
    for destination, source in sources.items():
        _copy_file(source, staging / destination)

    runtime_contract = RuntimeContract.current()
    compatibility = runtime_contract.as_compatibility()
    compatibility.update(
        {
            "artifactSchemaVersion": 2,
            "registrySchemaVersion": 2,
            "weightDtype": "float32",
            "quantization": "fp32",
            "validatedPlatform": {
                "operatingSystem": os.name,
                "pythonVersion": ".".join(str(value) for value in sys.version_info[:3]),
            },
            "activationRequirements": {
                "signedRegistryRequired": True,
                "signedArtifactManifestRequired": True,
                "releaseStatusRequired": "approved",
                "activationEligibleRequired": True,
                "exactFileSetRequired": True,
            },
        }
    )
    write_json(staging / "compatibility.json", compatibility)

    generation_policy = {
        "schemaVersion": 1,
        "policyId": "vfai016-bootstrap-generation-policy-v1",
        "artifactId": ARTIFACT_ID,
        "releaseStatus": "experimental",
        "servingEnabled": False,
        "activationEligible": False,
        "contextLength": int(selection["modelConfig"]["maxSequenceLength"]),
        "maximumInputTokens": 96,
        "maximumNewTokens": 32,
        "samplingEnabled": False,
        "streamingEnabled": False,
        "deterministicFallbackRequired": True,
        "promptLoggingAllowed": False,
        "reason": (
            "VFAI-015 retained this undertrained checkpoint only as an inactive experimental "
            "baseline; VFAI-017 must implement and evaluate local generation before release."
        ),
    }
    write_json(staging / "generation-policy.json", generation_policy)

    provenance_sources = [
        CHECKPOINT_ROOT / "checkpoint-manifest.json",
        CHECKPOINT_ROOT / "model" / "manifest.json",
        CHECKPOINT_ROOT / "model" / "config.json",
        CHECKPOINT_ROOT / "model" / "weights.pt",
        SELECTION_PATH,
        EVALUATION_PATH,
        DECISION_PATH,
        DECISION_POLICY_PATH,
        TOKENIZER_ROOT / "tokenizer_manifest.json",
    ]
    provenance = {
        "schemaVersion": 1,
        "artifactId": ARTIFACT_ID,
        "trainingRunId": selection["runId"],
        "checkpointStep": selection["step"],
        "candidateId": selection["candidateId"],
        "checkpointSource": _relative_project_path(CHECKPOINT_ROOT),
        "sourceEvidence": [_source_evidence(path) for path in provenance_sources],
        "decision": {
            "reportId": decision["reportId"],
            "reportSha256": decision["reportSha256"],
            "selectedOption": decision["decision"]["selectedOption"],
            "modelDisposition": decision["decision"]["modelDisposition"],
            "activationPermission": False,
        },
        "ownership": {
            "modelWeights": "VoltForge-owned, trained from random initialization",
            "tokenizer": "VoltForge-owned byte-level BPE",
            "hostedModelDependency": False,
            "pretrainedModelDependency": False,
            "pretrainedTokenizerDependency": False,
        },
    }
    write_json(staging / "provenance.json", provenance)

    role_by_path = {
        "model/config.json": "model-config",
        "model/weights.pt": "model-weights",
        "model/checkpoint-manifest.json": "checkpoint-model-manifest",
        "tokenizer/vocab.json": "tokenizer-vocabulary",
        "tokenizer/merges.json": "tokenizer-merges",
        "tokenizer/tokenizer-config.json": "tokenizer-config",
        "tokenizer/tokenizer-manifest.json": "tokenizer-manifest",
        "tokenizer/evaluation-report.json": "tokenizer-evaluation",
        "evaluation/gen1-bootstrap-heldout-v1.json": "model-evaluation",
        "evaluation/gen1-scale-decision-v1.json": "scale-decision",
        "provenance/gen1-bootstrap-best-v1.json": "checkpoint-selection",
        "provenance/gen1-scale-policy-v1.json": "decision-policy",
        "compatibility.json": "runtime-compatibility",
        "generation-policy.json": "generation-policy",
        "provenance.json": "package-provenance",
    }
    files = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        files.append(
            {
                "path": relative,
                "role": role_by_path[relative],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    package_sha256 = sha256_bytes(canonical_json_bytes(files))
    manifest = {
        "schemaVersion": 2,
        "artifactKind": "vfdlm-gen1-local-model-package",
        "artifactId": ARTIFACT_ID,
        "createdAtUtc": utc_now(),
        "immutable": True,
        "runtime": "pytorch-gen1-v1",
        "compatibilityPath": "compatibility.json",
        "generationPolicyPath": "generation-policy.json",
        "provenancePath": "provenance.json",
        "packageSha256": package_sha256,
        "model": {
            "architectureId": selection["modelConfig"]["architectureId"],
            "checkpointFormat": "pytorch-weights-only-state-dict-v1",
            "parameterCount": selection["parameterCount"],
            "trainableParameterCount": selection["parameterCount"],
            "nonTrainableParameterCount": 0,
            "contextLength": selection["modelConfig"]["maxSequenceLength"],
            "vocabSize": selection["modelConfig"]["vocabSize"],
            "quantization": "fp32",
            "weightDtype": "float32",
            "configPath": "model/config.json",
            "weightsPath": "model/weights.pt",
        },
        "tokenizer": {
            "tokenizerId": tokenizer["tokenizerId"],
            "version": tokenizer["version"],
            "contractVersion": tokenizer["contractVersion"],
            "releaseStatus": tokenizer["releaseStatus"],
            "vocabSize": tokenizer["vocabSize"],
            "manifestPath": "tokenizer/tokenizer-manifest.json",
            "artifactSha256": tokenizer["artifactSha256"],
        },
        "evaluation": {
            "reportPath": "evaluation/gen1-bootstrap-heldout-v1.json",
            "reportSha256": selection["evidence"]["reportSha256"],
            "releaseApproved": False,
            "globalMetricsPassing": 0,
            "globalMetricsRequired": 13,
        },
        "releaseState": {
            "releaseStatus": "experimental",
            "activationEligible": False,
            "servingAllowed": False,
            "decision": "retain-experimental-inactive",
            "blockingReason": (
                "Insufficient approved data, failed global model metrics, and no approved "
                "Gen1 generation runtime."
            ),
        },
        "files": files,
    }
    signed_manifest = sign_document(
        manifest,
        digest_field="manifestSha256",
        private_key_path=private_key_path,
        key_id=SIGNING_KEY_ID,
    )
    (staging / "artifact-manifest.json").write_bytes(json_file_bytes(signed_manifest))


def package_artifact(private_key_path: Path) -> dict[str, Any]:
    initialize_signing_key(
        private_key_path,
        DEFAULT_TRUST_STORE_PATH,
        key_id=SIGNING_KEY_ID,
    )
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    final_root = ARTIFACTS_ROOT / ARTIFACT_ID
    if final_root.exists():
        artifact = verify_artifact_directory(
            final_root,
            trust_store_path=DEFAULT_TRUST_STORE_PATH,
        )
    else:
        staging_container = Path(
            tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.staging-", dir=ARTIFACTS_ROOT)
        )
        staging = staging_container / ARTIFACT_ID
        staging.mkdir()
        try:
            _build_staging_artifact(staging, private_key_path=private_key_path)
            # Verification precedes the one atomic directory publication step.
            verify_artifact_directory(staging, trust_store_path=DEFAULT_TRUST_STORE_PATH)
            os.replace(staging, final_root)
            staging_container.rmdir()
        except Exception:
            if staging_container.exists():
                shutil.rmtree(staging_container)
            raise
        artifact = verify_artifact_directory(
            final_root,
            trust_store_path=DEFAULT_TRUST_STORE_PATH,
        )

    registry = register_artifact(
        artifact,
        registry_path=DEFAULT_REGISTRY_PATH,
        trust_store_path=DEFAULT_TRUST_STORE_PATH,
        private_key_path=private_key_path,
        key_id=SIGNING_KEY_ID,
    )
    registry = verify_registry(DEFAULT_REGISTRY_PATH, trust_store_path=DEFAULT_TRUST_STORE_PATH)
    if registry.get("activeArtifactId") is not None:
        raise RegistryManagerError(
            "PACKAGE_ACTIVATION_BOUNDARY_VIOLATION",
            "VFAI-016 must leave activeArtifactId null.",
        )
    manifest_mirror = MANIFESTS_ROOT / f"{ARTIFACT_ID}.json"
    manifest_bytes = (final_root / "artifact-manifest.json").read_bytes()
    if manifest_mirror.exists() and manifest_mirror.read_bytes() != manifest_bytes:
        raise RegistryManagerError(
            "PACKAGE_MANIFEST_MIRROR_CONFLICT",
            "Tracked artifact manifest mirror differs from the immutable local package.",
        )
    if not manifest_mirror.exists():
        manifest_mirror.parent.mkdir(parents=True, exist_ok=True)
        manifest_mirror.write_bytes(manifest_bytes)
    entry = next(item for item in registry["artifacts"] if item["artifactId"] == ARTIFACT_ID)
    report = {
        "schemaVersion": 1,
        "reportId": "vfai016-gen1-package-v1",
        "generatedAtUtc": utc_now(),
        "artifactId": ARTIFACT_ID,
        "artifactRoot": _relative_project_path(final_root),
        "artifactDirectoryTracked": False,
        "manifestSha256": artifact.manifest_sha256,
        "manifestFileSha256": artifact.manifest_file_sha256,
        "manifestMirrorPath": _relative_project_path(manifest_mirror),
        "packageBytes": artifact.package_bytes,
        "fileCount": len(artifact.manifest["files"]) + 1,
        "signature": deepcopy(artifact.manifest["signature"]),
        "verification": {
            "signatureVerified": True,
            "exactFileSetVerified": True,
            "allChecksumsVerified": True,
            "compatibilityVerified": True,
            "releaseStatus": artifact.release_status,
            "activationEligible": artifact.activation_eligible,
            "active": False,
        },
        "registry": {
            "path": _relative_project_path(DEFAULT_REGISTRY_PATH),
            "schemaVersion": registry["schemaVersion"],
            "revision": registry["revision"],
            "registrySha256": registry["registrySha256"],
            "signature": deepcopy(registry["signature"]),
            "activeArtifactId": registry["activeArtifactId"],
            "catalogEntry": entry,
        },
        "sourceEvidence": {
            "selection": _source_evidence(SELECTION_PATH),
            "evaluation": _source_evidence(EVALUATION_PATH),
            "scaleDecision": _source_evidence(DECISION_PATH),
            "checkpointManifest": _source_evidence(
                CHECKPOINT_ROOT / "checkpoint-manifest.json"
            ),
            "weights": _source_evidence(CHECKPOINT_ROOT / "model" / "weights.pt"),
            "tokenizerManifest": _source_evidence(TOKENIZER_ROOT / "tokenizer_manifest.json"),
        },
        "operatorBoundary": (
            "This report proves packaging integrity only. The artifact is experimental and "
            "must not be activated or served."
        ),
    }
    report["reportSha256"] = sha256_bytes(canonical_json_bytes(report))
    report_path = REPORTS_ROOT / f"{ARTIFACT_ID}-package.json"
    write_json(report_path, report)
    return report


def verify_package() -> dict[str, Any]:
    artifact = verify_artifact_directory(
        ARTIFACTS_ROOT / ARTIFACT_ID,
        trust_store_path=DEFAULT_TRUST_STORE_PATH,
    )
    registry = verify_registry(DEFAULT_REGISTRY_PATH, trust_store_path=DEFAULT_TRUST_STORE_PATH)
    entry = next(
        (item for item in registry["artifacts"] if item.get("artifactId") == ARTIFACT_ID),
        None,
    )
    if entry is None:
        raise RegistryManagerError("ACTIVE_MODEL_ARTIFACT_NOT_FOUND", "Package is not cataloged.")
    if entry["manifestSha256"] != artifact.manifest_sha256:
        raise RegistryManagerError(
            "MODEL_REGISTRY_ARTIFACT_MISMATCH", "Registry does not match the artifact manifest."
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
        result = package_artifact(args.private_key) if args.command == "package" else verify_package()
    except RegistryManagerError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": error.message}, indent=2))
        return 1
    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
