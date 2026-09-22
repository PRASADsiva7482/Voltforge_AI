"""Derive the signed VFAI-018 optimized runtime artifact immutably."""

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

from gen1_optimization.benchmark import (  # noqa: E402
    POLICY_PATH,
    REPORT_PATH as BENCHMARK_REPORT_PATH,
    build_policy,
    validate_report,
)
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


SOURCE_ARTIFACT_ID = "vfdlm-g1-edge-v0.1.1-runtime"
ARTIFACT_ID = "vfdlm-g1-edge-v0.1.2-optimized"
SIGNING_KEY_ID = "vf-local-registry-dev-2026-08"
DEFAULT_PRIVATE_KEY = PROJECT_ROOT / ".toolchains" / "registry-signing" / "vf-local-ed25519.pem"
REGISTRY_ROOT = PROJECT_ROOT / "model" / "registry"
ARTIFACTS_ROOT = REGISTRY_ROOT / "artifacts"
SOURCE_ROOT = ARTIFACTS_ROOT / SOURCE_ARTIFACT_ID
FINAL_ROOT = ARTIFACTS_ROOT / ARTIFACT_ID
MANIFEST_MIRROR = REGISTRY_ROOT / "manifests" / f"{ARTIFACT_ID}.json"
REPORT_PATH = REGISTRY_ROOT / "reports" / f"{ARTIFACT_ID}-package.json"
PACKAGED_BENCHMARK_PATH = "benchmarks/reports/gen1-inference-optimization-v1.json"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _copy_source_payload(staging: Path, source_manifest: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    replaced = {"generation-policy.json", "provenance.json"}
    for descriptor in source_manifest["files"]:
        relative = descriptor["path"]
        if relative in replaced:
            continue
        source = SOURCE_ROOT / Path(*relative.split("/"))
        destination = staging / Path(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        roles[relative] = descriptor["role"]
    return roles


def _build(staging: Path, private_key_path: Path) -> None:
    source = verify_artifact_directory(SOURCE_ROOT, trust_store_path=DEFAULT_TRUST_STORE_PATH)
    if source.release_status != "experimental" or source.activation_eligible:
        raise RegistryManagerError(
            "OPTIMIZED_PACKAGE_SOURCE_INVALID",
            "Optimized package source must remain experimental and inactive.",
        )
    benchmark = read_json(BENCHMARK_REPORT_PATH, "OPTIMIZATION_REPORT_INVALID")
    validate_report(benchmark)
    policy = read_json(POLICY_PATH, "OPTIMIZATION_POLICY_INVALID")
    if policy != build_policy(benchmark):
        raise RegistryManagerError(
            "OPTIMIZATION_POLICY_MISMATCH",
            "Optimization policy is not bound to the verified benchmark report.",
        )
    selected = benchmark["selection"]["selectedProfile"]
    if selected["quantization"] != "fp32":
        raise RegistryManagerError(
            "OPTIMIZED_PACKAGE_FORMAT_INVALID",
            "Only the portable FP32 format may be packaged by VFAI-018.",
        )

    source_manifest = dict(source.manifest)
    roles = _copy_source_payload(staging, source_manifest)
    generation_policy = read_json(
        SOURCE_ROOT / "generation-policy.json", "GENERATION_POLICY_INVALID"
    )
    generation_policy.update(
        {
            "artifactId": ARTIFACT_ID,
            "policyId": "vfai018-experimental-optimized-runtime-policy-v1",
            "reason": (
                "VFAI-018 selected a measured quality-preserving FP32 runtime profile. "
                "The undertrained model remains offline-only and activation-ineligible."
            ),
        }
    )
    write_json(staging / "generation-policy.json", generation_policy)
    roles["generation-policy.json"] = "generation-policy"

    shutil.copy2(POLICY_PATH, staging / "optimization-policy.json")
    roles["optimization-policy.json"] = "runtime-optimization-policy"
    packaged_benchmark = staging / Path(*PACKAGED_BENCHMARK_PATH.split("/"))
    packaged_benchmark.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BENCHMARK_REPORT_PATH, packaged_benchmark)
    roles[PACKAGED_BENCHMARK_PATH] = "runtime-optimization-evaluation"

    provenance = read_json(SOURCE_ROOT / "provenance.json", "PROVENANCE_INVALID")
    provenance = deepcopy(provenance)
    provenance.update(
        {
            "schemaVersion": 3,
            "artifactId": ARTIFACT_ID,
            "parentArtifact": {
                "artifactId": SOURCE_ARTIFACT_ID,
                "manifestSha256": source.manifest_sha256,
                "manifestFileSha256": source.manifest_file_sha256,
            },
            "optimizationRevision": {
                "item": "VFAI-018",
                "benchmarkReportSha256": benchmark["reportSha256"],
                "optimizationPolicySha256": policy["policySha256"],
                "selectedProfileId": selected["profileId"],
                "weightsChanged": False,
                "tokenizerContentChanged": False,
                "quantizedFormatReleased": False,
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
            "optimizationPolicyPath": "optimization-policy.json",
            "packageSha256": sha256_bytes(canonical_json_bytes(files)),
            "files": files,
        }
    )
    manifest["model"] = deepcopy(source_manifest["model"])
    manifest["model"]["quantization"] = "fp32"
    manifest["model"]["weightDtype"] = "float32"
    manifest["releaseState"] = {
        "releaseStatus": "experimental",
        "activationEligible": False,
        "servingAllowed": False,
        "offlineExperimentalInferenceAllowed": True,
        "decision": "optimized-runtime-ready-for-offline-evaluation-only",
        "blockingReason": (
            "VFAI-018 preserves runtime quality but cannot overcome insufficient training "
            "data, failed global metrics, or missing VFAI-019 output gates."
        ),
    }
    signed = sign_document(
        manifest,
        digest_field="manifestSha256",
        private_key_path=private_key_path,
        key_id=SIGNING_KEY_ID,
    )
    (staging / "artifact-manifest.json").write_bytes(json_file_bytes(signed))


def _verify_evidence_bindings(artifact) -> None:
    if artifact.manifest.get("optimizationPolicyPath") != "optimization-policy.json":
        raise RegistryManagerError(
            "OPTIMIZATION_POLICY_MISMATCH",
            "Optimized artifact does not declare the canonical optimization policy.",
        )
    if (artifact.root / "optimization-policy.json").read_bytes() != POLICY_PATH.read_bytes():
        raise RegistryManagerError(
            "OPTIMIZATION_POLICY_MISMATCH",
            "Optimized artifact policy differs from the measured workspace policy.",
        )
    packaged_report = artifact.root / Path(*PACKAGED_BENCHMARK_PATH.split("/"))
    if packaged_report.read_bytes() != BENCHMARK_REPORT_PATH.read_bytes():
        raise RegistryManagerError(
            "OPTIMIZATION_REPORT_MISMATCH",
            "Optimized artifact benchmark differs from the verified workspace report.",
        )


def package(private_key_path: Path) -> dict[str, Any]:
    initialize_signing_key(
        private_key_path, DEFAULT_TRUST_STORE_PATH, key_id=SIGNING_KEY_ID
    )
    if FINAL_ROOT.exists():
        artifact = verify_artifact_directory(
            FINAL_ROOT, trust_store_path=DEFAULT_TRUST_STORE_PATH
        )
    else:
        container = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.staging-", dir=ARTIFACTS_ROOT))
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

    _verify_evidence_bindings(artifact)

    register_artifact(
        artifact,
        registry_path=DEFAULT_REGISTRY_PATH,
        trust_store_path=DEFAULT_TRUST_STORE_PATH,
        private_key_path=private_key_path,
        key_id=SIGNING_KEY_ID,
    )
    registry = verify_registry(DEFAULT_REGISTRY_PATH, trust_store_path=DEFAULT_TRUST_STORE_PATH)
    if registry["activeArtifactId"] is not None:
        raise RegistryManagerError(
            "OPTIMIZED_PACKAGE_ACTIVATION_BOUNDARY_VIOLATION",
            "VFAI-018 must leave the optimized artifact inactive.",
        )
    manifest_bytes = (FINAL_ROOT / "artifact-manifest.json").read_bytes()
    if MANIFEST_MIRROR.exists() and MANIFEST_MIRROR.read_bytes() != manifest_bytes:
        raise RegistryManagerError(
            "OPTIMIZED_PACKAGE_MANIFEST_CONFLICT",
            "Tracked optimized manifest differs from the immutable package.",
        )
    if not MANIFEST_MIRROR.exists():
        MANIFEST_MIRROR.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_MIRROR.write_bytes(manifest_bytes)
    entry = next(item for item in registry["artifacts"] if item["artifactId"] == ARTIFACT_ID)
    benchmark = read_json(BENCHMARK_REPORT_PATH, "OPTIMIZATION_REPORT_INVALID")
    policy = read_json(POLICY_PATH, "OPTIMIZATION_POLICY_INVALID")
    report = {
        "schemaVersion": 1,
        "reportId": "vfai018-optimized-package-v1",
        "generatedAtUtc": utc_now(),
        "artifactId": ARTIFACT_ID,
        "parentArtifactId": SOURCE_ARTIFACT_ID,
        "artifactRoot": _relative(FINAL_ROOT),
        "manifestMirrorPath": _relative(MANIFEST_MIRROR),
        "manifestSha256": artifact.manifest_sha256,
        "manifestFileSha256": artifact.manifest_file_sha256,
        "packageBytes": artifact.package_bytes,
        "fileCount": len(artifact.manifest["files"]) + 1,
        "selectedProfileId": benchmark["selection"]["selectedProfile"]["profileId"],
        "fallbackProfileId": benchmark["selection"]["fallbackProfile"]["profileId"],
        "benchmarkReportSha256": benchmark["reportSha256"],
        "optimizationPolicySha256": policy["policySha256"],
        "weightsChanged": False,
        "tokenizerContentChanged": False,
        "quantizedFormatReleased": False,
        "verification": {
            "signatureVerified": True,
            "exactFileSetVerified": True,
            "allChecksumsVerified": True,
            "optimizationEvidenceBound": True,
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
    artifact = verify_artifact_directory(FINAL_ROOT, trust_store_path=DEFAULT_TRUST_STORE_PATH)
    _verify_evidence_bindings(artifact)
    registry = verify_registry(DEFAULT_REGISTRY_PATH, trust_store_path=DEFAULT_TRUST_STORE_PATH)
    entry = next(
        (item for item in registry["artifacts"] if item.get("artifactId") == ARTIFACT_ID), None
    )
    if entry is None or entry["manifestSha256"] != artifact.manifest_sha256:
        raise RegistryManagerError(
            "MODEL_REGISTRY_ARTIFACT_MISMATCH", "Optimized package is not cataloged exactly."
        )
    policy = read_json(FINAL_ROOT / "optimization-policy.json", "OPTIMIZATION_POLICY_INVALID")
    return {
        "artifactId": artifact.artifact_id,
        "manifestSha256": artifact.manifest_sha256,
        "packageBytes": artifact.package_bytes,
        "selectedProfileId": policy["selectedProfileId"],
        "fallbackProfileId": policy["fallbackProfileId"],
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
