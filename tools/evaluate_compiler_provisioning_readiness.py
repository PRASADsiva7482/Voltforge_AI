"""Generate or verify the content-free VFAI-FU-014 readiness receipt.

This evaluator audits installed compiler identities, retained receipt counts,
and the local download-cache shape. It never downloads, exports, provisions,
compiles, rewrites receipts, or claims clean-host acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import zipfile


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from compiler_provisioning import (  # noqa: E402
    CompilerCacheContractError,
    cache_key_from_inputs,
    verify_cache_export,
)
from synthetic_data.pipeline import check_release, verify_pipeline_lock  # noqa: E402
from synthetic_data.verifiers import (  # noqa: E402
    SyntheticVerificationError,
    reset_toolchain_identity_cache,
    toolchain_identity,
)


MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "compiler_provisioning/policy.v1.json"
TOOLCHAIN_MANIFEST_PATH = AI_ROOT / "synthetic_data/toolchains.v1.json"
PIPELINE_LOCK_PATH = AI_ROOT / "synthetic_data/pipeline-lock.v1.json"
GENERATION_REPORT_PATH = AI_ROOT / "synthetic_data/reports/current-generation-report.json"
COMPILE_RECEIPTS_PATH = AI_ROOT / "synthetic_data/receipts/v1/compile-receipts.jsonl"
DOWNLOAD_ROOT = AI_ROOT / ".toolchains/downloads"
CLI_CONFIG_PATH = AI_ROOT / ".toolchains/arduino-cli.yaml"
PACKAGE_INDEX_PATHS = (
    AI_ROOT / ".toolchains/arduino-data/package_index.json",
    AI_ROOT / ".toolchains/arduino-data/package_rp2040_index.json",
)
CLI_ARCHIVE_NAME = "arduino-cli_1.5.1_Windows_64bit.zip"
CACHE_CONTRACT_PATH = AI_ROOT / "compiler_provisioning/cache_contract.py"
DEFAULT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/compiler-provisioning-readiness-v1.json"
)
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})
EXPECTED_PROFILES = frozenset(
    {
        "arduino-avr-1.8.6",
        "arduino-megaavr-1.8.8",
        "arduino-renesas-uno-1.6.0",
        "esp32-3.3.11",
        "pico-6.0.0",
    }
)


class CompilerProvisioningReadinessError(RuntimeError):
    """VFAI-FU-014 evidence is invalid, unsafe, or stale."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _declared_digest_valid(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        return False
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == declared


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompilerProvisioningReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompilerProvisioningReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _backlog_item(backlog: Mapping[str, Any]) -> Mapping[str, Any]:
    items = backlog.get("items")
    matches = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("id") == "VFAI-FU-014"
    ] if isinstance(items, list) else []
    if len(matches) != 1:
        raise CompilerProvisioningReadinessError("expected exactly one VFAI-FU-014 item")
    return matches[0]


def _master_complete(master: Mapping[str, Any]) -> bool:
    items = master.get("items")
    return bool(
        master.get("status") in COMPLETE_STATUSES
        and isinstance(items, list)
        and items
        and all(
            isinstance(item, Mapping) and item.get("status") in COMPLETE_STATUSES
            for item in items
        )
    )


def _resolve_optional_path(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CompilerProvisioningReadinessError(f"{label} must be null or a path")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise CompilerProvisioningReadinessError(f"{label} escapes the AI workspace") from exc
    return path if path.is_file() else None


def _resolve_optional_directory(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CompilerProvisioningReadinessError(f"{label} must be null or a path")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise CompilerProvisioningReadinessError(f"{label} escapes the AI workspace") from exc
    return path if path.is_dir() else None


def _toolchain_source_evidence(
    manifest: Mapping[str, Any], pipeline_lock: Mapping[str, Any]
) -> dict[str, Any]:
    profiles = manifest.get("profiles")
    fqbns = [
        fqbn
        for profile in profiles.values()
        if isinstance(profile, Mapping)
        for fqbn in profile.get("fqbn", [])
    ] if isinstance(profiles, Mapping) else []
    lock_profiles = pipeline_lock.get("toolchains")
    ready = bool(
        manifest.get("schemaVersion") == 1
        and manifest.get("manifestId") == "vf-compiler-toolchain-manifest-v1"
        and manifest.get("networkPolicy") == "offline-after-install"
        and isinstance(profiles, Mapping)
        and set(profiles) == EXPECTED_PROFILES
        and len(fqbns) == len(set(fqbns)) == 11
        and isinstance(lock_profiles, Mapping)
        and set(lock_profiles) == EXPECTED_PROFILES
    )
    return {
        "ready": ready,
        "manifestId": manifest.get("manifestId"),
        "manifestVersion": manifest.get("manifestVersion"),
        "host": manifest.get("host"),
        "networkPolicy": manifest.get("networkPolicy"),
        "profileCount": len(profiles) if isinstance(profiles, Mapping) else 0,
        "exactFqbnCount": len(fqbns),
        "profileIds": sorted(profiles) if isinstance(profiles, Mapping) else [],
        "pipelineLockSha256": pipeline_lock.get("lockSha256"),
    }


def _installed_identity_evidence(manifest: Mapping[str, Any]) -> dict[str, Any]:
    reset_toolchain_identity_cache()
    verified: list[str] = []
    failures: list[dict[str, str]] = []
    for profile_id in sorted(EXPECTED_PROFILES):
        try:
            identity = toolchain_identity(profile_id, require_files=True)
        except SyntheticVerificationError as exc:
            failures.append({"profileId": profile_id, "error": str(exc)})
            continue
        if identity.get("profileId") == profile_id:
            verified.append(profile_id)
    return {
        "ready": len(verified) == len(EXPECTED_PROFILES) and not failures,
        "verifiedProfileCount": len(verified),
        "verifiedProfileIds": verified,
        "failedProfiles": failures,
        "arduinoCliVersion": manifest.get("arduinoCli", {}).get("version"),
        "arduinoCliExecutableSha256": manifest.get("arduinoCli", {}).get("sha256"),
    }


def _receipt_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    compile_report = report.get("compileVerification", {})
    release_decision = report.get("summary", {}).get("decision")
    ready = bool(
        release_decision == "pass"
        and compile_report.get("receiptCount") == 88
        and compile_report.get("acceptedCompileCount") == 66
        and compile_report.get("expectedFailureCount") == 22
    )
    return {
        "ready": ready,
        "releaseDecision": release_decision,
        "receiptCount": compile_report.get("receiptCount"),
        "acceptedCompileCount": compile_report.get("acceptedCompileCount"),
        "expectedFailureCount": compile_report.get("expectedFailureCount"),
        "freshCompilationPerformed": False,
    }


def _zip_member_sha256(path: Path, suffix: str) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if name.lower().endswith(suffix)]
            if len(matches) != 1:
                return None
            digest = hashlib.sha256()
            with archive.open(matches[0]) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    except (OSError, zipfile.BadZipFile):
        return None


def _cache_observation(
    manifest: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    files = sorted(path for path in DOWNLOAD_ROOT.rglob("*") if path.is_file())
    by_name: dict[str, list[Path]] = {}
    for path in files:
        by_name.setdefault(path.name, []).append(path)
    declared = [
        dict(item)
        for profile in manifest.get("profiles", {}).values()
        if isinstance(profile, Mapping)
        for item in profile.get("packageArchives", [])
        if isinstance(item, Mapping)
    ]
    matching: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for item in declared:
        name = str(item.get("name"))
        candidates = by_name.get(name, [])
        if len(candidates) != 1:
            missing.append(name)
        elif _sha256_file(candidates[0]) == item.get("sha256"):
            matching.append(name)
        else:
            mismatched.append(name)
    cli_archive = DOWNLOAD_ROOT / CLI_ARCHIVE_NAME
    cli_member_sha = (
        _zip_member_sha256(cli_archive, "/arduino-cli.exe")
        or _zip_member_sha256(cli_archive, "arduino-cli.exe")
        if cli_archive.is_file()
        else None
    )
    index_descriptors = [
        {
            "path": path.relative_to(AI_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "boundByToolchainManifest": False,
        }
        for path in PACKAGE_INDEX_PATHS
    ]
    command_contract = policy.get("cacheContract", {}).get("cliCommandContract")
    if not isinstance(command_contract, list):
        command_contract = [
            "arduino-cli",
            "--config-file",
            "<CONFIG>",
            "compile",
            "--jobs",
            "0",
            "--fqbn",
            "<EXACT_FQBN>",
            "--warnings",
            "all",
            "--build-path",
            "<ISOLATED_BUILD_PATH>",
            "<SKETCH_PATH>",
        ]
    manifest_sha = _sha256_file(TOOLCHAIN_MANIFEST_PATH)
    index_sha = [item["sha256"] for item in index_descriptors]
    draft_keys = {}
    for profile_id, profile in sorted(manifest.get("profiles", {}).items()):
        key_inputs = {
            "toolchainManifestSha256": manifest_sha,
            "profileId": profile_id,
            "coreId": profile.get("coreId"),
            "coreVersion": profile.get("coreVersion"),
            "fqbns": sorted(profile.get("fqbn", [])),
            "host": policy.get("scope", {}).get("supportedHost"),
            "arduinoCliVersion": manifest.get("arduinoCli", {}).get("version"),
            "arduinoCliExecutableSha256": manifest.get("arduinoCli", {}).get("sha256"),
            "cliCommandContract": command_contract,
            "packageIndexSha256": index_sha,
            "dependencyArchives": sorted(
                profile.get("packageArchives", []), key=lambda item: item.get("name", "")
            ),
            "source": profile.get("source"),
        }
        draft_keys[profile_id] = cache_key_from_inputs(key_inputs)
    pico = manifest.get("profiles", {}).get("pico-6.0.0", {})
    input_path = policy.get("inputClosure", {}).get("provisioningInputManifestPath")
    return {
        "downloadFileCount": len(files),
        "downloadBytes": sum(path.stat().st_size for path in files),
        "declaredArchiveCount": len(declared),
        "matchingDeclaredArchiveCount": len(matching),
        "missingDeclaredArchives": sorted(missing),
        "mismatchedDeclaredArchives": sorted(mismatched),
        "arduinoCliArchivePresent": cli_archive.is_file(),
        "arduinoCliArchiveBoundByToolchainManifest": False,
        "arduinoCliArchiveExecutableMatchesInstalledIdentity": cli_member_sha
        == manifest.get("arduinoCli", {}).get("sha256"),
        "packageIndexes": index_descriptors,
        "packageIndexesBoundByToolchainManifest": False,
        "picoDeclaredPackageArchiveCount": len(pico.get("packageArchives", [])),
        "picoSourceFallbackBundleBound": False,
        "provisioningInputManifestConfigured": isinstance(input_path, str) and bool(input_path),
        "currentDownloadsArePortableCacheExport": False,
        "draftProfileCacheKeys": draft_keys,
        "draftKeysAuthorizeExport": False,
    }


def _optional_receipt(
    value: Any,
    *,
    label: str,
    expected_kind: str,
) -> dict[str, Any]:
    path = _resolve_optional_path(value, label)
    if path is None:
        return {"ready": False, "path": None}
    receipt = _read_json(path, label)
    ready = bool(
        receipt.get("manifestKind") == expected_kind
        and _declared_digest_valid(receipt, "reportSha256")
    )
    return {
        "ready": ready,
        "path": path.relative_to(AI_ROOT).as_posix(),
        "reportSha256": receipt.get("reportSha256"),
        "receipt": receipt,
    }


def _completion_evidence(
    policy: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    input_receipt = _optional_receipt(
        policy.get("inputClosure", {}).get("provisioningInputManifestPath"),
        label="provisioning input manifest",
        expected_kind="vfai-fu-014-provisioning-input-closure-v1",
    )
    input_value = input_receipt.get("receipt", {})
    expected_profile_ids = sorted(manifest.get("profiles", {}))
    input_ready = bool(
        input_receipt.get("ready") is True
        and input_value.get("status") == "complete"
        and input_value.get("toolchainManifestSha256")
        == _sha256_file(TOOLCHAIN_MANIFEST_PATH)
        and input_value.get("supportedHost")
        == policy.get("scope", {}).get("supportedHost")
        and input_value.get("profileIds") == expected_profile_ids
        and input_value.get("profileCount")
        == policy.get("scope", {}).get("requiredProfileCount")
        and input_value.get("exactFqbnCount")
        == policy.get("scope", {}).get("requiredExactFqbnCount")
        and all(
            input_value.get(field) is True
            for field in (
                "allArchiveBytesChecksumVerified",
                "allPackageIndexesChecksumBound",
                "allSourceUrlsRecorded",
                "allLicensesRecorded",
                "completeHostDependencyClosure",
                "picoSourceFallbackBundleBound",
                "postInstallIdentityContractBound",
            )
        )
    )
    budget = _optional_receipt(
        policy.get("authorization", {}).get("ownerBudgetReceiptPath"),
        label="owner provisioning budget receipt",
        expected_kind="vfai-fu-014-owner-budget-v1",
    )
    budget_receipt = budget.get("receipt", {})
    required_budgets = policy.get("authorization", {}).get("requiredPositiveBudgets", [])
    budget_ready = bool(
        budget.get("ready") is True
        and budget_receipt.get("status") == "approved-by-owner"
        and all(
            isinstance(budget_receipt.get(name), (int, float))
            and not isinstance(budget_receipt.get(name), bool)
            and budget_receipt[name] > 0
            for name in required_budgets
        )
    )
    cache_contract = policy.get("cacheContract", {})
    cache_path_values = cache_contract.get("cacheExportManifestPaths")
    if not isinstance(cache_path_values, list):
        raise CompilerProvisioningReadinessError(
            "cache export manifest paths must be an array"
        )
    cache_paths = [
        _resolve_optional_path(value, "cache export manifest")
        for value in cache_path_values
    ]
    cache_summaries: list[dict[str, Any]] = []
    verified_cache_profiles: set[str] = set()
    for cache_manifest_path in cache_paths:
        if cache_manifest_path is None:
            continue
        cache_manifest = _read_json(cache_manifest_path, "cache export manifest")
        cache_root = _resolve_optional_directory(
            cache_manifest.get("cacheRootPath"), "cache root"
        )
        key_inputs = cache_manifest.get("keyInputs", {})
        profile_id = key_inputs.get("profileId") if isinstance(key_inputs, Mapping) else None
        profile = manifest.get("profiles", {}).get(profile_id, {})
        expected_binding = bool(
            isinstance(profile_id, str)
            and profile_id in expected_profile_ids
            and isinstance(profile, Mapping)
            and key_inputs.get("toolchainManifestSha256")
            == _sha256_file(TOOLCHAIN_MANIFEST_PATH)
            and key_inputs.get("host") == policy.get("scope", {}).get("supportedHost")
            and key_inputs.get("coreId") == profile.get("coreId")
            and key_inputs.get("coreVersion") == profile.get("coreVersion")
            and key_inputs.get("fqbns") == sorted(profile.get("fqbn", []))
            and key_inputs.get("arduinoCliVersion")
            == manifest.get("arduinoCli", {}).get("version")
            and key_inputs.get("arduinoCliExecutableSha256")
            == manifest.get("arduinoCli", {}).get("sha256")
            and key_inputs.get("cliCommandContract")
            == cache_contract.get("cliCommandContract")
            and key_inputs.get("dependencyArchives")
            == sorted(
                profile.get("packageArchives", []), key=lambda item: item.get("name", "")
            )
            and key_inputs.get("source") == profile.get("source")
            and isinstance(key_inputs.get("packageIndexSha256"), list)
            and bool(key_inputs.get("packageIndexSha256"))
        )
        if cache_root is None or not expected_binding:
            continue
        try:
            summary = verify_cache_export(cache_manifest, cache_root)
        except CompilerCacheContractError:
            continue
        if profile_id in verified_cache_profiles:
            continue
        verified_cache_profiles.add(profile_id)
        cache_summaries.append({"profileId": profile_id, **summary})
    required_cache_profiles = set(cache_contract.get("requiredProfileIds", []))
    cache_ready = bool(
        required_cache_profiles == set(expected_profile_ids)
        and verified_cache_profiles == required_cache_profiles
        and len(cache_paths) == len(required_cache_profiles)
    )
    clean_host = _optional_receipt(
        policy.get("completionInputs", {}).get("cleanHostProvisioningReceiptPath"),
        label="clean-host provisioning receipt",
        expected_kind="vfai-fu-014-clean-host-provisioning-v1",
    )
    restore = _optional_receipt(
        policy.get("completionInputs", {}).get("offlineCacheRestoreReceiptPath"),
        label="offline cache-restore receipt",
        expected_kind="vfai-fu-014-offline-cache-restore-v1",
    )
    clean_host_value = clean_host.get("receipt", {})
    clean_host_ready = bool(
        clean_host.get("ready") is True
        and clean_host_value.get("status") == "pass"
        and clean_host_value.get("supportedHost")
        == policy.get("scope", {}).get("supportedHost")
        and clean_host_value.get("verifiedProfileCount")
        == policy.get("scope", {}).get("requiredProfileCount")
        and clean_host_value.get("verifiedExactFqbnCount")
        == policy.get("scope", {}).get("requiredExactFqbnCount")
        and clean_host_value.get("receiptCount")
        == policy.get("scope", {}).get("requiredCompileReceiptCount")
        and clean_host_value.get("acceptedCompileCount")
        == policy.get("scope", {}).get("requiredSuccessfulCompileCount")
        and clean_host_value.get("expectedFailureCount")
        == policy.get("scope", {}).get("requiredExpectedFailureCount")
        and clean_host_value.get("postInstallIdentityVerified") is True
        and clean_host_value.get("retainedReceiptsMatched") is True
        and clean_host_value.get("networkAccessDuringCompilation") is False
    )
    restore_value = restore.get("receipt", {})
    restore_ready = bool(
        restore.get("ready") is True
        and restore_value.get("status") == "pass"
        and restore_value.get("supportedHost")
        == policy.get("scope", {}).get("supportedHost")
        and restore_value.get("verifiedProfileCount")
        == policy.get("scope", {}).get("requiredProfileCount")
        and restore_value.get("receiptCount")
        == policy.get("scope", {}).get("requiredCompileReceiptCount")
        and restore_value.get("cacheKeysVerified") is True
        and restore_value.get("retainedReceiptsMatched") is True
        and restore_value.get("networkAccessDuringCompilation") is False
        and set(restore_value.get("verifiedFailureCases", []))
        == set(policy.get("completionInputs", {}).get("requiredCacheFailureCases", []))
    )
    return {
        "provisioningInputClosureReady": input_ready,
        "ownerBudgetReady": budget_ready,
        "cacheExportReady": cache_ready,
        "cacheVerification": cache_summaries,
        "cleanHostProvisioningReady": clean_host_ready,
        "offlineCacheRestoreReady": restore_ready,
        "configuredPaths": {
            "provisioningInputManifest": input_receipt.get("path"),
            "ownerBudgetReceipt": budget.get("path"),
            "cacheExportManifests": [
                path.relative_to(AI_ROOT).as_posix() if path else None
                for path in cache_paths
            ],
            "cleanHostProvisioningReceipt": clean_host.get("path"),
            "offlineCacheRestoreReceipt": restore.get("path"),
        },
    }


def _expected_decision(gates: Mapping[str, Any]) -> str:
    if gates.get("masterBacklogComplete") is not True:
        return "await-master-backlog-completion"
    if gates.get("sourceEvidenceReady") is not True:
        return "repair-toolchain-source-evidence"
    if gates.get("installedIdentityReady") is not True:
        return "repair-installed-toolchain-identity"
    if gates.get("retainedReceiptsReady") is not True:
        return "repair-retained-compiler-receipts"
    if gates.get("provisioningInputClosureReady") is not True:
        return "complete-checksummed-provisioning-input-closure"
    if gates.get("ownerBudgetReady") is not True:
        return "await-owner-provisioning-and-cache-budget"
    if gates.get("cacheExportReady") is not True:
        return "create-content-addressed-offline-cache-export"
    if gates.get("cleanHostProvisioningReady") is not True:
        return "await-approved-clean-host-provisioning"
    if gates.get("offlineCacheRestoreReady") is not True:
        return "await-offline-cache-restore-verification"
    return "clean-host-results-ready-for-independent-review"


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    installed: Mapping[str, Any],
    receipts: Mapping[str, Any],
    cache: Mapping[str, Any],
    completion: Mapping[str, Any],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-014":
        raise CompilerProvisioningReadinessError("policy is not bound to VFAI-FU-014")
    item = _backlog_item(followup_backlog)
    gates = {
        "masterBacklogComplete": _master_complete(master_backlog),
        "sourceEvidenceReady": source.get("ready") is True,
        "installedIdentityReady": installed.get("ready") is True,
        "retainedReceiptsReady": receipts.get("ready") is True,
        "provisioningInputClosureReady": completion.get(
            "provisioningInputClosureReady"
        ) is True,
        "ownerBudgetReady": completion.get("ownerBudgetReady") is True,
        "cacheExportReady": completion.get("cacheExportReady") is True,
        "cleanHostProvisioningReady": completion.get(
            "cleanHostProvisioningReady"
        ) is True,
        "offlineCacheRestoreReady": completion.get("offlineCacheRestoreReady") is True,
    }
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-014-compiler-provisioning-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": item.get("id"), "recordedStatus": item.get("status")},
        "policy": {"policyId": policy.get("policyId"), "status": policy.get("status")},
        "toolchainSource": dict(source),
        "installedIdentity": dict(installed),
        "retainedReceipts": dict(receipts),
        "localCacheObservation": dict(cache),
        "completionInputs": dict(completion),
        "gates": gates,
        "decision": _expected_decision(gates),
        "completionClaimed": False,
        "networkAccessed": False,
        "archivesDownloadedByEvaluator": 0,
        "cacheExportsCreatedByEvaluator": 0,
        "hostsProvisionedByEvaluator": 0,
        "compilerRunsExecutedByEvaluator": 0,
        "compileReceiptsRewrittenByEvaluator": False,
        "syntheticDataModifiedByEvaluator": False,
        "existingToolchainsModifiedByEvaluator": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _report_digest(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _report_digest(report):
        raise CompilerProvisioningReadinessError("VFAI-FU-014 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or report.get("decision") != _expected_decision(gates):
        raise CompilerProvisioningReadinessError("VFAI-FU-014 receipt decision is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("networkAccessed") is not False,
            report.get("archivesDownloadedByEvaluator") != 0,
            report.get("cacheExportsCreatedByEvaluator") != 0,
            report.get("hostsProvisionedByEvaluator") != 0,
            report.get("compilerRunsExecutedByEvaluator") != 0,
            report.get("compileReceiptsRewrittenByEvaluator") is not False,
            report.get("syntheticDataModifiedByEvaluator") is not False,
            report.get("existingToolchainsModifiedByEvaluator") is not False,
        )
    ):
        raise CompilerProvisioningReadinessError("VFAI-FU-014 receipt overclaims mutation")


def _current_inputs() -> dict[str, Any]:
    master = _read_json(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json(POLICY_PATH, "compiler provisioning policy")
    manifest = _read_json(TOOLCHAIN_MANIFEST_PATH, "toolchain manifest")
    try:
        pipeline_lock = verify_pipeline_lock()
        release_report = check_release(recompile=False)
    except (RuntimeError, SyntheticVerificationError) as exc:
        raise CompilerProvisioningReadinessError(
            f"verified synthetic source evidence failed: {exc}"
        ) from exc
    source_paths = (
        MASTER_BACKLOG_PATH,
        FOLLOWUP_BACKLOG_PATH,
        POLICY_PATH,
        TOOLCHAIN_MANIFEST_PATH,
        PIPELINE_LOCK_PATH,
        GENERATION_REPORT_PATH,
        COMPILE_RECEIPTS_PATH,
        CLI_CONFIG_PATH,
        *PACKAGE_INDEX_PATHS,
        CACHE_CONTRACT_PATH,
        Path(__file__),
    )
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "source": _toolchain_source_evidence(manifest, pipeline_lock),
        "installed": _installed_identity_evidence(manifest),
        "receipts": _receipt_evidence(release_report),
        "cache": _cache_observation(manifest, policy),
        "completion": _completion_evidence(policy, manifest),
        "source_sha256": {
            path.relative_to(AI_ROOT).as_posix(): _sha256_file(path)
            for path in source_paths
        },
    }


def _build_current_report(generated_on: str) -> dict[str, Any]:
    return build_report(generated_on=generated_on, **_current_inputs())


def evaluate(path: Path, generated_on: str) -> dict[str, Any]:
    report = _build_current_report(generated_on)
    _write_json(report, path)
    return report


def verify(path: Path) -> dict[str, Any]:
    report = _read_json(path, "VFAI-FU-014 readiness receipt")
    validate_report(report)
    if report != _build_current_report(str(report.get("generatedOn"))):
        raise CompilerProvisioningReadinessError("VFAI-FU-014 readiness receipt is stale")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--generated-on", required=True)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = (
        evaluate(args.output.resolve(), args.generated_on)
        if args.command == "evaluate"
        else verify(args.input.resolve())
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": args.command,
                "reportId": report["reportId"],
                "decision": report["decision"],
                "verifiedProfileCount": report["installedIdentity"][
                    "verifiedProfileCount"
                ],
                "retainedReceiptCount": report["retainedReceipts"]["receiptCount"],
                "localDownloadFileCount": report["localCacheObservation"][
                    "downloadFileCount"
                ],
                "localDownloadBytes": report["localCacheObservation"]["downloadBytes"],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
