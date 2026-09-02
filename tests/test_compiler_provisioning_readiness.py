from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from compiler_provisioning import (
    CompilerCacheContractError,
    cache_key_from_inputs,
    verify_cache_export,
)
from tools.evaluate_compiler_provisioning_readiness import (
    CompilerProvisioningReadinessError,
    POLICY_PATH,
    _receipt_evidence,
    build_report,
    validate_report,
)


def _policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _report(**completion_overrides: bool) -> dict[str, object]:
    completion = {
        "provisioningInputClosureReady": False,
        "ownerBudgetReady": False,
        "cacheExportReady": False,
        "cleanHostProvisioningReady": False,
        "offlineCacheRestoreReady": False,
    }
    completion.update(completion_overrides)
    return build_report(
        master_backlog={
            "status": "completed",
            "items": [{"id": "VFAI-035", "status": "done"}],
        },
        followup_backlog={
            "items": [{"id": "VFAI-FU-014", "status": "accepted_for_later"}]
        },
        policy=_policy(),
        source={"ready": True},
        installed={"ready": True},
        receipts={"ready": True},
        cache={"draftKeysAuthorizeExport": False},
        completion=completion,
        generated_on="2026-09-01",
        source_sha256={"policy": "source"},
    )


def _key_inputs() -> dict[str, object]:
    return {
        "toolchainManifestSha256": "1" * 64,
        "profileId": "arduino-avr-1.8.6",
        "coreId": "arduino:avr",
        "coreVersion": "1.8.6",
        "fqbns": ["arduino:avr:uno"],
        "host": "windows-x86_64",
        "arduinoCliVersion": "1.5.1",
        "arduinoCliExecutableSha256": "2" * 64,
        "cliCommandContract": ["arduino-cli", "compile", "<EXACT_FQBN>"],
        "packageIndexSha256": ["3" * 64],
        "dependencyArchives": [
            {"name": "avr-gcc.zip", "bytes": 8, "sha256": "4" * 64}
        ],
        "source": {"url": "https://example.invalid", "revision": "1.8.6"},
    }


def _cache_manifest(root: Path) -> dict[str, object]:
    payload = b"compiler"
    (root / "tool.bin").write_bytes(payload)
    inputs = _key_inputs()
    return {
        "schemaVersion": 1,
        "manifestKind": "vfai-fu-014-compiler-cache-export-v1",
        "keyInputs": inputs,
        "cacheKey": cache_key_from_inputs(inputs),
        "files": [
            {
                "path": "tool.bin",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "totalBytes": len(payload),
    }


def test_current_checkpoint_stops_before_provisioning_input_closure() -> None:
    report = _report()

    assert report["decision"] == "complete-checksummed-provisioning-input-closure"
    assert report["networkAccessed"] is False
    assert report["archivesDownloadedByEvaluator"] == 0
    assert report["cacheExportsCreatedByEvaluator"] == 0
    assert report["hostsProvisionedByEvaluator"] == 0
    validate_report(report)


def test_retained_receipt_gate_reads_the_generation_summary_decision() -> None:
    evidence = _receipt_evidence(
        {
            "summary": {"decision": "pass"},
            "compileVerification": {
                "receiptCount": 88,
                "acceptedCompileCount": 66,
                "expectedFailureCount": 22,
            },
        }
    )

    assert evidence["ready"] is True
    assert evidence["releaseDecision"] == "pass"


def test_completion_gates_advance_in_fail_closed_order() -> None:
    assert _report(provisioningInputClosureReady=True)["decision"] == (
        "await-owner-provisioning-and-cache-budget"
    )
    assert _report(
        provisioningInputClosureReady=True,
        ownerBudgetReady=True,
    )["decision"] == "create-content-addressed-offline-cache-export"
    assert _report(
        provisioningInputClosureReady=True,
        ownerBudgetReady=True,
        cacheExportReady=True,
    )["decision"] == "await-approved-clean-host-provisioning"
    assert _report(
        provisioningInputClosureReady=True,
        ownerBudgetReady=True,
        cacheExportReady=True,
        cleanHostProvisioningReady=True,
    )["decision"] == "await-offline-cache-restore-verification"


def test_all_receipts_still_require_independent_review() -> None:
    report = _report(
        provisioningInputClosureReady=True,
        ownerBudgetReady=True,
        cacheExportReady=True,
        cleanHostProvisioningReady=True,
        offlineCacheRestoreReady=True,
    )

    assert report["decision"] == "clean-host-results-ready-for-independent-review"
    assert report["completionClaimed"] is False
    validate_report(report)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("host", "linux-x86_64"),
        ("fqbns", ["arduino:avr:nano"]),
        ("cliCommandContract", ["arduino-cli", "compile", "--clean"]),
        ("dependencyArchives", []),
    ],
)
def test_cache_key_binds_runtime_identity(field: str, replacement: object) -> None:
    original = _key_inputs()
    changed = deepcopy(original)
    changed[field] = replacement

    assert cache_key_from_inputs(original) != cache_key_from_inputs(changed)


def test_cache_export_accepts_only_exact_inventory(tmp_path: Path) -> None:
    manifest = _cache_manifest(tmp_path)

    result = verify_cache_export(manifest, tmp_path)

    assert result["exactFileSet"] is True
    assert result["fileCount"] == 1
    assert result["totalBytes"] == 8


def test_cache_export_rejects_missing_extra_corrupt_and_stale_files(
    tmp_path: Path,
) -> None:
    manifest = _cache_manifest(tmp_path)
    (tmp_path / "tool.bin").unlink()
    with pytest.raises(CompilerCacheContractError, match="missing"):
        verify_cache_export(manifest, tmp_path)

    manifest = _cache_manifest(tmp_path)
    (tmp_path / "extra.bin").write_bytes(b"extra")
    with pytest.raises(CompilerCacheContractError, match="undeclared"):
        verify_cache_export(manifest, tmp_path)
    (tmp_path / "extra.bin").unlink()

    (tmp_path / "tool.bin").write_bytes(b"corrupt!")
    with pytest.raises(CompilerCacheContractError, match="changed"):
        verify_cache_export(manifest, tmp_path)

    manifest = _cache_manifest(tmp_path)
    manifest["cacheKey"] = "0" * 64
    with pytest.raises(CompilerCacheContractError, match="stale"):
        verify_cache_export(manifest, tmp_path)


def test_cache_export_rejects_escaping_or_noncanonical_paths(tmp_path: Path) -> None:
    manifest = _cache_manifest(tmp_path)
    manifest["files"][0]["path"] = "../tool.bin"  # type: ignore[index]
    with pytest.raises(CompilerCacheContractError, match="escapes"):
        verify_cache_export(manifest, tmp_path)

    manifest = _cache_manifest(tmp_path)
    manifest["files"][0]["path"] = r"folder\tool.bin"  # type: ignore[index]
    with pytest.raises(CompilerCacheContractError, match="escapes"):
        verify_cache_export(manifest, tmp_path)


def test_policy_freezes_scope_command_and_mutation_boundaries() -> None:
    policy = _policy()

    assert policy["scope"]["requiredProfileCount"] == 5  # type: ignore[index]
    assert policy["scope"]["requiredExactFqbnCount"] == 11  # type: ignore[index]
    assert policy["scope"]["requiredCompileReceiptCount"] == 88  # type: ignore[index]
    assert policy["cacheContract"]["cliCommandContract"][0] == "arduino-cli"  # type: ignore[index]
    assert policy["authorization"]["ownerBudgetReceiptPath"] is None  # type: ignore[index]
    assert policy["mutationBoundary"][  # type: ignore[index]
        "compilerCacheMayBeExportedWithoutBudget"
    ] is False
    assert "compile-with-network-access" in policy["forbidden"]  # type: ignore[operator]


def test_tampered_readiness_receipt_fails_closed() -> None:
    report = _report()
    report["hostsProvisionedByEvaluator"] = 1

    with pytest.raises(CompilerProvisioningReadinessError, match="checksum"):
        validate_report(report)
