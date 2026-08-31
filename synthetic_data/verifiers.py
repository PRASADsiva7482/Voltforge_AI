"""Deterministic corpus, netlist, and compiler gates for synthetic labels."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from electronics_corpus import get_electronics_corpus
from engine.pin_router import PinRouter


AI_ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN_ROOT = AI_ROOT / ".toolchains"
CLI_PATH = TOOLCHAIN_ROOT / "arduino-cli-1.5.1" / "arduino-cli.exe"
CLI_CONFIG_PATH = TOOLCHAIN_ROOT / "arduino-cli.yaml"
TOOLCHAIN_MANIFEST_PATH = Path(__file__).with_name("toolchains.v1.json")
OFFLINE_NETWORK_POLICY = "offline-no-network"
TEMPORARY_BUILD_POLICY = "workspace-temporary-only"
_ACTIVE_COMPILER_SESSION: Path | None = None


class SyntheticVerificationError(RuntimeError):
    """A candidate or retained verification receipt failed closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def verify_board(board: str) -> dict[str, Any]:
    corpus = get_electronics_corpus()
    result = corpus.lookup("board", board)
    if result.status != "found":
        raise SyntheticVerificationError(f"Board is not an exact supported variant: {board} ({result.reasonCode})")
    record = result.records[0]
    claims = corpus.claims(record)
    required = ("mcu", "architecture", "logic-voltage-v", "flash-bytes", "sram-bytes")
    if any(claims.get(name) is None for name in required):
        raise SyntheticVerificationError(f"Board record has incomplete required claims: {record['recordId']}")
    return {
        "status": "pass",
        "verifier": "exact-corpus-board-v1",
        "recordId": record["recordId"],
        "effectiveRevision": record["effectiveRevision"]["revision"],
        "recordSha256": content_sha256(record),
        "claims": {name: claims[name] for name in required},
        "name": record["subject"]["name"],
        "variant": record["subject"]["variant"],
    }


def verify_pin_route(board: str) -> dict[str, Any]:
    board_receipt = verify_board(board)
    result = PinRouter.get_board_pinout(board)
    if result.get("status") != "found":
        raise SyntheticVerificationError(f"Pin map is unavailable for exact board {board}")
    required = ("i2c_sda", "i2c_scl", "gnd")
    if any(not result.get(name) for name in required):
        raise SyntheticVerificationError(f"Pin map lacks a default I2C route for {board}")
    supply = result.get("vcc_3v3") or result.get("vcc_5v")
    if not supply:
        raise SyntheticVerificationError(f"Pin map lacks a supported supply pin for {board}")
    return {
        "status": "pass",
        "verifier": "exact-corpus-pin-router-v1",
        "boardRecordId": board_receipt["recordId"],
        "pinMapRecordId": result["recordId"],
        "effectiveRevision": result["effectiveRevision"],
        "route": {
            "supply": supply,
            "ground": result["gnd"],
            "sda": result["i2c_sda"],
            "scl": result["i2c_scl"],
        },
    }


def expected_connections(board: str, component: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    corpus = get_electronics_corpus()
    result = corpus.lookup_wiring_recipe(board, component)
    routed = PinRouter.resolve_connections_result(component, board)
    if result.status != "found" or routed.get("status") != "found":
        raise SyntheticVerificationError(
            f"No exact wiring recipe for {board}+{component}: {result.reasonCode}"
        )
    record = result.records[0]
    claims = corpus.claims(record)
    expected = [
        {"componentTerminal": item["componentTerminal"], "boardPin": item["boardPin"]}
        for item in claims["connections"]
    ]
    routed_pairs = [
        {
            "componentTerminal": item["from"].split("/", 1)[1],
            "boardPin": item["to"].split("/", 1)[1],
        }
        for item in routed["connections"]
    ]
    if expected != routed_pairs:
        raise SyntheticVerificationError(f"Corpus and PinRouter disagree for {record['recordId']}")
    return record, expected


def verify_netlist(
    board: str,
    component: str,
    actual_connections: Sequence[Mapping[str, str]],
    *,
    expected_valid: bool,
) -> dict[str, Any]:
    record, expected = expected_connections(board, component)
    normalize = lambda items: sorted(
        (str(item.get("componentTerminal")), str(item.get("boardPin"))) for item in items
    )
    expected_set = set(normalize(expected))
    actual_list = normalize(actual_connections)
    actual_set = set(actual_list)
    duplicate_count = len(actual_list) - len(actual_set)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    actual_valid = not missing and not unexpected and duplicate_count == 0
    if actual_valid != expected_valid:
        raise SyntheticVerificationError(
            f"Netlist label mismatch for {board}: expected_valid={expected_valid}, actual_valid={actual_valid}"
        )
    return {
        "status": "pass",
        "verifier": "exact-recipe-netlist-v1",
        "knowledgeRecordId": record["recordId"],
        "effectiveRevision": record["effectiveRevision"]["revision"],
        "recordSha256": content_sha256(record),
        "expectedValid": expected_valid,
        "actualValid": actual_valid,
        "expectedConnections": expected,
        "actualConnections": [dict(item) for item in actual_connections],
        "missingConnections": [
            {"componentTerminal": terminal, "boardPin": pin} for terminal, pin in missing
        ],
        "unexpectedConnections": [
            {"componentTerminal": terminal, "boardPin": pin} for terminal, pin in unexpected
        ],
        "duplicateConnections": duplicate_count,
    }


def lookup_unknown_board(query: str) -> dict[str, Any]:
    result = get_electronics_corpus().lookup("board", query)
    if result.status == "found":
        raise SyntheticVerificationError(f"Expected an unresolved board query, but {query} is supported")
    return {
        "status": "pass",
        "verifier": "exact-corpus-unknown-v1",
        "query": query,
        "lookupStatus": result.status,
        "reasonCode": result.reasonCode,
        "candidateRecordIds": [item["recordId"] for item in result.records],
    }


def safety_receipt() -> dict[str, Any]:
    corpus = get_electronics_corpus()
    records = [item for item in corpus.records if item["recordType"] == "safety-constraint"]
    if not records:
        raise SyntheticVerificationError("The curated safety constraint pack is empty")
    record = records[0]
    claims = corpus.claims(record)
    return {
        "status": "pass",
        "verifier": "curated-safety-rule-v1",
        "recordId": record["recordId"],
        "recordSha256": content_sha256(record),
        "severity": claims["severity"],
        "condition": claims["condition"],
        "requiredAction": claims["required-action"],
        "unknownPolicy": claims["unknown-policy"],
    }


def _toolchain_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(TOOLCHAIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SyntheticVerificationError("Compiler toolchain manifest is invalid or missing") from error
    if manifest.get("schemaVersion") != 1 or not isinstance(manifest.get("profiles"), dict):
        raise SyntheticVerificationError("Compiler toolchain manifest contract is invalid")
    return manifest


def _profile_for_fqbn(fqbn: str) -> str:
    manifest = _toolchain_manifest()
    matches = [
        profile_id
        for profile_id, profile in manifest["profiles"].items()
        if fqbn in profile.get("fqbn", [])
    ]
    if len(matches) != 1:
        raise SyntheticVerificationError(f"No unique compiler profile is bound to FQBN {fqbn}")
    return str(matches[0])


def _validate_toolchain_source(profile_id: str, profile: Mapping[str, Any], *, require_files: bool) -> None:
    source = profile.get("source")
    if not source:
        return
    source_root = AI_ROOT / str(source["path"])
    if not source_root.exists():
        if require_files:
            raise SyntheticVerificationError(f"Pinned compiler source is missing: {source_root}")
        return
    try:
        actual_revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SyntheticVerificationError(f"Pinned compiler source cannot be verified: {profile_id}") from error
    if actual_revision != source["revision"]:
        raise SyntheticVerificationError(f"Pinned compiler source revision changed: {profile_id}")
    for relative, expected in source.get("submodules", {}).items():
        submodule = source_root / str(relative)
        try:
            actual = subprocess.check_output(
                ["git", "-C", str(submodule), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise SyntheticVerificationError(
                f"Pinned compiler submodule cannot be verified: {profile_id}:{relative}"
            ) from error
        if actual != expected:
            raise SyntheticVerificationError(
                f"Pinned compiler submodule revision changed: {profile_id}:{relative}"
            )


def _toolchain_identity_uncached(
    profile_id: str = "arduino-avr-1.8.6", *, require_files: bool
) -> dict[str, Any]:
    manifest = _toolchain_manifest()
    profile = manifest["profiles"].get(profile_id)
    if not isinstance(profile, dict):
        raise SyntheticVerificationError(f"Unknown compiler toolchain profile: {profile_id}")
    cli = manifest.get("arduinoCli")
    if not isinstance(cli, dict):
        raise SyntheticVerificationError("Arduino CLI identity is missing from compiler manifest")
    expected_files = {str(cli["path"]): str(cli["sha256"])}
    expected_files.update({str(path): str(value) for path, value in profile.get("fileHashes", {}).items()})
    missing = []
    mismatches = []
    for relative, expected in expected_files.items():
        path = AI_ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise SyntheticVerificationError(
            f"Pinned compiler toolchain checksum mismatch for {profile_id}: {mismatches}"
        )
    if require_files and missing:
        raise SyntheticVerificationError(
            f"Pinned compiler toolchain is incomplete for {profile_id}: {missing}"
        )
    _validate_toolchain_source(profile_id, profile, require_files=require_files)
    identity = {
        "manifestId": manifest["manifestId"],
        "manifestVersion": manifest["manifestVersion"],
        "profileId": profile_id,
        "coreId": profile["coreId"],
        "coreVersion": profile["coreVersion"],
        "arduinoCliVersion": cli["version"],
        "arduinoCliExecutableSha256": cli["sha256"],
        "license": dict(profile["license"]),
        "packageArchives": [dict(item) for item in profile.get("packageArchives", [])],
        "fileHashes": dict(profile.get("fileHashes", {})),
    }
    if profile.get("source"):
        identity["source"] = dict(profile["source"])
    return identity


@lru_cache(maxsize=None)
def _cached_toolchain_identity(profile_id: str, require_files: bool) -> str:
    return canonical_json(
        _toolchain_identity_uncached(profile_id, require_files=require_files)
    )


def reset_toolchain_identity_cache() -> None:
    """Start a fresh immutable toolchain-identity verification operation."""

    _cached_toolchain_identity.cache_clear()


def toolchain_identity(
    profile_id: str = "arduino-avr-1.8.6", *, require_files: bool
) -> dict[str, Any]:
    """Verify one profile once per operation and return an isolated identity."""

    return json.loads(_cached_toolchain_identity(profile_id, require_files))


def _normalized_compiler_output(value: str, temporary_root: Path) -> str:
    normalized = value.replace(str(temporary_root), "<TEMP_SKETCH>")
    normalized = normalized.replace(str(temporary_root).replace("\\", "/"), "<TEMP_SKETCH>")
    normalized = normalized.replace(str(AI_ROOT), "<AI_ROOT>")
    normalized = normalized.replace(str(AI_ROOT).replace("\\", "/"), "<AI_ROOT>")
    normalized = normalized.replace("\\", "/")
    normalized = re.sub(r"[A-Za-z]:/[^\r\n\"]*?vfai-compile-[^/\r\n\"]+", "<TEMP_SKETCH>", normalized)
    normalized = re.sub(
        r"[A-Za-z]:/[^\r\n\"]+/AppData/Local/arduino/sketches/[A-Fa-f0-9]+",
        "<CLI_CACHE>",
        normalized,
    )
    return normalized.strip()


@contextmanager
def compiler_session() -> Iterator[None]:
    """Share temporary compiler state for one release while isolating each FQBN.

    Arduino CLI's ``--clean`` flag disables its build cache.  A release has many
    sources per FQBN, so cleaning every case needlessly recompiles the same core
    and makes re-verification prohibitively expensive.  The session keeps one
    temporary build directory per exact FQBN, lets the CLI reuse that state, and
    removes the complete session when the release operation ends.
    """

    global _ACTIVE_COMPILER_SESSION
    previous = _ACTIVE_COMPILER_SESSION
    work_root = TOOLCHAIN_ROOT / "compile-work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vfai-compile-session-", dir=work_root) as temp_name:
        _ACTIVE_COMPILER_SESSION = Path(temp_name)
        try:
            yield
        finally:
            _ACTIVE_COMPILER_SESSION = previous


def compiler_receipt(case: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one sketch and return a deterministic, source-bound receipt."""

    profile_id = str(case.get("toolchainId") or _profile_for_fqbn(str(case["fqbn"])))
    identity = toolchain_identity(profile_id, require_files=True)
    source = str(case["source"])
    expected_success = bool(case["expectedSuccess"])
    work_root = TOOLCHAIN_ROOT / "compile-work"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary_owner = None
    if _ACTIVE_COMPILER_SESSION is None:
        temporary_owner = tempfile.TemporaryDirectory(prefix="vfai-compile-", dir=work_root)
        temporary_root = Path(temporary_owner.name)
    else:
        temporary_root = _ACTIVE_COMPILER_SESSION
    fqbn_key = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case["fqbn"]))
    sketch_name = "verified_sketch"
    sketch_dir = temporary_root / "sketches" / fqbn_key / sketch_name
    build_dir = temporary_root / "build" / fqbn_key
    sketch_dir.mkdir(parents=True, exist_ok=True)
    (sketch_dir / f"{sketch_name}.ino").write_text(source, encoding="utf-8", newline="\n")
    command = [
        str(CLI_PATH),
        "--config-file",
        str(CLI_CONFIG_PATH),
        "compile",
        "--jobs",
        "0",
        "--fqbn",
        str(case["fqbn"]),
        "--warnings",
        "all",
        "--build-path",
        str(build_dir),
        str(sketch_dir),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        succeeded = completed.returncode == 0
        normalized_output = _normalized_compiler_output(
            completed.stdout + "\n" + completed.stderr, temporary_root
        )
    finally:
        if temporary_owner is not None:
            temporary_owner.cleanup()
    if succeeded != expected_success:
        raise SyntheticVerificationError(
            f"Compiler outcome mismatch for {case['caseId']}: expected {expected_success}, got {succeeded}"
        )
    source_hash = sha256_bytes(source.encode("utf-8"))
    receipt_material = {
        "caseId": case["caseId"],
        "fqbn": case["fqbn"],
        "toolchainId": profile_id,
        "sourceSha256": source_hash,
        "expectedSuccess": expected_success,
        "observedSuccess": succeeded,
        "toolchain": identity,
        "networkPolicy": OFFLINE_NETWORK_POLICY,
        "buildPathPolicy": TEMPORARY_BUILD_POLICY,
    }
    return {
        "schemaVersion": 1,
        "receiptId": "vf-compile-receipt-v1-" + content_sha256(receipt_material)[:24],
        **receipt_material,
        "exitCode": completed.returncode,
        "compilerCommand": [
            "arduino-cli",
            "--config-file",
            "<LOCAL_CONFIG>",
            "compile",
            "--jobs",
            "0",
            "--fqbn",
            str(case["fqbn"]),
            "--warnings",
            "all",
            "--build-path",
            "<TEMP_BUILD>",
            "<TEMP_SKETCH>",
        ],
        "outcomeClass": "compile-success" if succeeded else "expected-compile-failure",
        "normalizedOutputSha256": sha256_bytes(normalized_output.encode("utf-8")),
        "normalizedOutputSummary": (
            "The pinned compiler accepted this exact source and target."
            if succeeded
            else "The pinned compiler rejected this intentionally broken source as expected."
        ),
    }


def validate_compiler_receipt(case: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    profile_id = str(case.get("toolchainId") or _profile_for_fqbn(str(case["fqbn"])))
    source_hash = sha256_bytes(str(case["source"]).encode("utf-8"))
    if (
        receipt.get("caseId") != case["caseId"]
        or receipt.get("toolchainId") != profile_id
        or receipt.get("sourceSha256") != source_hash
    ):
        raise SyntheticVerificationError(f"Compile receipt is not bound to source case {case['caseId']}")
    if receipt.get("fqbn") != case["fqbn"]:
        raise SyntheticVerificationError(f"Compile receipt FQBN changed for {case['caseId']}")
    expected = bool(case["expectedSuccess"])
    if receipt.get("expectedSuccess") is not expected or receipt.get("observedSuccess") is not expected:
        raise SyntheticVerificationError(f"Compile receipt outcome is invalid for {case['caseId']}")
    expected_class = "compile-success" if expected else "expected-compile-failure"
    expected_summary = (
        "The pinned compiler accepted this exact source and target."
        if expected
        else "The pinned compiler rejected this intentionally broken source as expected."
    )
    if receipt.get("outcomeClass") != expected_class or receipt.get("normalizedOutputSummary") != expected_summary:
        raise SyntheticVerificationError(f"Compile receipt normalized outcome changed for {case['caseId']}")
    exit_code = receipt.get("exitCode")
    if not isinstance(exit_code, int) or (exit_code == 0) is not expected:
        raise SyntheticVerificationError(f"Compile receipt exit code is invalid for {case['caseId']}")
    output_hash = receipt.get("normalizedOutputSha256")
    if not isinstance(output_hash, str) or re.fullmatch(r"[a-f0-9]{64}", output_hash) is None:
        raise SyntheticVerificationError(f"Compile receipt output checksum is invalid for {case['caseId']}")
    if receipt.get("networkPolicy") != OFFLINE_NETWORK_POLICY:
        raise SyntheticVerificationError(f"Compile receipt network policy is invalid for {case['caseId']}")
    if receipt.get("buildPathPolicy") != TEMPORARY_BUILD_POLICY:
        raise SyntheticVerificationError(f"Compile receipt build path policy is invalid for {case['caseId']}")
    expected_command = [
        "arduino-cli",
        "--config-file",
        "<LOCAL_CONFIG>",
        "compile",
        "--jobs",
        "0",
        "--fqbn",
        str(case["fqbn"]),
        "--warnings",
        "all",
        "--build-path",
        "<TEMP_BUILD>",
        "<TEMP_SKETCH>",
    ]
    if receipt.get("compilerCommand") != expected_command:
        raise SyntheticVerificationError(f"Compile receipt command is invalid for {case['caseId']}")
    if receipt.get("toolchain") != toolchain_identity(profile_id, require_files=False):
        raise SyntheticVerificationError(f"Compile receipt toolchain changed for {case['caseId']}")
    receipt_material = {
        "caseId": case["caseId"],
        "fqbn": case["fqbn"],
        "toolchainId": profile_id,
        "sourceSha256": source_hash,
        "expectedSuccess": expected,
        "observedSuccess": expected,
        "toolchain": receipt["toolchain"],
        "networkPolicy": OFFLINE_NETWORK_POLICY,
        "buildPathPolicy": TEMPORARY_BUILD_POLICY,
    }
    expected_id = "vf-compile-receipt-v1-" + content_sha256(receipt_material)[:24]
    if receipt.get("receiptId") != expected_id:
        raise SyntheticVerificationError(f"Compile receipt ID is invalid for {case['caseId']}")
    return dict(receipt)
