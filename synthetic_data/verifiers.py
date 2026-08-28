"""Deterministic corpus, netlist, and compiler gates for synthetic labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from electronics_corpus import get_electronics_corpus
from engine.pin_router import PinRouter


AI_ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN_ROOT = AI_ROOT / ".toolchains"
CLI_PATH = TOOLCHAIN_ROOT / "arduino-cli-1.5.1" / "arduino-cli.exe"
CLI_CONFIG_PATH = TOOLCHAIN_ROOT / "arduino-cli.yaml"
AVR_GPP_PATH = (
    TOOLCHAIN_ROOT
    / "arduino-data"
    / "packages"
    / "arduino"
    / "tools"
    / "avr-gcc"
    / "7.3.0-atmel3.6.1-arduino7"
    / "bin"
    / "avr-g++.exe"
)
AVR_PLATFORM_PATH = (
    TOOLCHAIN_ROOT
    / "arduino-data"
    / "packages"
    / "arduino"
    / "hardware"
    / "avr"
    / "1.8.6"
    / "platform.txt"
)

TOOLCHAIN_CONTRACT = {
    "arduinoCliVersion": "1.5.1",
    "arduinoCliWindowsX64ZipSha256": "fabe42e0eb04d00e776a66178299ff95a46c623dbc260f997e58fd514853dd40",
    "arduinoCliExecutableSha256": "1017de89179c3167e6b8a38ca6cc4091fa69a3cf97aafa7810f01423003f5571",
    "arduinoAvrCoreVersion": "1.8.6",
    "avrGccPackageVersion": "7.3.0-atmel3.6.1-arduino7",
    "avrGppExecutableSha256": "098f5708a5d70e3abb6b504145b46c9320ef4b6d43b6b8f1c09b431e611d543b",
    "avrPlatformTxtSha256": "513b2ee073e686a9e823a6a512e64a5a0ee33af3e1d7108e0fc869a11485a7aa",
}


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


def toolchain_identity(*, require_files: bool) -> dict[str, Any]:
    paths = (CLI_PATH, AVR_GPP_PATH, AVR_PLATFORM_PATH)
    if require_files and any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise SyntheticVerificationError(f"Pinned Arduino toolchain is incomplete: {missing}")
    identity = dict(TOOLCHAIN_CONTRACT)
    if all(path.is_file() for path in paths):
        actual = {
            "arduinoCliExecutableSha256": sha256_file(CLI_PATH),
            "avrGppExecutableSha256": sha256_file(AVR_GPP_PATH),
            "avrPlatformTxtSha256": sha256_file(AVR_PLATFORM_PATH),
        }
        mismatches = [name for name, value in actual.items() if identity[name] != value]
        if mismatches:
            raise SyntheticVerificationError(
                "Pinned Arduino toolchain checksum mismatch: " + ", ".join(mismatches)
            )
    return identity


def _normalized_compiler_output(value: str, temporary_root: Path) -> str:
    normalized = value.replace(str(temporary_root), "<TEMP_SKETCH>")
    normalized = normalized.replace(str(temporary_root).replace("\\", "/"), "<TEMP_SKETCH>")
    normalized = normalized.replace(str(AI_ROOT), "<AI_ROOT>")
    normalized = normalized.replace(str(AI_ROOT).replace("\\", "/"), "<AI_ROOT>")
    normalized = normalized.replace("\\", "/")
    normalized = re.sub(r"[A-Za-z]:/[^\r\n\"]*?vfai-compile-[^/\r\n\"]+", "<TEMP_SKETCH>", normalized)
    return normalized.strip()


def compiler_receipt(case: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one sketch and return a deterministic, source-bound receipt."""

    identity = toolchain_identity(require_files=True)
    source = str(case["source"])
    expected_success = bool(case["expectedSuccess"])
    work_root = TOOLCHAIN_ROOT / "compile-work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vfai-compile-", dir=work_root) as temp_name:
        temporary_root = Path(temp_name)
        sketch_name = "verified_sketch"
        sketch_dir = temporary_root / sketch_name
        sketch_dir.mkdir()
        (sketch_dir / f"{sketch_name}.ino").write_text(source, encoding="utf-8", newline="\n")
        command = [
            str(CLI_PATH),
            "--config-file",
            str(CLI_CONFIG_PATH),
            "compile",
            "--fqbn",
            str(case["fqbn"]),
            "--warnings",
            "all",
            str(sketch_dir),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        succeeded = completed.returncode == 0
        normalized_output = _normalized_compiler_output(
            completed.stdout + "\n" + completed.stderr, temporary_root
        )
    if succeeded != expected_success:
        raise SyntheticVerificationError(
            f"Compiler outcome mismatch for {case['caseId']}: expected {expected_success}, got {succeeded}"
        )
    source_hash = sha256_bytes(source.encode("utf-8"))
    receipt_material = {
        "caseId": case["caseId"],
        "fqbn": case["fqbn"],
        "sourceSha256": source_hash,
        "expectedSuccess": expected_success,
        "observedSuccess": succeeded,
        "toolchain": identity,
    }
    return {
        "schemaVersion": 1,
        "receiptId": "vf-compile-receipt-v1-" + content_sha256(receipt_material)[:24],
        **receipt_material,
        "exitCode": completed.returncode,
        "outcomeClass": "compile-success" if succeeded else "expected-compile-failure",
        "normalizedOutputSha256": sha256_bytes(normalized_output.encode("utf-8")),
        "normalizedOutputSummary": (
            "The pinned compiler accepted this exact source and target."
            if succeeded
            else "The pinned compiler rejected this intentionally broken source as expected."
        ),
    }


def validate_compiler_receipt(case: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    source_hash = sha256_bytes(str(case["source"]).encode("utf-8"))
    if receipt.get("caseId") != case["caseId"] or receipt.get("sourceSha256") != source_hash:
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
    if receipt.get("toolchain") != toolchain_identity(require_files=False):
        raise SyntheticVerificationError(f"Compile receipt toolchain changed for {case['caseId']}")
    receipt_material = {
        "caseId": case["caseId"],
        "fqbn": case["fqbn"],
        "sourceSha256": source_hash,
        "expectedSuccess": expected,
        "observedSuccess": expected,
        "toolchain": receipt["toolchain"],
    }
    expected_id = "vf-compile-receipt-v1-" + content_sha256(receipt_material)[:24]
    if receipt.get("receiptId") != expected_id:
        raise SyntheticVerificationError(f"Compile receipt ID is invalid for {case['caseId']}")
    return dict(receipt)
