"""Build and verify the immutable VFAI-009 synthetic-data release."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping

from data_governance.governance import (
    default_manifest_path,
    require_approved_shard,
    require_approved_source,
    sha256_file,
    write_approved_shard_manifest,
)
from evaluation.leakage import exclude_held_out_records
from task_schema.io import read_task_shard, write_task_shard
from task_schema.schema import validate_task_record

from synthetic_data.generator import SOURCE_IDS, expected_task_names, generate_candidates
from synthetic_data.grammar import (
    DETERMINISTIC_SEED,
    FIRMWARE_TARGETS,
    GENERATOR_ID,
    GENERATOR_VERSION,
    GRAMMAR_ID,
    GRAMMAR_VERSION,
    SUPPORTED_BOARDS,
    render_firmware_sources,
)
from synthetic_data.verifiers import (
    SyntheticVerificationError,
    canonical_json,
    compiler_receipt,
    content_sha256,
    toolchain_identity,
    validate_compiler_receipt,
)


AI_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = AI_ROOT / "synthetic_data"
LOCK_PATH = SYNTHETIC_ROOT / "pipeline-lock.v1.json"
RECEIPT_PATH = SYNTHETIC_ROOT / "receipts" / "v1" / "compile-receipts.jsonl"
REPORT_PATH = SYNTHETIC_ROOT / "reports" / "current-generation-report.json"
SHARD_ROOT = SYNTHETIC_ROOT / "shards" / "v1"
SHARDS = {
    "grounded-electronics.jsonl": {
        "domain_chat",
        "pin_routing",
        "wiring",
    },
    "circuit-safety.jsonl": {
        "circuit_validation",
        "structured_output",
        "uncertainty",
        "refusal",
    },
    "firmware.jsonl": {"firmware_generation"},
    "compiler-repair.jsonl": {"compiler_repair"},
}
LEXICAL_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\s]", re.UNICODE)

LOCKED_DEPENDENCIES = (
    "synthetic_data/grammar.py",
    "synthetic_data/verifiers.py",
    "synthetic_data/generator.py",
    "synthetic_data/pipeline.py",
    "tools/build_verified_synthetic_data.py",
    "electronics_corpus/catalog.v1.json",
    "electronics_corpus/knowledge-record.schema.json",
    "task_schema/task-record.schema.json",
    "evaluation/leakage.py",
    "engine/pin_router.py",
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(AI_ROOT).as_posix()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for value in values
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_pipeline_lock() -> dict[str, Any]:
    dependencies = []
    for relative in LOCKED_DEPENDENCIES:
        path = AI_ROOT / relative
        if not path.is_file():
            raise SyntheticVerificationError(f"Pipeline dependency is missing: {relative}")
        dependencies.append({"path": relative, "sha256": sha256_file(path)})
    lock = {
        "schemaVersion": 1,
        "pipelineId": "vf-verified-synthetic-pipeline-v1",
        "pipelineVersion": GENERATOR_VERSION,
        "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION},
        "grammar": {
            "id": GRAMMAR_ID,
            "version": GRAMMAR_VERSION,
            "deterministicSeed": DETERMINISTIC_SEED,
        },
        "toolchain": toolchain_identity(require_files=False),
        "dependencies": dependencies,
        "outputs": {
            "shards": [f"synthetic_data/shards/v1/{name}" for name in SHARDS],
            "compileReceipts": _relative(RECEIPT_PATH),
            "generationReport": _relative(REPORT_PATH),
        },
        "contracts": {
            "taskRecord": "vf-task-record-jsonl-v1",
            "compileReceipt": "vf-compile-receipt-v1",
            "heldOutExclusion": "evaluation.leakage.exclude_held_out_records",
            "structuredApplicationMode": "proposal-only",
        },
    }
    lock["lockSha256"] = content_sha256(lock)
    return lock


def write_pipeline_lock() -> Path:
    _atomic_write(LOCK_PATH, _json_bytes(build_pipeline_lock()))
    return LOCK_PATH


def verify_pipeline_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise SyntheticVerificationError("Synthetic pipeline lock is missing")
    try:
        current = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SyntheticVerificationError("Synthetic pipeline lock is invalid") from error
    expected = build_pipeline_lock()
    if current != expected:
        raise SyntheticVerificationError(
            "Synthetic pipeline dependencies changed; review and rewrite pipeline-lock.v1.json first"
        )
    return current


def _compile_all() -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    cases = render_firmware_sources()
    for index, case in enumerate(cases, start=1):
        print(f"compile gate {index}/{len(cases)}: {case['caseId']}", flush=True)
        receipt = compiler_receipt(case)
        receipts[str(case["caseId"])] = receipt
    return receipts


def _load_receipts() -> dict[str, dict[str, Any]]:
    if not RECEIPT_PATH.is_file():
        raise SyntheticVerificationError("Compile receipts are missing; run the write build with the pinned toolchain")
    receipts: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(RECEIPT_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            receipt = json.loads(line)
        except json.JSONDecodeError as error:
            raise SyntheticVerificationError(f"Invalid compile receipt JSON at line {line_number}") from error
        case_id = receipt.get("caseId")
        if not isinstance(case_id, str) or case_id in receipts:
            raise SyntheticVerificationError(f"Invalid or duplicate compile receipt at line {line_number}")
        receipts[case_id] = receipt
    expected_cases = {str(case["caseId"]): case for case in render_firmware_sources()}
    if set(receipts) != set(expected_cases):
        raise SyntheticVerificationError("Compile receipt case coverage does not match the locked grammar")
    return {
        case_id: validate_compiler_receipt(expected_cases[case_id], receipt)
        for case_id, receipt in sorted(receipts.items())
    }


def _deduplicate(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = []
    rejected = []
    seen: dict[str, str] = {}
    for record in records:
        signature = content_sha256(record)
        if signature in seen:
            rejected.append(
                {
                    "candidate": record["recordId"],
                    "reasonCode": "exact-duplicate",
                    "detail": f"Duplicates accepted record {seen[signature]}",
                }
            )
        else:
            seen[signature] = record["recordId"]
            accepted.append(record)
    return accepted, rejected


def _shuffle_group(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    values = list(records)
    name_seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:12], 16)
    random.Random(DETERMINISTIC_SEED + name_seed).shuffle(values)
    return values


def _build_dataset(
    receipts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidates, rejections = generate_candidates(receipts)
    deduplicated, duplicate_rejections = _deduplicate(candidates)
    rejections.extend(duplicate_rejections)
    accepted, leakage_rejections = exclude_held_out_records(deduplicated)
    rejections.extend(
        {
            "candidate": deduplicated[item["recordIndex"]]["recordId"],
            "reasonCode": "held-out-evaluation-collision",
            "detail": {
                key: value for key, value in item.items() if key != "recordIndex"
            },
        }
        for item in leakage_rejections
    )
    accepted = [validate_task_record(item) for item in accepted]
    record_ids = [item["recordId"] for item in accepted]
    if len(record_ids) != len(set(record_ids)):
        raise SyntheticVerificationError("Accepted record IDs are not unique")
    task_names = {item["task"] for item in accepted}
    missing_tasks = expected_task_names() - task_names
    if missing_tasks:
        raise SyntheticVerificationError(f"Required task categories were lost: {sorted(missing_tasks)}")

    shards: dict[str, list[dict[str, Any]]] = {}
    for filename, tasks in SHARDS.items():
        shards[filename] = _shuffle_group(
            [record for record in accepted if record["task"] in tasks], filename
        )
    assigned = sum(len(records) for records in shards.values())
    if assigned != len(accepted):
        raise SyntheticVerificationError("Accepted records were not assigned to exactly one shard")

    all_bytes = b"".join(_jsonl_bytes(shards[name]) for name in SHARDS)
    task_counts = Counter(record["task"] for record in accepted)
    board_counts = Counter(
        str(record["input"]["projectContext"].get("boardType"))
        for record in accepted
        if record["input"]["projectContext"].get("boardType") in SUPPORTED_BOARDS
    )
    component_counts = Counter()
    for record in accepted:
        payload = record["input"]["projectContext"]["payload"]
        component = payload.get("component")
        if component:
            component_counts[str(component)] += 1
    reason_counts = Counter(str(item["reasonCode"]) for item in rejections)
    receipt_values = list(receipts.values())
    shard_descriptors = [
        {
            "path": f"synthetic_data/shards/v1/{name}",
            "recordCount": len(records),
            "sha256": hashlib.sha256(_jsonl_bytes(records)).hexdigest(),
        }
        for name, records in shards.items()
    ]
    report = {
        "schemaVersion": 1,
        "reportId": "vf-verified-synthetic-generation-report-v1",
        "pipelineLockSha256": json.loads(LOCK_PATH.read_text(encoding="utf-8"))["lockSha256"],
        "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION},
        "grammar": {
            "id": GRAMMAR_ID,
            "version": GRAMMAR_VERSION,
            "deterministicSeed": DETERMINISTIC_SEED,
        },
        "summary": {
            "candidateCount": len(candidates) + len(rejections) - len(duplicate_rejections) - len(leakage_rejections),
            "acceptedRecordCount": len(accepted),
            "rejectedCandidateCount": len(rejections),
            "exactDuplicateCount": reason_counts.get("exact-duplicate", 0),
            "heldOutCollisionCount": reason_counts.get("held-out-evaluation-collision", 0),
            "shardCount": len(shards),
            "decision": "pass",
        },
        "taskBalance": {
            task: {
                "records": count,
                "percent": round(count * 100 / len(accepted), 3),
            }
            for task, count in sorted(task_counts.items())
        },
        "coverage": {
            "supportedBoards": {
                "required": list(SUPPORTED_BOARDS),
                "covered": sorted(board_counts),
                "recordCounts": dict(sorted(board_counts.items())),
                "coveragePercent": round(len(board_counts) * 100 / len(SUPPORTED_BOARDS), 3),
            },
            "components": {
                "covered": sorted(component_counts),
                "recordCounts": dict(sorted(component_counts.items())),
            },
            "compiledFirmwareTargets": [
                {"board": target.board, "fqbn": target.fqbn} for target in FIRMWARE_TARGETS
            ],
        },
        "tokens": {
            "countingContract": "exact UTF-8 bytes plus stable lexical pre-tokenizer count",
            "utf8ByteTokenCount": len(all_bytes),
            "lexicalTokenCount": len(LEXICAL_TOKEN_PATTERN.findall(all_bytes.decode("utf-8"))),
            "note": "VFAI-010 will produce the model-token count after the project-owned tokenizer is trained.",
        },
        "compileVerification": {
            "receiptCount": len(receipt_values),
            "acceptedCompileCount": sum(1 for item in receipt_values if item["observedSuccess"]),
            "expectedFailureCount": sum(1 for item in receipt_values if not item["observedSuccess"]),
            "toolchain": toolchain_identity(require_files=False),
            "receiptsPath": _relative(RECEIPT_PATH),
            "receiptsSha256": hashlib.sha256(
                _jsonl_bytes(receipts[case_id] for case_id in sorted(receipts))
            ).hexdigest(),
        },
        "duplicates": {
            "method": "SHA-256 of canonical complete task record",
            "rejected": reason_counts.get("exact-duplicate", 0),
        },
        "rejections": {
            "reasonCounts": dict(sorted(reason_counts.items())),
            "items": sorted(rejections, key=lambda item: (str(item["reasonCode"]), str(item["candidate"]))),
        },
        "leakage": {
            "gate": "evaluation.leakage.exclude_held_out_records",
            "heldOutCollisionsRejected": len(leakage_rejections),
        },
        "shards": shard_descriptors,
    }
    report["reportSha256"] = content_sha256(report)
    return shards, report


def _require_sources() -> None:
    for source_id in SOURCE_IDS:
        require_approved_source(source_id, "synthetic-generation")
        require_approved_source(source_id, "training")


def write_release() -> dict[str, Any]:
    """Recompile all firmware and atomically write the checked-in v1 release."""

    verify_pipeline_lock()
    _require_sources()
    receipts = _compile_all()
    receipt_bytes = _jsonl_bytes(receipts[case_id] for case_id in sorted(receipts))
    _atomic_write(RECEIPT_PATH, receipt_bytes)
    shards, report = _build_dataset(receipts)
    for filename, records in shards.items():
        write_task_shard(SHARD_ROOT / filename, records)
    _atomic_write(REPORT_PATH, _json_bytes(report))
    for filename in SHARDS:
        write_approved_shard_manifest(
            SHARD_ROOT / filename,
            source_ids=SOURCE_IDS,
            producer_id="vf-verified-synthetic-pipeline-v1",
            producer_version=GENERATOR_VERSION,
            producer_path="synthetic_data/pipeline-lock.v1.json",
        )
    check_release(recompile=False)
    return report


def check_release(*, recompile: bool = False) -> dict[str, Any]:
    """Verify all bytes, labels, leakage gates, receipts, and governance without writes."""

    verify_pipeline_lock()
    _require_sources()
    receipts = _load_receipts()
    if recompile:
        current = _compile_all()
        if current != receipts:
            raise SyntheticVerificationError("Fresh compiler receipts differ from the retained receipts")
    expected_receipt_bytes = _jsonl_bytes(receipts[case_id] for case_id in sorted(receipts))
    if RECEIPT_PATH.read_bytes() != expected_receipt_bytes:
        raise SyntheticVerificationError("Compile receipt file is not canonical")
    shards, report = _build_dataset(receipts)
    for filename, records in shards.items():
        path = SHARD_ROOT / filename
        if not path.is_file() or path.read_bytes() != _jsonl_bytes(records):
            raise SyntheticVerificationError(f"Synthetic shard is stale: {filename}")
        if read_task_shard(path) != records:
            raise SyntheticVerificationError(f"Synthetic shard failed task-schema round trip: {filename}")
        require_approved_shard(path, "training", manifest_path=default_manifest_path(path))
    if not REPORT_PATH.is_file() or REPORT_PATH.read_bytes() != _json_bytes(report):
        raise SyntheticVerificationError("Synthetic generation report is stale")
    return report
