from __future__ import annotations

from collections import Counter
import json

from evaluation.leakage import exclude_held_out_records
from synthetic_data.generator import expected_task_names
from synthetic_data.grammar import (
    DETERMINISTIC_SEED,
    FIRMWARE_TARGETS,
    GENERATOR_VERSION,
    GRAMMAR_VERSION,
    NON_AVR_FIRMWARE_BOARDS,
    SUPPORTED_BOARDS,
)
from synthetic_data.pipeline import SHARD_ROOT, SHARDS, check_release, verify_pipeline_lock
from task_schema.io import read_task_shard


def test_checked_in_verified_synthetic_release_is_current_and_complete() -> None:
    report = check_release()
    assert report["summary"] == {
        "candidateCount": 251,
        "acceptedRecordCount": 250,
        "rejectedCandidateCount": 1,
        "exactDuplicateCount": 1,
        "heldOutCollisionCount": 0,
        "shardCount": 4,
        "decision": "pass",
    }
    assert report["grammar"]["deterministicSeed"] == DETERMINISTIC_SEED
    assert report["grammar"]["version"] == GRAMMAR_VERSION
    assert report["generator"]["version"] == GENERATOR_VERSION
    assert report["coverage"]["supportedBoards"]["covered"] == sorted(SUPPORTED_BOARDS)
    assert report["coverage"]["supportedBoards"]["coveragePercent"] == 100.0
    assert report["tokens"]["utf8ByteTokenCount"] > 0
    assert report["tokens"]["lexicalTokenCount"] > 0


def test_every_task_record_is_unique_leakage_free_and_deterministically_evidenced() -> None:
    records = [record for name in SHARDS for record in read_task_shard(SHARD_ROOT / name)]
    assert len(records) == 250
    assert len({record["recordId"] for record in records}) == len(records)
    assert {record["task"] for record in records} == expected_task_names()
    accepted, rejected = exclude_held_out_records(records)
    assert accepted == records
    assert rejected == []
    assert all(record["input"]["toolEvidence"] for record in records)
    assert all(
        evidence["authority"] == "deterministic"
        for record in records
        for evidence in record["input"]["toolEvidence"]
    )


def test_circuit_pin_and_wiring_labels_have_passed_receipts() -> None:
    records = [record for name in SHARDS for record in read_task_shard(SHARD_ROOT / name)]
    checked_tasks = {"pin_routing", "wiring", "circuit_validation", "structured_output"}
    checked = [record for record in records if record["task"] in checked_tasks]
    assert len(checked) == 121
    for record in checked:
        evidence = record["input"]["toolEvidence"]
        assert all(item["status"] == "complete" for item in evidence)
        assert all(item["payload"]["status"] == "pass" for item in evidence)
    assert all(
        action["applicationMode"] == "proposal-only"
        and action["requiresUserConfirmation"] is True
        for record in records
        for action in record["output"].get("structuredActions", [])
    )


def test_firmware_and_repairs_are_source_bound_to_real_compile_receipts() -> None:
    firmware = read_task_shard(SHARD_ROOT / "firmware.jsonl")
    repairs = read_task_shard(SHARD_ROOT / "compiler-repair.jsonl")
    assert len(firmware) == 44
    assert len(repairs) == 22
    assert all(
        record["input"]["toolEvidence"][0]["payload"]["observedSuccess"] is True
        for record in firmware
    )
    for record in repairs:
        outcomes = Counter(
            item["payload"]["observedSuccess"] for item in record["input"]["toolEvidence"]
        )
        assert outcomes == {False: 1, True: 1}


def test_pipeline_lock_binds_grammar_verifiers_schema_leakage_and_toolchain() -> None:
    lock = verify_pipeline_lock()
    paths = {item["path"] for item in lock["dependencies"]}
    assert {
        "synthetic_data/grammar.py",
        "synthetic_data/verifiers.py",
        "synthetic_data/generator.py",
        "synthetic_data/toolchains.v1.json",
        "task_schema/task-record.schema.json",
        "evaluation/leakage.py",
        "electronics_corpus/catalog.v1.json",
    }.issubset(paths)
    assert set(lock["toolchains"]) == {
        "arduino-avr-1.8.6",
        "arduino-megaavr-1.8.8",
        "arduino-renesas-uno-1.6.0",
        "esp32-3.3.11",
        "pico-6.0.0",
    }
    assert all(
        profile["arduinoCliVersion"] == "1.5.1"
        for profile in lock["toolchains"].values()
    )
    assert {target.toolchain_id for target in FIRMWARE_TARGETS} == set(lock["toolchains"])
    assert len(lock["lockSha256"]) == 64


def test_generation_report_has_explicit_rejection_reasons_and_shard_balance() -> None:
    report = check_release()
    assert report["rejections"]["reasonCounts"] == {"exact-duplicate": 1}
    assert {item["path"].split("/")[-1] for item in report["shards"]} == set(SHARDS)
    assert sum(item["recordCount"] for item in report["shards"]) == 250
    assert json.dumps(report["taskBalance"], sort_keys=True)


def test_every_exact_board_has_a_pinned_compiler_profile_and_receipts() -> None:
    report = check_release()
    assert NON_AVR_FIRMWARE_BOARDS == ()
    assert {target.board for target in FIRMWARE_TARGETS} == set(SUPPORTED_BOARDS)
    assert report["compileVerification"]["receiptCount"] == 88
    assert report["compileVerification"]["acceptedCompileCount"] == 66
    assert report["compileVerification"]["expectedFailureCount"] == 22
    assert report["coverage"]["compiledFirmwareTargets"] == [
        {
            "board": target.board,
            "fqbn": target.fqbn,
            "toolchainId": target.toolchain_id,
        }
        for target in FIRMWARE_TARGETS
    ]
