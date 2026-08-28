from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest


from data_governance.governance import require_approved_source
from electronics_corpus.builder import build_records, check_outputs
from electronics_corpus.schema import (
    KNOWLEDGE_SCHEMA_PATH,
    KnowledgeContractError,
    build_knowledge_json_schema,
    validate_knowledge_record,
)
from electronics_corpus.store import CATALOG_PATH, ElectronicsCorpus
from engine.deep_firmware import DeepFirmwareAnalyzer
from engine.pin_router import PinRouter


def test_checked_schema_and_every_generated_fact_are_valid_and_evidence_bound() -> None:
    assert json.loads(KNOWLEDGE_SCHEMA_PATH.read_text(encoding="utf-8")) == (
        build_knowledge_json_schema()
    )
    records = build_records()
    assert len(records) == 49
    assert {item["recordType"] for item in records} == {
        "board",
        "pin-map",
        "component",
        "wiring-recipe",
        "firmware-api",
        "compiler-diagnostic",
        "simulation-behavior",
        "safety-constraint",
    }
    for record in records:
        assert validate_knowledge_record(record) == record
        assert record["effectiveRevision"]["revision"] == "revision:1.0.0"
        assert record["effectiveRevision"]["validFrom"] == "2026-08-28"
        evidence_ids = {
            item["evidenceId"] for item in record["provenance"]["evidence"]
        }
        assert evidence_ids
        assert all(set(claim["evidenceRefs"]) <= evidence_ids for claim in record["claims"])


def test_missing_or_unresolved_evidence_fails_closed() -> None:
    record = deepcopy(build_records()[0])
    record["provenance"]["evidence"] = []
    with pytest.raises(KnowledgeContractError):
        validate_knowledge_record(record)

    record = deepcopy(build_records()[0])
    record["claims"][0]["evidenceRefs"] = ["evidence.not-present"]
    with pytest.raises(KnowledgeContractError) as error:
        validate_knowledge_record(record)
    assert error.value.code == "KNOWLEDGE_SEMANTIC_VALIDATION_FAILED"


def test_catalog_binds_all_pack_schema_and_builder_checksums() -> None:
    summary = check_outputs()
    assert summary["recordCount"] == 49
    corpus = ElectronicsCorpus()
    assert len(corpus.records) == 49
    assert corpus.catalog["recordSchema"]["sha256"]
    assert corpus.catalog["builder"]["sha256"]
    assert all(pack["sha256"] for pack in corpus.catalog["packs"])


def test_pack_tampering_is_detected_before_lookup(tmp_path: Path) -> None:
    source_root = CATALOG_PATH.parent
    copied_root = tmp_path / "electronics_corpus"
    shutil.copytree(
        source_root,
        copied_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    pack = copied_root / "packs" / "v1" / "board.jsonl"
    pack.write_text(pack.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(KnowledgeContractError) as error:
        ElectronicsCorpus(copied_root / "catalog.v1.json")
    assert error.value.code == "KNOWLEDGE_PACK_CHECKSUM_MISMATCH"


def test_exact_variants_conflicts_and_unknowns_are_distinct() -> None:
    corpus = ElectronicsCorpus()
    uno = corpus.lookup("board", "ARDUINO_UNO")
    assert uno.status == "found"
    assert uno.records[0]["subject"]["variant"].startswith("R3")

    uno_family = corpus.lookup("board", "board-family:arduino:uno")
    assert uno_family.status == "ambiguous"
    assert uno_family.reasonCode == "variant-required"
    assert {item["subject"]["variant"] for item in uno_family.records} == {
        "R3 / A000066 / ATmega328P",
        "ABX00087 / RA4M1 with ESP32-S3-MINI-1",
    }

    esp_family = corpus.lookup("board", "board-family:espressif:esp32-devkitc")
    assert esp_family.status == "ambiguous"
    assert len(esp_family.records) == 2

    led = corpus.lookup("component", "LED")
    assert led.status == "unknown"
    assert led.reasonCode == "variant-required"
    unsupported = corpus.lookup("component", "MPU9999")
    assert unsupported.status == "unknown"
    assert unsupported.records == ()


def test_conflicting_hardware_revisions_remain_separate_in_report() -> None:
    report = json.loads(
        (CATALOG_PATH.parent / "reports" / "current-corpus-report.json").read_text(
            encoding="utf-8"
        )
    )
    groups = {item["conflictGroup"]: item["recordIds"] for item in report["conflictGroups"]}
    assert len(groups["conflict.board.arduino-uno"]) == 2
    assert len(groups["conflict.board.esp32-devkit"]) == 2
    assert len(groups["conflict.board.raspberry-pi-pico"]) == 2
    assert len(groups["conflict.component.ssd1306-breakout"]) == 2


def test_pin_router_uses_only_exact_curated_recipes() -> None:
    pinout = PinRouter.get_board_pinout("ARDUINO_UNO")
    assert pinout["status"] == "found"
    assert pinout["i2c_sda"] == "A4"
    assert pinout["effectiveRevision"] == "revision:1.0.0"

    recipe = PinRouter.resolve_connections_result(
        "ADAFRUIT_SSD1306_STEMMA_128X64", "ARDUINO_UNO"
    )
    assert recipe["status"] == "found"
    assert len(recipe["connections"]) == 4
    assert all(item["knowledgeRecordId"] == recipe["recordId"] for item in recipe["connections"])

    mega = PinRouter.get_board_pinout("ARDUINO_MEGA")
    assert mega["status"] == "found"
    assert mega["i2c_sda"] == "D20"

    unknown_board = PinRouter.get_board_pinout("NOT_A_BOARD")
    assert unknown_board == {
        "status": "unknown",
        "reasonCode": "unsupported-or-missing-evidence",
        "requestedBoard": "NOT_A_BOARD",
    }
    assert PinRouter.get_board_pinout("STM32_BLUE_PILL")["reasonCode"] == "variant-required"
    unknown_component = PinRouter.resolve_connections_result("RELAY", "ARDUINO_UNO")
    assert unknown_component["status"] == "unknown"
    assert unknown_component["connections"] == []
    assert PinRouter.resolve_connections("UNLISTED_SENSOR", "ARDUINO_UNO") == []


def test_firmware_memory_analysis_never_substitutes_uno_for_unknown_board() -> None:
    known = DeepFirmwareAnalyzer.analyze_memory_usage("int sample = 1;", "ARDUINO_UNO")
    assert known["status"] == "found"
    assert known["sramCapacity_KB"] == 2
    assert known["knowledgeRecordId"].endswith("board.arduino.uno-r3")

    mega = DeepFirmwareAnalyzer.analyze_memory_usage("int sample = 1;", "ARDUINO_MEGA")
    assert mega["status"] == "found"
    assert mega["sramCapacity_KB"] == 8

    unknown = DeepFirmwareAnalyzer.analyze_memory_usage("int sample = 1;", "NO_BOARD")
    assert unknown["status"] == "unknown"
    assert unknown["estimatedSRAM_bytes"] is None
    assert "flashCapacity_KB" not in unknown
    estimate = DeepFirmwareAnalyzer.estimate_binary_size("void setup() {}", "NO_BOARD")
    assert estimate["fitsInFlash"] is None
    assert estimate["flashUsagePercent"] is None


def test_curated_source_is_approved_for_runtime_generation_and_training_lineage() -> None:
    for usage in ("runtime-retrieval", "synthetic-generation", "training"):
        source = require_approved_source(
            "vf-src-curated-electronics-corpus-v1", usage
        )
        assert source["revision"] == "1.0.0"
        assert source["privacy"]["containsPrivateUserData"] is False
