"""Build an exact-variant coverage report for the UI board catalog.

The UI catalog is intentionally broader than the verified electronics corpus.
This module keeps that distinction explicit so a selectable canvas footprint is
never treated as an electrically verified AI target.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from electronics_corpus.schema import CORPUS_SOURCE_REVISION
from electronics_corpus.store import ElectronicsCorpus


CoverageStatus = Literal["verified", "variant-required", "unsupported"]
COVERAGE_REPORT_VERSION = "1.0.0"
UI_CATALOG_VERSION = "board-catalog-v1"

# Keep this inventory independent from the UI package. The verifier compares
# the two catalogs in CI, while the AI runtime remains deployable on its own.
_UI_BOARDS: tuple[tuple[str, str, str], ...] = (
    ("ARDUINO_UNO", "Arduino Uno R3", "Arduino"),
    ("ARDUINO_UNO_R4", "Arduino Uno R4", "Arduino"),
    ("ARDUINO_NANO", "Arduino Nano", "Arduino"),
    ("ARDUINO_NANO_EVERY", "Arduino Nano Every", "Arduino"),
    ("ARDUINO_NANO_33_IOT", "Arduino Nano 33 IoT", "Arduino"),
    ("ARDUINO_MEGA", "Arduino Mega 2560", "Arduino"),
    ("ARDUINO_LEONARDO", "Arduino Leonardo", "Arduino"),
    ("ARDUINO_MICRO", "Arduino Micro", "Arduino"),
    ("ARDUINO_DUE", "Arduino Due", "Arduino"),
    ("ARDUINO_GIGA_R1", "Arduino GIGA R1 WiFi", "Arduino"),
    ("ARDUINO_PORTENTA_H7", "Arduino Portenta H7", "Arduino"),
    ("ESP8266", "ESP8266 NodeMCU", "ESP8266"),
    ("ESP8266_WEMOS_D1_MINI", "WeMos D1 Mini", "ESP8266"),
    ("ESP8266_ESP01", "ESP-01", "ESP8266"),
    ("ESP8266_ESP12E", "ESP-12E", "ESP8266"),
    ("ESP32", "ESP32 DevKit V1", "ESP32"),
    ("ESP32_WROOM", "ESP32-WROOM", "ESP32"),
    ("ESP32_WROVER", "ESP32-WROVER", "ESP32"),
    ("ESP32_S2", "ESP32-S2", "ESP32"),
    ("ESP32_S3", "ESP32-S3", "ESP32"),
    ("ESP32_C3", "ESP32-C3", "ESP32"),
    ("ESP32_C6", "ESP32-C6", "ESP32"),
    ("ESP32_H2", "ESP32-H2", "ESP32"),
    ("ESP32_TTGO", "TTGO ESP32", "ESP32"),
    ("ESP32_LILYGO", "LilyGO ESP32", "ESP32"),
    ("ESP32_M5STACK", "M5Stack ESP32", "ESP32"),
    ("RASPBERRY_PI_PICO", "Raspberry Pi Pico", "Raspberry Pi Pico"),
    ("RASPBERRY_PI_PICO_W", "Raspberry Pi Pico W", "Raspberry Pi Pico"),
    ("RASPBERRY_PI_PICO_2", "Raspberry Pi Pico 2", "Raspberry Pi Pico"),
    ("STM32_BLUE_PILL", "STM32 Blue Pill", "STM32"),
    ("STM32_BLACK_PILL", "STM32 Black Pill", "STM32"),
    ("TEENSY_4_0", "Teensy 4.0", "Teensy"),
    ("TEENSY_4_1", "Teensy 4.1", "Teensy"),
    ("TEENSY_LC", "Teensy LC", "Teensy"),
    ("SEEED_XIAO_SAMD21", "Seeed XIAO SAMD21", "Seeed XIAO"),
    ("SEEED_XIAO_RP2040", "Seeed XIAO RP2040", "Seeed XIAO"),
    ("SEEED_XIAO_ESP32C3", "Seeed XIAO ESP32C3", "Seeed XIAO"),
    ("SEEED_XIAO_ESP32S3", "Seeed XIAO ESP32S3", "Seeed XIAO"),
    ("SEEED_XIAO_NRF52840", "Seeed XIAO nRF52840", "Seeed XIAO"),
    ("ADAFRUIT_FEATHER_M0", "Adafruit Feather M0", "Adafruit Feather"),
    ("ADAFRUIT_FEATHER_M4", "Adafruit Feather M4", "Adafruit Feather"),
    ("ADAFRUIT_FEATHER_ESP32", "Adafruit Feather ESP32", "Adafruit Feather"),
    ("ADAFRUIT_FEATHER_RP2040", "Adafruit Feather RP2040", "Adafruit Feather"),
    ("ADAFRUIT_FEATHER_NRF52840", "Adafruit Feather nRF52840", "Adafruit Feather"),
    ("SPARKFUN_THING_PLUS_ESP32", "SparkFun Thing Plus ESP32", "SparkFun Thing Plus"),
    ("SPARKFUN_THING_PLUS_RP2040", "SparkFun Thing Plus RP2040", "SparkFun Thing Plus"),
    ("SPARKFUN_THING_PLUS_ARTEMIS", "SparkFun Thing Plus Artemis", "SparkFun Thing Plus"),
    ("ATMEL_AVR_ATMEGA328P", "ATmega328P DIP", "Atmel AVR"),
    ("ATMEL_AVR_ATTINY", "ATtiny Series DIP", "Atmel AVR"),
)

_VERIFIED: dict[str, tuple[str, str]] = {
    "ARDUINO_UNO": ("vf-knowledge-v1-board.arduino.uno-r3", "R3 / A000066 / ATmega328P"),
    "ARDUINO_UNO_R4": ("vf-knowledge-v1-board.arduino.uno-r4-wifi", "ABX00087 / RA4M1 with ESP32-S3-MINI-1"),
    "ARDUINO_NANO_EVERY": ("vf-knowledge-v1-board.arduino.nano-every-atmega4809", "ABX00028 / ATmega4809"),
    "ARDUINO_MEGA": ("vf-knowledge-v1-board.arduino.mega-2560-r3", "Rev3 / A000067 / ATmega2560"),
    "ARDUINO_LEONARDO": ("vf-knowledge-v1-board.arduino.leonardo-atmega32u4", "A000057 / ATmega32U4"),
    "ARDUINO_MICRO": ("vf-knowledge-v1-board.arduino.micro-atmega32u4", "A000053 / ATmega32U4"),
    "RASPBERRY_PI_PICO": ("vf-knowledge-v1-board.raspberry-pi.pico-rp2040", "Pico 1 / RP2040 / non-wireless"),
    "RASPBERRY_PI_PICO_2": ("vf-knowledge-v1-board.raspberry-pi.pico2-rp2350", "Pico 2 / RP2350 / non-wireless"),
}

_VARIANT_REQUIRED: dict[str, tuple[str | None, str]] = {
    "ARDUINO_NANO": ("vf-knowledge-v1-board.generic.arduino-nano", "Select Nano classic, Nano Every, Nano 33 IoT, and the carrier/revision."),
    "ESP8266": ("vf-knowledge-v1-board.generic.esp8266-development-board", "Select the exact ESP8266 module and carrier-board revision."),
    "ESP8266_WEMOS_D1_MINI": (None, "WeMos D1 Mini board revision and ESP-12 module variant are not selected."),
    "ESP8266_ESP01": (None, "ESP-01 flash size, module revision, and carrier are not selected."),
    "ESP8266_ESP12E": (None, "ESP-12E module revision and carrier are not selected."),
    "ESP32": ("vf-knowledge-v1-board.generic.esp32-development-board", "Select the exact ESP32 module, flash density, carrier, and board revision."),
    "ESP32_WROOM": (None, "ESP32-WROOM module revision, flash density, and carrier are not selected."),
    "ESP32_WROVER": (None, "ESP32-WROVER module revision, PSRAM, flash density, and carrier are not selected."),
    "ESP32_S2": (None, "ESP32-S2 module and carrier-board revision are not selected."),
    "ESP32_S3": ("vf-knowledge-v1-board.generic.esp32-s3-development-board", "Select the exact ESP32-S3 module, flash density, carrier, and board revision."),
    "ESP32_C3": (None, "ESP32-C3 module, flash density, carrier, and revision are not selected."),
    "ESP32_C6": (None, "ESP32-C6 module, flash density, carrier, and revision are not selected."),
    "ESP32_H2": (None, "ESP32-H2 module, flash density, carrier, and revision are not selected."),
    "ESP32_TTGO": (None, "TTGO product SKU and revision are not selected."),
    "ESP32_LILYGO": (None, "LilyGO product SKU and revision are not selected."),
    "ESP32_M5STACK": (None, "M5Stack product SKU, module, and revision are not selected."),
    "STM32_BLUE_PILL": ("vf-knowledge-v1-board.generic.stm32-blue-pill", "Select manufacturer, MCU marking, flash density, USB configuration, and revision."),
    "STM32_BLACK_PILL": (None, "Select the exact STM32F401/F411 MCU marking, carrier, and revision."),
    "SPARKFUN_THING_PLUS_ESP32": (None, "Select the exact SparkFun product SKU and revision."),
    "SPARKFUN_THING_PLUS_RP2040": (None, "Select the exact SparkFun product SKU and revision."),
    "SPARKFUN_THING_PLUS_ARTEMIS": (None, "Select the exact SparkFun product SKU and revision."),
    "ATMEL_AVR_ATTINY": (None, "Select the exact ATtiny device, package, fuse configuration, and board wiring."),
}

_UNSUPPORTED_REASONS: dict[str, str] = {
    "ARDUINO_NANO_33_IOT": "No exact Arduino Nano 33 IoT board record is curated yet.",
    "ARDUINO_DUE": "No exact Arduino Due board record is curated yet.",
    "ARDUINO_GIGA_R1": "No exact Arduino GIGA R1 WiFi board record is curated yet.",
    "ARDUINO_PORTENTA_H7": "No exact Arduino Portenta H7 board record is curated yet.",
    "RASPBERRY_PI_PICO_W": "Wireless Pico W requires its exact board revision and wireless firmware boundary.",
    "TEENSY_4_0": "No exact Teensy 4.0 board record is curated yet.",
    "TEENSY_4_1": "No exact Teensy 4.1 board record is curated yet.",
    "TEENSY_LC": "No exact Teensy LC board record is curated yet.",
    "SEEED_XIAO_SAMD21": "No exact Seeed XIAO SAMD21 SKU and revision record is curated yet.",
    "SEEED_XIAO_RP2040": "No exact Seeed XIAO RP2040 SKU and revision record is curated yet.",
    "SEEED_XIAO_ESP32C3": "No exact Seeed XIAO ESP32C3 SKU and revision record is curated yet.",
    "SEEED_XIAO_ESP32S3": "No exact Seeed XIAO ESP32S3 SKU and revision record is curated yet.",
    "SEEED_XIAO_NRF52840": "No exact Seeed XIAO nRF52840 SKU and revision record is curated yet.",
    "ADAFRUIT_FEATHER_M0": "No exact Adafruit Feather M0 product SKU and revision record is curated yet.",
    "ADAFRUIT_FEATHER_M4": "No exact Adafruit Feather M4 product SKU and revision record is curated yet.",
    "ADAFRUIT_FEATHER_ESP32": "No exact Adafruit Feather ESP32 product SKU and revision record is curated yet.",
    "ADAFRUIT_FEATHER_RP2040": "No exact Adafruit Feather RP2040 product SKU and revision record is curated yet.",
    "ADAFRUIT_FEATHER_NRF52840": "No exact Adafruit Feather nRF52840 product SKU and revision record is curated yet.",
    "ATMEL_AVR_ATMEGA328P": "A DIP MCU is not a board; package, clock, fuses, carrier, and wiring must be selected first.",
}


def _catalog_sha256(corpus: ElectronicsCorpus) -> str:
    return hashlib.sha256(corpus.catalog_path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _build_entries(corpus: ElectronicsCorpus) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for board_type, display_name, family in _UI_BOARDS:
        if board_type in _VERIFIED:
            record_id, variant = _VERIFIED[board_type]
            record = corpus.by_id.get(record_id)
            if record is None or record["recordType"] != "board" or record["supportStatus"] != "supported":
                raise RuntimeError(f"Verified coverage record is unavailable or unsupported: {record_id}")
            if record["subject"]["variant"] != variant:
                raise RuntimeError(f"Verified coverage variant drifted: {record_id}")
            entries.append({
                "boardType": board_type,
                "displayName": display_name,
                "family": family,
                "status": "verified",
                "reasonCode": "exact-evidenced-variant",
                "reason": "Exact board variant, electrical profile, and manufacturer evidence are curated.",
                "recordId": record_id,
                "recordRevision": record["effectiveRevision"]["revision"],
                "variant": variant,
            })
        elif board_type in _VARIANT_REQUIRED:
            record_id, reason = _VARIANT_REQUIRED[board_type]
            entry: dict[str, Any] = {
                "boardType": board_type,
                "displayName": display_name,
                "family": family,
                "status": "variant-required",
                "reasonCode": "exact-variant-required",
                "reason": reason,
            }
            if record_id is not None:
                record = corpus.by_id.get(record_id)
                if record is None or record["supportStatus"] != "variant-required":
                    raise RuntimeError(f"Variant gate record is unavailable: {record_id}")
                entry["recordId"] = record_id
                entry["recordRevision"] = record["effectiveRevision"]["revision"]
            entries.append(entry)
        else:
            entries.append({
                "boardType": board_type,
                "displayName": display_name,
                "family": family,
                "status": "unsupported",
                "reasonCode": "curation-not-promoted",
                "reason": _UNSUPPORTED_REASONS[board_type],
            })
    return entries


@lru_cache(maxsize=1)
def validate_hardware_coverage() -> dict[str, Any]:
    corpus = ElectronicsCorpus()
    entries = _build_entries(corpus)
    if len(entries) != len(_UI_BOARDS) or len({item["boardType"] for item in entries}) != len(entries):
        raise RuntimeError("Hardware coverage must contain one unique entry per UI board type.")
    counts = Counter(item["status"] for item in entries)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-001-ui-hardware-coverage",
        "reportVersion": COVERAGE_REPORT_VERSION,
        "uiCatalogVersion": UI_CATALOG_VERSION,
        "corpusVersion": CORPUS_SOURCE_REVISION,
        "corpusCatalogSha256": _catalog_sha256(corpus),
        "asOfDate": "2026-08-31",
        "entryCount": len(entries),
        "summary": {
            "verified": counts["verified"],
            "variantRequired": counts["variant-required"],
            "unsupported": counts["unsupported"],
        },
        "entries": entries,
    }
    report["reportSha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def get_hardware_coverage() -> dict[str, Any]:
    """Return a defensive copy of the validated coverage report."""

    return json.loads(json.dumps(validate_hardware_coverage()))
