"""Deterministic source for the curated VoltForge electronics corpus v1."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from electronics_corpus.schema import (
    CORPUS_CONTRACT_VERSION,
    CORPUS_ROOT,
    KNOWLEDGE_SCHEMA_PATH,
    build_knowledge_json_schema,
    validate_knowledge_record,
)


BUILDER_ID = "vf-curated-electronics-corpus-builder"
BUILDER_VERSION = "1.0.0"
SOURCE_ID = "vf-src-curated-electronics-corpus-v1"
EFFECTIVE_FROM = "2026-08-28"
VERIFIED_AT = "2026-08-28"
PACK_ORDER = (
    "board",
    "pin-map",
    "component",
    "wiring-recipe",
    "firmware-api",
    "compiler-diagnostic",
    "simulation-behavior",
    "safety-constraint",
)
PACK_FILENAMES = {record_type: f"packs/v1/{record_type}.jsonl" for record_type in PACK_ORDER}

UNO_R3_ID = "vf-knowledge-v1-board.arduino.uno-r3"
UNO_R4_ID = "vf-knowledge-v1-board.arduino.uno-r4-wifi"
MEGA_ID = "vf-knowledge-v1-board.arduino.mega-2560-r3"
NANO_ID = "vf-knowledge-v1-board.arduino.nano-classic-atmega328p"
ESP32_ID = "vf-knowledge-v1-board.espressif.esp32-devkitc-v4-wroom32e-n4"
ESP32_S3_ID = "vf-knowledge-v1-board.espressif.esp32-s3-devkitc1-n8"
PICO_ID = "vf-knowledge-v1-board.raspberry-pi.pico-rp2040"
PICO2_ID = "vf-knowledge-v1-board.raspberry-pi.pico2-rp2350"
OLED_NEW_ID = "vf-knowledge-v1-component.adafruit.ssd1306-128x64-stemma"
OLED_OLD_ID = "vf-knowledge-v1-component.adafruit.ssd1306-128x64-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def evidence(
    evidence_id: str,
    *,
    kind: str,
    publisher: str,
    title: str,
    revision: str,
    locator: str,
    uri: str | None = None,
) -> dict[str, Any]:
    return {
        "evidenceId": evidence_id,
        "evidenceKind": kind,
        "publisher": publisher,
        "title": title,
        "documentRevision": revision,
        "locator": locator,
        "uri": uri,
        "verifiedAt": VERIFIED_AT,
        "extractionPolicy": "normalized-facts-only-no-source-prose",
    }


def project_evidence(evidence_id: str, title: str, locator: str) -> dict[str, Any]:
    return evidence(
        evidence_id,
        kind="project-engineering-rule",
        publisher="VoltForge",
        title=title,
        revision=BUILDER_VERSION,
        locator=locator,
    )


def claim(
    property_name: str,
    value: Any,
    evidence_refs: Iterable[str],
    *,
    status: str = "verified",
    unit: str | None = None,
    conditions: Iterable[str] = (),
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "claimId": f"claim.{property_name}",
        "property": property_name,
        "value": value,
        "status": status,
        "conditions": list(conditions),
        "evidenceRefs": list(evidence_refs),
    }
    if unit is not None:
        item["unit"] = unit
    return item


def relation(relation_type: str, target: str) -> dict[str, str]:
    return {"relationType": relation_type, "targetRecordId": target}


def record(
    record_type: str,
    slug: str,
    *,
    subject_id: str,
    family_id: str,
    name: str,
    variant: str,
    aliases: Iterable[str],
    claims: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    support_status: str = "supported",
    conflict_group: str | None = None,
    relations: Iterable[dict[str, str]] = (),
    tags: Iterable[str] = (),
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schemaVersion": 1,
        "contractVersion": CORPUS_CONTRACT_VERSION,
        "recordId": f"vf-knowledge-v1-{slug}",
        "recordType": record_type,
        "supportStatus": support_status,
        "subject": {
            "subjectId": subject_id,
            "familyId": family_id,
            "name": name,
            "variant": variant,
            "aliases": list(aliases),
        },
        "effectiveRevision": {
            "revision": "revision:1.0.0",
            "validFrom": EFFECTIVE_FROM,
            "validTo": None,
            "supersedes": None,
        },
        "provenance": {
            "sourceId": SOURCE_ID,
            "sourceRevision": CORPUS_CONTRACT_VERSION,
            "curator": "VoltForge",
            "evidence": evidence_items,
        },
        "claims": claims,
        "relations": list(relations),
        "tags": sorted(set(tags)),
    }
    if conflict_group is not None:
        value["conflictGroup"] = conflict_group
    return value


def _board_records() -> list[dict[str, Any]]:
    uno_r3 = evidence(
        "evidence.arduino.uno-r3",
        kind="manufacturer-documentation",
        publisher="Arduino",
        title="Arduino UNO R3 hardware documentation and datasheet",
        revision="A000066-current-at-2026-08-28",
        locator="Tech Specs and pinout",
        uri="https://docs.arduino.cc/hardware/uno-rev3/",
    )
    uno_r4 = evidence(
        "evidence.arduino.uno-r4-wifi",
        kind="manufacturer-documentation",
        publisher="Arduino",
        title="Arduino UNO R4 WiFi hardware documentation",
        revision="ABX00087-current-at-2026-08-28",
        locator="Tech Specs and full pinout",
        uri="https://docs.arduino.cc/hardware/uno-r4-wifi/",
    )
    mega = evidence(
        "evidence.arduino.mega-2560-r3",
        kind="manufacturer-documentation",
        publisher="Arduino",
        title="Arduino Mega 2560 Rev3 hardware documentation",
        revision="A000067-current-at-2026-08-28",
        locator="Tech Specs and pinout",
        uri="https://docs.arduino.cc/hardware/mega-2560/",
    )
    nano = evidence(
        "evidence.arduino.nano-classic",
        kind="manufacturer-documentation",
        publisher="Arduino",
        title="Arduino Nano classic hardware documentation",
        revision="A000005-current-at-2026-08-28",
        locator="Tech Specs and pinout",
        uri="https://docs.arduino.cc/hardware/nano/",
    )
    esp32 = evidence(
        "evidence.espressif.esp32-devkitc-v4",
        kind="manufacturer-documentation",
        publisher="Espressif Systems",
        title="ESP32-DevKitC V4 user guide",
        revision="latest-verified-2026-08-28",
        locator="Board variants, header blocks, pin layout, and power options",
        uri="https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32/esp32-devkitc/user_guide.html",
    )
    esp32_s3 = evidence(
        "evidence.espressif.esp32-s3-devkitc1",
        kind="manufacturer-documentation",
        publisher="Espressif Systems",
        title="ESP32-S3-DevKitC-1 user guide",
        revision="latest-verified-2026-08-28",
        locator="Module variants and pin layout",
        uri="https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide.html",
    )
    pico = evidence(
        "evidence.raspberry-pi.pico-series",
        kind="manufacturer-documentation",
        publisher="Raspberry Pi Ltd",
        title="Pico-series microcontroller board documentation",
        revision="current-at-2026-08-28",
        locator="Pico and Pico 2 family specifications and board documentation",
        uri="https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html",
    )
    specs = [
        (
            "board.arduino.uno-r3",
            "board:arduino:uno-r3",
            "board-family:arduino:uno",
            "Arduino UNO R3",
            "R3 / A000066 / ATmega328P",
            ["ARDUINO_UNO", "UNO_R3", "A000066"],
            "conflict.board.arduino-uno",
            uno_r3,
            {
                "manufacturer": "Arduino",
                "mcu": "ATmega328P",
                "architecture": "8-bit AVR",
                "logic-voltage-v": 5.0,
                "flash-bytes": 32768,
                "sram-bytes": 2048,
                "clock-hz": 16000000,
                "digital-io-count": 14,
                "analog-input-count": 6,
            },
        ),
        (
            "board.arduino.uno-r4-wifi",
            "board:arduino:uno-r4-wifi",
            "board-family:arduino:uno",
            "Arduino UNO R4 WiFi",
            "ABX00087 / RA4M1 with ESP32-S3-MINI-1",
            ["ARDUINO_UNO_R4", "UNO_R4_WIFI", "ABX00087"],
            "conflict.board.arduino-uno",
            uno_r4,
            {
                "manufacturer": "Arduino",
                "mcu": "Renesas RA4M1",
                "architecture": "32-bit Arm Cortex-M4",
                "logic-voltage-v": 5.0,
                "flash-bytes": 262144,
                "sram-bytes": 32768,
                "clock-hz": 48000000,
                "digital-io-count": 14,
                "analog-input-count": 6,
            },
        ),
        (
            "board.arduino.mega-2560-r3",
            "board:arduino:mega-2560-r3",
            "board-family:arduino:mega",
            "Arduino Mega 2560 Rev3",
            "Rev3 / A000067 / ATmega2560",
            ["ARDUINO_MEGA", "ARDUINO_MEGA_2560", "A000067"],
            None,
            mega,
            {
                "manufacturer": "Arduino",
                "mcu": "ATmega2560",
                "architecture": "8-bit AVR",
                "logic-voltage-v": 5.0,
                "flash-bytes": 262144,
                "sram-bytes": 8192,
                "clock-hz": 16000000,
                "digital-io-count": 54,
                "analog-input-count": 16,
            },
        ),
        (
            "board.arduino.nano-classic-atmega328p",
            "board:arduino:nano-classic-atmega328p",
            "board-family:arduino:nano",
            "Arduino Nano classic",
            "A000005 / ATmega328P",
            ["ARDUINO_NANO", "ARDUINO_NANO_CLASSIC", "A000005"],
            None,
            nano,
            {
                "manufacturer": "Arduino",
                "mcu": "ATmega328P",
                "architecture": "8-bit AVR",
                "logic-voltage-v": 5.0,
                "flash-bytes": 32768,
                "sram-bytes": 2048,
                "clock-hz": 16000000,
                "digital-io-count": 22,
                "analog-input-count": 8,
            },
        ),
        (
            "board.espressif.esp32-devkitc-v4-wroom32e-n4",
            "board:espressif:esp32-devkitc-v4-wroom32e-n4",
            "board-family:espressif:esp32-devkitc",
            "ESP32-DevKitC V4 with ESP32-WROOM-32E-N4",
            "V4 / ESP32-WROOM-32E-N4",
            ["ESP32", "ESP32_DEVKITC_V4_WROOM32E_N4", "ESP32_WROOM_32E_N4"],
            "conflict.board.esp32-devkit",
            esp32,
            {
                "manufacturer": "Espressif Systems",
                "mcu": "ESP32 / ESP32-WROOM-32E-N4",
                "architecture": "dual-core Xtensa LX6",
                "logic-voltage-v": 3.3,
                "flash-bytes": 4194304,
                "sram-bytes": 532480,
                "clock-hz": 240000000,
                "module-variant-required": True,
            },
        ),
        (
            "board.espressif.esp32-s3-devkitc1-n8",
            "board:espressif:esp32-s3-devkitc1-n8",
            "board-family:espressif:esp32-devkitc",
            "ESP32-S3-DevKitC-1 with ESP32-S3-WROOM-1-N8",
            "DevKitC-1 / ESP32-S3-WROOM-1-N8",
            ["ESP32_S3", "ESP32_S3_DEVKITC_1_N8"],
            "conflict.board.esp32-devkit",
            esp32_s3,
            {
                "manufacturer": "Espressif Systems",
                "mcu": "ESP32-S3 / ESP32-S3-WROOM-1-N8",
                "architecture": "dual-core Xtensa LX7",
                "logic-voltage-v": 3.3,
                "flash-bytes": 8388608,
                "sram-bytes": 524288,
                "clock-hz": 240000000,
                "module-variant-required": True,
            },
        ),
        (
            "board.raspberry-pi.pico-rp2040",
            "board:raspberry-pi:pico-rp2040",
            "board-family:raspberry-pi:pico",
            "Raspberry Pi Pico",
            "Pico 1 / RP2040 / non-wireless",
            ["RASPBERRY_PI_PICO", "PICO_RP2040"],
            "conflict.board.raspberry-pi-pico",
            pico,
            {
                "manufacturer": "Raspberry Pi Ltd",
                "mcu": "RP2040",
                "architecture": "dual-core Arm Cortex-M0+",
                "logic-voltage-v": 3.3,
                "flash-bytes": 2097152,
                "sram-bytes": 270336,
                "clock-hz": 133000000,
                "digital-io-count": 26,
                "analog-input-count": 3,
            },
        ),
        (
            "board.raspberry-pi.pico2-rp2350",
            "board:raspberry-pi:pico2-rp2350",
            "board-family:raspberry-pi:pico",
            "Raspberry Pi Pico 2",
            "Pico 2 / RP2350 / non-wireless",
            ["RASPBERRY_PI_PICO_2", "PICO2_RP2350"],
            "conflict.board.raspberry-pi-pico",
            pico,
            {
                "manufacturer": "Raspberry Pi Ltd",
                "mcu": "RP2350",
                "architecture": "dual-core Arm Cortex-M33 or Hazard3",
                "logic-voltage-v": 3.3,
                "flash-bytes": 4194304,
                "sram-bytes": 532480,
                "clock-hz": 150000000,
                "digital-io-count": 26,
                "analog-input-count": 3,
            },
        ),
    ]
    records = []
    for slug, subject_id, family_id, name, variant, aliases, conflict, ev, values in specs:
        refs = [ev["evidenceId"]]
        claims = [claim("support-status", "supported", refs)]
        claims.extend(
            claim(key, value, refs, unit="V" if key == "logic-voltage-v" else None)
            for key, value in values.items()
        )
        records.append(
            record(
                "board",
                slug,
                subject_id=subject_id,
                family_id=family_id,
                name=name,
                variant=variant,
                aliases=aliases,
                claims=claims,
                evidence_items=[ev],
                conflict_group=conflict,
                tags=("board", values["manufacturer"].lower().replace(" ", "-")),
            )
        )
    unresolved = project_evidence(
        "evidence.vf.board-variant-selection",
        "VoltForge exact board-variant selection policy",
        "electronics_corpus/builder.py::_board_records",
    )
    unresolved_rows = [
        (
            "board.generic.stm32-blue-pill",
            "board:generic:stm32-blue-pill",
            "board-family:generic:stm32-blue-pill",
            "Generic STM32 Blue Pill",
            "manufacturer, MCU marking, flash density, USB resistor, and clone revision not selected",
            ("STM32_BLUE_PILL", "BLUE_PILL"),
        ),
        (
            "board.generic.esp8266-development-board",
            "board:generic:esp8266-development-board",
            "board-family:generic:esp8266-development-board",
            "Generic ESP8266 development board",
            "module and carrier-board revision not selected",
            ("ESP8266", "ESP8266_DEV_BOARD"),
        ),
    ]
    for slug, subject_id, family_id, name, variant, aliases in unresolved_rows:
        refs = [unresolved["evidenceId"]]
        records.append(
            record(
                "board",
                slug,
                subject_id=subject_id,
                family_id=family_id,
                name=name,
                variant=variant,
                aliases=aliases,
                claims=[
                    claim("support-status", "variant-required", refs),
                    claim("manufacturer", None, refs, status="unknown"),
                    claim("mcu", None, refs, status="unknown"),
                    claim("architecture", None, refs, status="unknown"),
                    claim("logic-voltage-v", None, refs, status="unknown", unit="V"),
                    claim("flash-bytes", None, refs, status="unknown"),
                    claim("sram-bytes", None, refs, status="unknown"),
                ],
                evidence_items=[unresolved],
                support_status="variant-required",
                tags=("board", "variant-required", "unknown-ratings"),
            )
        )
    return records


def _pin_map_records() -> list[dict[str, Any]]:
    source = {
        UNO_R3_ID: (
            "pin-map.arduino.uno-r3",
            "pin-map:arduino:uno-r3",
            "Arduino UNO R3 pin map",
            "R3 headers",
            ["ARDUINO_UNO", "UNO_R3"],
            "evidence.pinout.uno-r3",
            "Arduino UNO R3 full pinout",
            "https://docs.arduino.cc/resources/pinouts/A000066-full-pinout.pdf",
            {"i2c": {"sda": "A4", "scl": "A5"}, "spi": {"copi": "D11", "cipo": "D12", "sck": "D13", "cs": "D10"}, "uart0": {"tx": "D1", "rx": "D0"}},
            {"5V": ["power-5v"], "3V3": ["power-3v3"], "GND": ["ground"], "A0-A5": ["adc"], "D3,D5,D6,D9,D10,D11": ["pwm"]},
        ),
        UNO_R4_ID: (
            "pin-map.arduino.uno-r4-wifi",
            "pin-map:arduino:uno-r4-wifi",
            "Arduino UNO R4 WiFi pin map",
            "ABX00087 main headers and Qwiic",
            ["ARDUINO_UNO_R4", "UNO_R4_WIFI"],
            "evidence.pinout.uno-r4",
            "Arduino UNO R4 WiFi full pinout",
            "https://docs.arduino.cc/resources/pinouts/ABX00087-full-pinout.pdf",
            {"i2c-main": {"sda": "A4", "scl": "A5"}, "i2c-qwiic": {"sda": "QWIIC_SDA", "scl": "QWIIC_SCL", "logicV": 3.3}, "spi": {"copi": "D11", "cipo": "D12", "sck": "D13", "cs": "D10"}, "uart0": {"tx": "D1", "rx": "D0"}},
            {"5V": ["power-5v"], "3V3": ["power-3v3"], "GND": ["ground"], "QWIIC": ["i2c", "3v3-only"], "ESP_HEADER": ["3v3-only"]},
        ),
        MEGA_ID: (
            "pin-map.arduino.mega-2560-r3",
            "pin-map:arduino:mega-2560-r3",
            "Arduino Mega 2560 Rev3 pin map",
            "Rev3 / A000067 headers",
            ["ARDUINO_MEGA", "ARDUINO_MEGA_2560"],
            "evidence.pinout.mega-2560-r3",
            "Arduino Mega 2560 Rev3 full pinout",
            "https://docs.arduino.cc/resources/pinouts/A000067-full-pinout.pdf",
            {"i2c": {"sda": "D20", "scl": "D21"}, "spi": {"copi": "D51", "cipo": "D50", "sck": "D52", "cs": "D53"}, "uart0": {"tx": "D1", "rx": "D0"}},
            {"5V": ["power-5v"], "3V3": ["power-3v3"], "GND": ["ground"], "A0-A15": ["adc"], "D2-D13,D44-D46": ["pwm"]},
        ),
        NANO_ID: (
            "pin-map.arduino.nano-classic-atmega328p",
            "pin-map:arduino:nano-classic-atmega328p",
            "Arduino Nano classic pin map",
            "A000005 / ATmega328P headers",
            ["ARDUINO_NANO", "ARDUINO_NANO_CLASSIC"],
            "evidence.pinout.nano-classic",
            "Arduino Nano classic full pinout",
            "https://docs.arduino.cc/resources/pinouts/A000005-full-pinout.pdf",
            {"i2c": {"sda": "A4", "scl": "A5"}, "spi": {"copi": "D11", "cipo": "D12", "sck": "D13", "cs": "D10"}, "uart0": {"tx": "TX1", "rx": "RX0"}},
            {"5V": ["power-5v"], "3V3": ["power-3v3"], "GND": ["ground"], "A0-A7": ["adc"], "D3,D5,D6,D9,D10,D11": ["pwm"]},
        ),
        ESP32_ID: (
            "pin-map.espressif.esp32-devkitc-v4-wroom32e-n4",
            "pin-map:espressif:esp32-devkitc-v4-wroom32e-n4",
            "ESP32-DevKitC V4 WROOM-32E-N4 pin map",
            "V4 WROOM header variant",
            ["ESP32_DEVKITC_V4_WROOM32E_N4", "ESP32_WROOM_32E_N4"],
            "evidence.pinout.esp32-devkitc-v4",
            "ESP32-DevKitC V4 header blocks",
            "https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32/esp32-devkitc/user_guide.html#header-block",
            {"i2c-project-default": {"sda": "GPIO21", "scl": "GPIO22"}, "spi-vspi-project-default": {"copi": "GPIO23", "cipo": "GPIO19", "sck": "GPIO18", "cs": "GPIO5"}, "uart0": {"tx": "GPIO1", "rx": "GPIO3"}},
            {"3V3": ["power-3v3"], "5V": ["power-input"], "GND": ["ground"], "GPIO34-GPIO39": ["input-only"], "GPIO6-GPIO11": ["flash-reserved"]},
        ),
        ESP32_S3_ID: (
            "pin-map.espressif.esp32-s3-devkitc1-n8",
            "pin-map:espressif:esp32-s3-devkitc1-n8",
            "ESP32-S3-DevKitC-1 N8 pin map",
            "WROOM-1 N8 module variant",
            ["ESP32_S3", "ESP32_S3_DEVKITC_1_N8"],
            "evidence.pinout.esp32-s3-devkitc1",
            "ESP32-S3-DevKitC-1 pin layout",
            "https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide.html#pin-layout",
            {"i2c-project-default": {"sda": "GPIO8", "scl": "GPIO9"}, "spi-project-default": {"copi": "GPIO11", "cipo": "GPIO13", "sck": "GPIO12", "cs": "GPIO10"}, "uart0": {"tx": "GPIO43", "rx": "GPIO44"}},
            {"3V3": ["power-3v3"], "5V": ["power-input"], "GND": ["ground"], "GPIO19,GPIO20": ["usb-native"]},
        ),
        PICO_ID: (
            "pin-map.raspberry-pi.pico-rp2040",
            "pin-map:raspberry-pi:pico-rp2040",
            "Raspberry Pi Pico RP2040 pin map",
            "Pico 1 non-wireless",
            ["RASPBERRY_PI_PICO", "PICO_RP2040"],
            "evidence.pinout.pico-rp2040",
            "Raspberry Pi Pico datasheet and pinout",
            "https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf",
            {"i2c0-project-default": {"sda": "GP4", "scl": "GP5"}, "spi0-project-default": {"copi": "GP19", "cipo": "GP16", "sck": "GP18", "cs": "GP17"}, "uart0-project-default": {"tx": "GP0", "rx": "GP1"}},
            {"3V3_OUT": ["power-3v3"], "VSYS": ["power-input"], "GND": ["ground"], "GP26-GP28": ["adc"], "GP0-GP22,GP26-GP28": ["gpio"]},
        ),
        PICO2_ID: (
            "pin-map.raspberry-pi.pico2-rp2350",
            "pin-map:raspberry-pi:pico2-rp2350",
            "Raspberry Pi Pico 2 RP2350 pin map",
            "Pico 2 non-wireless",
            ["RASPBERRY_PI_PICO_2", "PICO2_RP2350"],
            "evidence.pinout.pico2-rp2350",
            "Raspberry Pi Pico 2 datasheet and pinout",
            "https://datasheets.raspberrypi.com/pico/pico-2-datasheet.pdf",
            {"i2c0-project-default": {"sda": "GP4", "scl": "GP5"}, "spi0-project-default": {"copi": "GP19", "cipo": "GP16", "sck": "GP18", "cs": "GP17"}, "uart0-project-default": {"tx": "GP0", "rx": "GP1"}},
            {"3V3_OUT": ["power-3v3"], "VSYS": ["power-input"], "GND": ["ground"], "GP26-GP28": ["adc"], "GP0-GP22,GP26-GP28": ["gpio"]},
        ),
    }
    records = []
    for board_id, item in source.items():
        slug, subject_id, name, variant, aliases, ev_id, title, uri, buses, capabilities = item
        ev = evidence(
            ev_id,
            kind="manufacturer-documentation",
            publisher=name.split(" ")[0] if "ESP32" not in name else "Espressif Systems",
            title=title,
            revision="current-at-2026-08-28",
            locator="Connector pin map and alternate functions",
            uri=uri,
        )
        refs = [ev_id]
        records.append(
            record(
                "pin-map",
                slug,
                subject_id=subject_id,
                family_id=board_id.replace("vf-knowledge-v1-board.", "pin-map-family:"),
                name=name,
                variant=variant,
                aliases=aliases,
                claims=[
                    claim("board-record-id", board_id, refs),
                    claim("power-pins", {key: value for key, value in capabilities.items() if any("power" in cap or cap == "ground" for cap in value)}, refs),
                    claim("default-buses", buses, refs, status="conditional", conditions=("Project defaults; verify alternate-function routing before changing pins.",)),
                    claim("pin-capabilities", capabilities, refs),
                ],
                evidence_items=[ev],
                relations=(relation("describes", board_id),),
                tags=("pin-map", "board"),
            )
        )
    return records


def _component_records() -> list[dict[str, Any]]:
    adafruit = evidence(
        "evidence.adafruit.oled-breakouts",
        kind="maintainer-documentation",
        publisher="Adafruit Industries",
        title="Monochrome OLED Breakouts",
        revision="last-edited-2025-12-18",
        locator="128x64 I2C/SPI variants, power, reset, and wiring",
        uri="https://learn.adafruit.com/monochrome-oled-breakouts/wiring-128x64-oleds",
    )
    project = project_evidence(
        "evidence.vf.component-selection",
        "VoltForge exact-component selection policy",
        "electronics_corpus/builder.py::_component_records",
    )
    return [
        record(
            "component",
            "component.adafruit.ssd1306-128x64-stemma",
            subject_id="component:adafruit:ssd1306-128x64-stemma",
            family_id="component-family:ssd1306-oled-module",
            name="Adafruit 0.96 inch 128x64 OLED STEMMA QT",
            variant="STEMMA QT I2C-default with auto-reset",
            aliases=("ADAFRUIT_SSD1306_STEMMA_128X64", "OLED_SSD1306_STEMMA_128X64"),
            conflict_group="conflict.component.ssd1306-breakout",
            claims=[
                claim("support-status", "supported", [adafruit["evidenceId"]]),
                claim("manufacturer", "Adafruit Industries", [adafruit["evidenceId"]]),
                claim("part-number", "0.96-inch 128x64 STEMMA QT product variant", [adafruit["evidenceId"]], status="conditional", conditions=("Confirm product revision visually before wiring.",)),
                claim("terminals", ["VIN", "GND", "DATA/SDA", "CLK/SCL"], [adafruit["evidenceId"]]),
                claim("electrical-ratings", {"logic": "module-level shifted per maintained breakout guide", "power": "3V or guide-approved VIN connection"}, [adafruit["evidenceId"]], status="conditional", conditions=("Applies only to the STEMMA QT revision.",)),
                claim("interfaces", ["I2C-default", "SPI-after-hardware-configuration"], [adafruit["evidenceId"]]),
                claim("reset", "on-board auto-reset; external RST not required", [adafruit["evidenceId"]]),
            ],
            evidence_items=[adafruit],
            tags=("component", "display", "i2c", "ssd1306"),
        ),
        record(
            "component",
            "component.adafruit.ssd1306-128x64-v1",
            subject_id="component:adafruit:ssd1306-128x64-v1",
            family_id="component-family:ssd1306-oled-module",
            name="Adafruit 128x64 OLED Version 1.0",
            variant="older 3.3V breakout without built-in level shifting",
            aliases=("ADAFRUIT_SSD1306_128X64_V1", "OLED_SSD1306_V1"),
            conflict_group="conflict.component.ssd1306-breakout",
            claims=[
                claim("support-status", "reference-only", [adafruit["evidenceId"]]),
                claim("manufacturer", "Adafruit Industries", [adafruit["evidenceId"]]),
                claim("part-number", "older 128x64 Version 1.0 breakout", [adafruit["evidenceId"]]),
                claim("terminals", ["VDD", "VBAT", "GND", "CLK", "DATA", "RES", "CS", "D/C"], [adafruit["evidenceId"]]),
                claim("electrical-ratings", {"logicVoltageV": 3.3, "fiveVoltLogicTolerant": False}, [adafruit["evidenceId"]]),
                claim("interfaces", ["SPI", "I2C-after-jumper-configuration"], [adafruit["evidenceId"]]),
                claim("reset", "external reset connection required by this revision", [adafruit["evidenceId"]]),
            ],
            evidence_items=[adafruit],
            support_status="reference-only",
            tags=("component", "display", "ssd1306", "legacy-variant"),
        ),
        record(
            "component",
            "component.generic.led",
            subject_id="component:generic:led",
            family_id="component-family:led",
            name="Generic LED",
            variant="manufacturer part not selected",
            aliases=("LED", "LIGHT_EMITTING_DIODE"),
            claims=[
                claim("support-status", "variant-required", [project["evidenceId"]]),
                claim("manufacturer", None, [project["evidenceId"]], status="unknown"),
                claim("part-number", None, [project["evidenceId"]], status="unknown"),
                claim("terminals", ["anode", "cathode"], [project["evidenceId"]]),
                claim("electrical-ratings", None, [project["evidenceId"]], status="unknown", conditions=("Forward voltage and current require an exact part datasheet.",)),
            ],
            evidence_items=[project],
            support_status="variant-required",
            tags=("component", "led", "variant-required"),
        ),
        record(
            "component",
            "component.generic.relay-module",
            subject_id="component:generic:relay-module",
            family_id="component-family:relay-module",
            name="Generic relay module",
            variant="coil, driver, isolation, and trigger polarity not selected",
            aliases=("RELAY", "RELAY_MODULE"),
            claims=[
                claim("support-status", "variant-required", [project["evidenceId"]]),
                claim("manufacturer", None, [project["evidenceId"]], status="unknown"),
                claim("part-number", None, [project["evidenceId"]], status="unknown"),
                claim("terminals", None, [project["evidenceId"]], status="unknown"),
                claim("electrical-ratings", None, [project["evidenceId"]], status="unknown", conditions=("Coil voltage, trigger voltage/current, contact ratings, and isolation require an exact module datasheet.",)),
            ],
            evidence_items=[project],
            support_status="variant-required",
            tags=("component", "relay", "variant-required", "safety"),
        ),
    ]


def _wiring_records() -> list[dict[str, Any]]:
    project = project_evidence(
        "evidence.vf.wiring-normalization",
        "VoltForge revision-specific wiring normalization",
        "electronics_corpus/builder.py::_wiring_records",
    )
    adafruit = evidence(
        "evidence.adafruit.oled-wiring",
        kind="maintainer-documentation",
        publisher="Adafruit Industries",
        title="128x64 OLED I2C wiring",
        revision="last-edited-2025-12-18",
        locator="STEMMA QT I2C wiring and auto-reset note",
        uri="https://learn.adafruit.com/monochrome-oled-breakouts/wiring-128x64-oleds",
    )
    board_rows = [
        (UNO_R3_ID, "arduino.uno-r3", "Arduino UNO R3 to Adafruit SSD1306 STEMMA QT", ["ARDUINO_UNO+ADAFRUIT_SSD1306_STEMMA_128X64"], {"VIN": "3V3", "GND": "GND", "DATA/SDA": "A4", "CLK/SCL": "A5"}),
        (UNO_R4_ID, "arduino.uno-r4-wifi", "Arduino UNO R4 WiFi to Adafruit SSD1306 STEMMA QT", ["ARDUINO_UNO_R4+ADAFRUIT_SSD1306_STEMMA_128X64"], {"VIN": "QWIIC_3V3", "GND": "QWIIC_GND", "DATA/SDA": "QWIIC_SDA", "CLK/SCL": "QWIIC_SCL"}),
        (MEGA_ID, "arduino.mega-2560-r3", "Arduino Mega 2560 Rev3 to Adafruit SSD1306 STEMMA QT", ["ARDUINO_MEGA+ADAFRUIT_SSD1306_STEMMA_128X64"], {"VIN": "3V3", "GND": "GND", "DATA/SDA": "D20", "CLK/SCL": "D21"}),
        (NANO_ID, "arduino.nano-classic", "Arduino Nano classic to Adafruit SSD1306 STEMMA QT", ["ARDUINO_NANO+ADAFRUIT_SSD1306_STEMMA_128X64"], {"VIN": "3V3", "GND": "GND", "DATA/SDA": "A4", "CLK/SCL": "A5"}),
        (ESP32_ID, "espressif.esp32-devkitc-v4", "ESP32-DevKitC V4 to Adafruit SSD1306 STEMMA QT", ["ESP32_DEVKITC_V4_WROOM32E_N4+ADAFRUIT_SSD1306_STEMMA_128X64"], {"VIN": "3V3", "GND": "GND", "DATA/SDA": "GPIO21", "CLK/SCL": "GPIO22"}),
        (ESP32_S3_ID, "espressif.esp32-s3-devkitc1", "ESP32-S3-DevKitC-1 to Adafruit SSD1306 STEMMA QT", ["ESP32_S3+ADAFRUIT_SSD1306_STEMMA_128X64"], {"VIN": "3V3", "GND": "GND", "DATA/SDA": "GPIO8", "CLK/SCL": "GPIO9"}),
        (PICO_ID, "raspberry-pi.pico", "Raspberry Pi Pico to Adafruit SSD1306 STEMMA QT", ["RASPBERRY_PI_PICO+ADAFRUIT_SSD1306_STEMMA_128X64"], {"VIN": "3V3_OUT", "GND": "GND", "DATA/SDA": "GP4", "CLK/SCL": "GP5"}),
        (PICO2_ID, "raspberry-pi.pico2", "Raspberry Pi Pico 2 to Adafruit SSD1306 STEMMA QT", ["RASPBERRY_PI_PICO_2+ADAFRUIT_SSD1306_STEMMA_128X64"], {"VIN": "3V3_OUT", "GND": "GND", "DATA/SDA": "GP4", "CLK/SCL": "GP5"}),
    ]
    records = []
    refs = [project["evidenceId"], adafruit["evidenceId"]]
    for board_id, slug, name, aliases, mapping in board_rows:
        records.append(
            record(
                "wiring-recipe",
                f"wiring.{slug}.ssd1306-stemma-i2c",
                subject_id=f"wiring:{slug}:ssd1306-stemma-i2c",
                family_id="wiring-family:ssd1306-stemma-i2c",
                name=name,
                variant="I2C at 3.3V using exact board and breakout revisions",
                aliases=aliases,
                claims=[
                    claim("board-record-id", board_id, refs),
                    claim("component-record-id", OLED_NEW_ID, refs),
                    claim("connections", [{"componentTerminal": terminal, "boardPin": pin} for terminal, pin in mapping.items()], refs),
                    claim("preconditions", ["Exact board and OLED variants match the record IDs.", "Power is off while wiring.", "I2C address is discovered or confirmed before firmware initialization."], refs),
                    claim("post-checks", ["Common ground is continuous.", "No signal exceeds the selected board or module logic rail.", "I2C device acknowledges at the configured address."], refs),
                ],
                evidence_items=[project, adafruit],
                relations=(relation("requires", board_id), relation("requires", OLED_NEW_ID)),
                tags=("wiring", "i2c", "ssd1306", "3v3"),
            )
        )
    return records


def _firmware_records() -> list[dict[str, Any]]:
    arduino = evidence(
        "evidence.arduino.language-reference",
        kind="maintainer-documentation",
        publisher="Arduino",
        title="Arduino Language Reference",
        revision="current-at-2026-08-28",
        locator="Digital I/O and Wire communication APIs",
        uri="https://docs.arduino.cc/language-reference/",
    )
    adafruit = evidence(
        "evidence.adafruit.ssd1306-library",
        kind="maintainer-documentation",
        publisher="Adafruit Industries",
        title="Adafruit SSD1306 maintained examples",
        revision="current-at-2026-08-28",
        locator="SSD1306_128x64_i2c example and constructor parameters",
        uri="https://learn.adafruit.com/monochrome-oled-breakouts/arduino-library-and-examples",
    )
    pico = evidence(
        "evidence.raspberry-pi.pico-sdk",
        kind="maintainer-documentation",
        publisher="Raspberry Pi Ltd",
        title="Raspberry Pi Pico-series C/C++ SDK",
        revision="current-at-2026-08-28",
        locator="hardware_gpio and hardware_i2c APIs",
        uri="https://datasheets.raspberrypi.com/pico/raspberry-pi-pico-c-sdk.pdf",
    )
    arduino_boards = [UNO_R3_ID, UNO_R4_ID, MEGA_ID, NANO_ID, ESP32_ID, ESP32_S3_ID]
    pico_boards = [PICO_ID, PICO2_ID]
    rows = [
        ("firmware.arduino.gpio", "firmware-api:arduino:gpio", "Arduino digital I/O API", "Arduino core selected by exact board package", ["ARDUINO_GPIO_API"], arduino, "Arduino", "board-package-version-must-be-recorded", "Arduino.h", ["pinMode", "digitalRead", "digitalWrite"], arduino_boards),
        ("firmware.arduino.wire", "firmware-api:arduino:wire", "Arduino Wire I2C API", "Arduino core selected by exact board package", ["ARDUINO_WIRE_API", "WIRE_H"], arduino, "Arduino Wire", "board-package-version-must-be-recorded", "Wire.h", ["Wire.begin", "Wire.beginTransmission", "Wire.write", "Wire.endTransmission", "Wire.requestFrom", "Wire.read"], arduino_boards),
        ("firmware.adafruit.ssd1306", "firmware-api:adafruit:ssd1306", "Adafruit SSD1306 Arduino library", "library release must be pinned in the project", ["ADAFRUIT_SSD1306_API"], adafruit, "Adafruit_SSD1306", "exact-library-version-required", "Adafruit_SSD1306.h", ["Adafruit_SSD1306", "begin", "clearDisplay", "display", "setCursor", "print"], arduino_boards),
        ("firmware.pico-sdk.i2c", "firmware-api:raspberry-pi:pico-sdk-i2c", "Pico SDK hardware I2C API", "Pico SDK release must be pinned", ["PICO_SDK_I2C_API"], pico, "Pico SDK hardware_i2c", "exact-sdk-version-required", "hardware/i2c.h", ["i2c_init", "gpio_set_function", "i2c_write_blocking", "i2c_read_blocking"], pico_boards),
    ]
    records = []
    for slug, subject_id, name, variant, aliases, ev, framework, policy, include, symbols, board_ids in rows:
        refs = [ev["evidenceId"]]
        records.append(
            record(
                "firmware-api",
                slug,
                subject_id=subject_id,
                family_id="firmware-family:" + subject_id.split(":", 1)[1].split(":")[0],
                name=name,
                variant=variant,
                aliases=aliases,
                claims=[
                    claim("framework", framework, refs),
                    claim("version-policy", policy, refs, status="conditional", conditions=("Do not infer API compatibility across unrecorded package versions.",)),
                    claim("include", include, refs),
                    claim("api-symbols", symbols, refs),
                    claim("supported-board-record-ids", board_ids, refs),
                ],
                evidence_items=[ev],
                relations=tuple(relation("applies-to", board_id) for board_id in board_ids),
                tags=("firmware", "api"),
            )
        )
    return records


def _diagnostic_records() -> list[dict[str, Any]]:
    ev = evidence(
        "evidence.vf.compiler-diagnostics",
        kind="toolchain-documentation",
        publisher="VoltForge",
        title="VoltForge normalized compiler and uploader diagnostic patterns",
        revision=BUILDER_VERSION,
        locator="electronics_corpus/builder.py::_diagnostic_records",
    )
    rows = [
        ("compiler.missing-header", "Missing header", "gcc-compatible-cpp", r"fatal error: .+\.h: No such file or directory", ["Library is not installed.", "Header spelling or case is wrong.", "Selected board package does not provide the header."], ["Read the exact missing header from the diagnostic.", "Verify the project's pinned library and board package.", "Do not substitute an unrelated library automatically."]),
        ("compiler.undeclared-identifier", "Undeclared identifier", "gcc-compatible-cpp", r"['`].+['`] was not declared in this scope", ["Name is misspelled.", "Declaration is out of scope.", "Required header or generated symbol is missing."], ["Resolve the exact source location.", "Compare declaration spelling and scope.", "Add a declaration only after its intended type and ownership are known."]),
        ("compiler.undefined-reference", "Undefined reference at link", "gcc-compatible-linker", r"undefined reference to .+", ["Declaration exists but no linked definition is available.", "Signature or namespace differs.", "Required object or library was not linked."], ["Compare declaration and definition signatures.", "Inspect the linker inputs and pinned library version.", "Do not hide the failure with a stub implementation."]),
        ("compiler.avrdude-sync", "AVR upload synchronization failure", "avrdude", r"stk500_(recv|getsync).*(not in sync|programmer is not responding)", ["Wrong board, processor, port, or programmer selection.", "Serial port is busy or unavailable.", "Bootloader, cable, reset timing, or target power may be faulty."], ["Confirm the exact board and processor variant.", "Confirm the selected port and close competing serial clients.", "Inspect cable, power, and bootloader state before retrying."]),
    ]
    refs = [ev["evidenceId"]]
    return [
        record(
            "compiler-diagnostic",
            slug,
            subject_id="diagnostic:" + slug,
            family_id="diagnostic-family:compiler",
            name=name,
            variant=toolchain,
            aliases=(name.upper().replace(" ", "_"),),
            claims=[
                claim("toolchain", toolchain, refs),
                claim("diagnostic-pattern", pattern, refs),
                claim("causes", causes, refs, status="conditional", conditions=("Causes are candidates until project/tool evidence selects one.",)),
                claim("remediation", remediation, refs),
            ],
            evidence_items=[ev],
            tags=("compiler", "diagnostic", toolchain),
        )
        for slug, name, toolchain, pattern, causes, remediation in rows
    ]


def _simulation_records() -> list[dict[str, Any]]:
    ev = project_evidence(
        "evidence.vf.simulation-models",
        "VoltForge deterministic simulation model boundaries",
        "electronics_corpus/builder.py::_simulation_records",
    )
    refs = [ev["evidenceId"]]
    rows = [
        ("simulation.resistor-ideal", "Ideal resistor model", "linear time-invariant resistance", ["dc-operating-point", "transient", "ac-small-signal"], ["Resistance is constant.", "Temperature coefficient and parasitics are omitted."], ["Does not predict pulse overload, tolerance distribution, package heating, or high-frequency parasitics."]),
        ("simulation.capacitor-ideal", "Ideal capacitor model", "constant capacitance", ["transient", "ac-small-signal"], ["Capacitance is constant.", "Initial voltage is explicit or zero."], ["Does not include ESR, ESL, leakage, dielectric absorption, voltage coefficient, or breakdown."]),
        ("simulation.led-piecewise", "Piecewise LED model", "configured forward-drop plus current path", ["dc-operating-point", "transient"], ["Forward drop is a configured approximation, not a universal LED property."], ["Does not predict color, luminous intensity, thermal runaway, reverse breakdown, or a manufacturer-specific I-V curve."]),
        ("simulation.i2c-digital", "Digital I2C bus model", "open-drain logical bus with configured pull-ups", ["digital-event"], ["Participants obey configured open-drain behavior.", "Pull-up presence is represented as a logical constraint."], ["Does not predict rise time, capacitance, ringing, level-shifter analog behavior, or EMC performance."]),
    ]
    return [
        record(
            "simulation-behavior",
            slug,
            subject_id="simulation:" + slug,
            family_id="simulation-family:deterministic",
            name=name,
            variant=variant,
            aliases=(name.upper().replace(" ", "_"),),
            claims=[
                claim("model", variant, refs),
                claim("supported-analyses", analyses, refs),
                claim("assumptions", assumptions, refs),
                claim("limitations", limitations, refs),
            ],
            evidence_items=[ev],
            tags=("simulation", "model-boundary"),
        )
        for slug, name, variant, analyses, assumptions, limitations in rows
    ]


def _safety_records() -> list[dict[str, Any]]:
    ev = project_evidence(
        "evidence.vf.safety-policy",
        "VoltForge deterministic electronics safety constraints",
        "electronics_corpus/builder.py::_safety_records",
    )
    refs = [ev["evidenceId"]]
    rows = [
        ("safety.logic-voltage", "Logic voltage compatibility", "critical", "A signal or supply voltage is connected to a board/component pin.", "Compare against the exact variant's evidenced absolute and recommended ratings; add translation or change the design when incompatible.", "If either endpoint rating is absent, report unknown and do not assert compatibility."),
        ("safety.gpio-current", "GPIO current limit", "critical", "A GPIO sources or sinks a load.", "Verify per-pin, port-group, and total-device current against the exact board/MCU revision and add a driver when required.", "If exact current limits are absent, report unknown and do not use a family default."),
        ("safety.led-current-limit", "LED current limiting", "high", "An LED is driven from a voltage source or GPIO.", "Select a resistor/driver from evidenced supply voltage, exact LED forward-voltage range, target current, and power margin.", "If the LED part or ratings are missing, request them; do not invent a forward voltage or current."),
        ("safety.inductive-flyback", "Inductive load flyback protection", "critical", "A relay coil, solenoid, or motor is switched by a semiconductor.", "Provide a correctly oriented and rated flyback/suppression path unless the exact driver/load already documents equivalent protection.", "If integrated protection is not evidenced, treat protection as absent."),
        ("safety.common-ground", "Common signal reference", "high", "Non-isolated devices exchange a signal.", "Provide a compatible common reference path and verify isolation intent before connecting grounds.", "If isolation topology is unknown, stop and request the schematic/reference evidence."),
        ("safety.i2c-pullups", "I2C pull-up ownership", "high", "An I2C bus is assembled or modified.", "Inventory all pull-ups, voltage domains, bus capacitance, and target speed; ensure the combined pull-up is safe for every participant.", "If module pull-ups or bus capacitance are unknown, do not prescribe a universal resistor value."),
        ("safety.high-energy-boundary", "Mains and high-energy boundary", "critical", "A request involves mains, high voltage, batteries with hazardous fault energy, or safety-critical control.", "Refuse unverified construction instructions and require qualified review, applicable standards, isolation, protection, and certified components.", "Missing jurisdiction, standard, ratings, or professional review keeps the design unsupported."),
    ]
    return [
        record(
            "safety-constraint",
            slug,
            subject_id="constraint:" + slug,
            family_id="constraint-family:electronics-safety",
            name=name,
            variant="VoltForge deterministic rule v1",
            aliases=(name.upper().replace(" ", "_"),),
            claims=[
                claim("severity", severity, refs),
                claim("condition", condition, refs),
                claim("required-action", action, refs),
                claim("unknown-policy", unknown_policy, refs),
            ],
            evidence_items=[ev],
            tags=("safety", severity),
        )
        for slug, name, severity, condition, action, unknown_policy in rows
    ]


def build_records() -> list[dict[str, Any]]:
    records = [
        *_board_records(),
        *_pin_map_records(),
        *_component_records(),
        *_wiring_records(),
        *_firmware_records(),
        *_diagnostic_records(),
        *_simulation_records(),
        *_safety_records(),
    ]
    validated = [validate_knowledge_record(item) for item in records]
    ids = [item["recordId"] for item in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("Curated corpus record IDs must be unique.")
    known = set(ids)
    for item in validated:
        missing = sorted(
            relation_item["targetRecordId"]
            for relation_item in item.get("relations", [])
            if relation_item["targetRecordId"] not in known
        )
        if missing:
            raise ValueError(f"{item['recordId']} has unresolved relations: {missing}")
    return sorted(validated, key=lambda item: (PACK_ORDER.index(item["recordType"]), item["recordId"]))


def expected_outputs() -> dict[Path, bytes]:
    records = build_records()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[item["recordType"]].append(item)
    schema_bytes = _canonical_json(build_knowledge_json_schema(), pretty=True)
    builder_bytes = Path(__file__).read_bytes()
    outputs: dict[Path, bytes] = {KNOWLEDGE_SCHEMA_PATH: schema_bytes}
    packs = []
    for record_type in PACK_ORDER:
        payload = b"".join(_canonical_json(item) for item in grouped[record_type])
        relative = PACK_FILENAMES[record_type]
        path = CORPUS_ROOT / relative
        outputs[path] = payload
        packs.append(
            {
                "packId": f"vf-pack-{record_type}-v1",
                "recordType": record_type,
                "path": relative,
                "recordCount": len(grouped[record_type]),
                "sizeBytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    counts = {record_type: len(grouped[record_type]) for record_type in PACK_ORDER}
    conflicts: dict[str, list[str]] = defaultdict(list)
    for item in records:
        if item.get("conflictGroup"):
            conflicts[item["conflictGroup"]].append(item["recordId"])
    catalog = {
        "schemaVersion": 1,
        "corpusId": "voltforge-curated-electronics-corpus",
        "version": CORPUS_CONTRACT_VERSION,
        "effectiveFrom": EFFECTIVE_FROM,
        "source": {"sourceId": SOURCE_ID, "sourceRevision": CORPUS_CONTRACT_VERSION},
        "recordSchema": {
            "path": "knowledge-record.schema.json",
            "sha256": _sha256_bytes(schema_bytes),
        },
        "builder": {
            "id": BUILDER_ID,
            "version": BUILDER_VERSION,
            "path": "builder.py",
            "sha256": _sha256_bytes(builder_bytes),
        },
        "recordCount": len(records),
        "recordTypeCounts": counts,
        "packs": packs,
        "supportPolicy": {
            "lookupModes": ["exact", "variant-qualified"],
            "rankedRetrievalIncluded": False,
            "unsupportedResult": "unknown",
            "ambiguousResult": "variant-required",
        },
    }
    outputs[CORPUS_ROOT / "catalog.v1.json"] = _canonical_json(catalog, pretty=True)
    report = {
        "schemaVersion": 1,
        "reportId": "vfai-008-current-curated-corpus",
        "corpusVersion": CORPUS_CONTRACT_VERSION,
        "asOfDate": EFFECTIVE_FROM,
        "recordCount": len(records),
        "recordTypeCounts": counts,
        "supportedRecordCount": sum(item["supportStatus"] == "supported" for item in records),
        "variantRequiredRecordCount": sum(item["supportStatus"] == "variant-required" for item in records),
        "referenceOnlyRecordCount": sum(item["supportStatus"] == "reference-only" for item in records),
        "evidenceReferenceCount": sum(len(item["provenance"]["evidence"]) for item in records),
        "publisherCounts": dict(
            sorted(
                Counter(
                    evidence_item["publisher"]
                    for item in records
                    for evidence_item in item["provenance"]["evidence"]
                ).items()
            )
        ),
        "conflictGroups": [
            {"conflictGroup": key, "recordIds": sorted(value)}
            for key, value in sorted(conflicts.items())
        ],
        "unsupportedPolicy": "No match or missing evidence returns unknown; family conflicts require an exact variant.",
        "allRecordsSchemaValid": True,
        "allClaimsEvidenceBound": True,
        "allRelationsResolved": True,
        "packChecksumsBoundByCatalog": True,
    }
    outputs[CORPUS_ROOT / "reports/current-corpus-report.json"] = _canonical_json(report, pretty=True)
    return outputs


def write_outputs() -> dict[str, Any]:
    outputs = expected_outputs()
    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return corpus_summary(outputs)


def check_outputs() -> dict[str, Any]:
    outputs = expected_outputs()
    stale = [str(path.relative_to(CORPUS_ROOT)).replace("\\", "/") for path, payload in outputs.items() if not path.is_file() or path.read_bytes() != payload]
    if stale:
        raise ValueError(f"Curated corpus outputs are stale or missing: {stale}")
    return corpus_summary(outputs)


def corpus_summary(outputs: dict[Path, bytes]) -> dict[str, Any]:
    catalog = json.loads(outputs[CORPUS_ROOT / "catalog.v1.json"])
    return {
        "ok": True,
        "corpusId": catalog["corpusId"],
        "version": catalog["version"],
        "recordCount": catalog["recordCount"],
        "recordTypeCounts": catalog["recordTypeCounts"],
        "packCount": len(catalog["packs"]),
    }
