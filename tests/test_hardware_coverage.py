from __future__ import annotations

from hardware_coverage import get_hardware_coverage


def test_coverage_is_one_to_one_with_curated_ui_inventory() -> None:
    report = get_hardware_coverage()
    assert report["entryCount"] == 49
    assert report["summary"] == {"verified": 8, "variantRequired": 22, "unsupported": 19}
    assert len({entry["boardType"] for entry in report["entries"]}) == 49


def test_generic_aliases_are_gated_and_exact_variants_are_verified() -> None:
    entries = {entry["boardType"]: entry for entry in get_hardware_coverage()["entries"]}
    assert entries["ARDUINO_NANO"]["status"] == "variant-required"
    assert entries["ESP32"]["status"] == "variant-required"
    assert entries["ESP32_S3"]["status"] == "variant-required"
    assert entries["ARDUINO_NANO_EVERY"]["status"] == "verified"
    assert entries["ARDUINO_LEONARDO"]["status"] == "verified"
    assert entries["ARDUINO_MICRO"]["status"] == "verified"
