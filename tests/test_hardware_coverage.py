from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hardware_coverage import get_hardware_coverage
from tools.verify_hardware_coverage import verify


UI_ROOT = Path(__file__).resolve().parents[2] / "Voltforge_UI"
UI_BOARD_CATALOG = UI_ROOT / "src/features/canvas/boardCatalog.ts"
UI_COMPONENT_CATALOG = UI_ROOT / "src/features/canvas/componentCatalog.ts"
UI_BOARD_GEOMETRY = UI_ROOT / "src/features/canvas/boardGeometryFixtures.v1.json"


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


def test_exact_board_pinout_fixtures_match_electrical_coverage() -> None:
    result = verify(UI_BOARD_CATALOG, UI_COMPONENT_CATALOG, UI_BOARD_GEOMETRY)

    assert result["boardGeometry"] == {
        "fixtureCount": 8,
        "manufacturerVerifiedPinoutCount": 8,
        "mechanicallyVerifiedArtworkCount": 0,
        "documentedArtworkLimitationCount": 8,
        "pinCount": 328,
        "sourceHosts": ["datasheets.raspberrypi.com", "docs.arduino.cc"],
    }


def test_geometry_verifier_rejects_mechanical_overclaim(tmp_path: Path) -> None:
    document = json.loads(UI_BOARD_GEOMETRY.read_text(encoding="utf-8"))
    tampered = deepcopy(document)
    tampered["artworkMechanicallyVerified"] = True
    path = tmp_path / "tampered-board-geometry.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="must not claim mechanical verification"):
        verify(UI_BOARD_CATALOG, UI_COMPONENT_CATALOG, path)
