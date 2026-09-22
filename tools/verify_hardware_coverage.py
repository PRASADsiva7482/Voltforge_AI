"""Verify the AI hardware coverage report against the UI board catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_coverage import get_component_coverage, get_hardware_coverage


BOARD_PATTERN = re.compile(
    r"board\('(?P<type>[A-Z0-9_]+)',\s*'(?P<name>[^']+)',\s*'(?P<family>[^']+)'"
)
DIRECT_COMPONENT_PATTERN = re.compile(
    r"makeComponent\(\{\s*id:\s*'[^']+',\s*name:\s*'(?P<name>[^']+)',"
    r"\s*type:\s*'(?P<type>[A-Z0-9_]+)',\s*category:\s*'(?P<category>[A-Z]+)'",
    re.DOTALL,
)
TUPLE_COMPONENT_PATTERN = re.compile(
    r"^\s*\[\s*'(?P<type>[A-Z0-9_]+)'\s*,\s*'(?P<name>[^']+)'"
    r"\s*,\s*'(?P<category>[A-Z]+)'(?:\s*,|\s*\])",
    re.MULTILINE,
)

OFFICIAL_GEOMETRY_HOSTS = {"docs.arduino.cc", "datasheets.raspberrypi.com"}


def _read_ui_catalog(path: Path) -> list[tuple[str, str, str]]:
    text = path.read_text(encoding="utf-8")
    return [(match.group("type"), match.group("name"), match.group("family")) for match in BOARD_PATTERN.finditer(text)]


def _read_ui_component_catalog(path: Path) -> list[tuple[str, str, str]]:
    text = path.read_text(encoding="utf-8")
    values = [
        (match.group("type"), match.group("name"), match.group("category"))
        for pattern in (DIRECT_COMPONENT_PATTERN, TUPLE_COMPONENT_PATTERN)
        for match in pattern.finditer(text)
    ]
    if len(values) != len({item[0] for item in values}):
        raise ValueError("UI component catalog parser found duplicate component types.")
    return values


def _verify_board_geometry_fixtures(
    path: Path,
    hardware_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1:
        raise ValueError("Board geometry fixture schemaVersion must be 1.")
    if document.get("artworkMechanicallyVerified") is not False:
        raise ValueError("Generated board artwork must not claim mechanical verification.")

    fixtures = document.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("Board geometry fixtures must be a non-empty list.")
    fixture_types = [fixture.get("boardType") for fixture in fixtures]
    if len(fixture_types) != len(set(fixture_types)):
        raise ValueError("Board geometry fixtures contain duplicate board types.")

    verified_entries = {
        entry["boardType"]: entry
        for entry in hardware_entries
        if entry["status"] == "verified"
    }
    if set(fixture_types) != set(verified_entries):
        raise ValueError(
            "Manufacturer pinout fixtures must exactly cover AI-verified board types."
        )

    pin_count = 0
    source_hosts: set[str] = set()
    for fixture in fixtures:
        board_type = fixture["boardType"]
        if fixture.get("pinoutStatus") != "manufacturer-verified":
            raise ValueError(f"Pinout fixture is not verified: {board_type}")
        if fixture.get("artworkStatus") != "documented-limitation":
            raise ValueError(f"Artwork limitation is missing: {board_type}")
        if fixture.get("recordId") != verified_entries[board_type].get("recordId"):
            raise ValueError(f"Geometry fixture record binding drifted: {board_type}")
        if not fixture.get("geometryRevision") or not fixture.get("artworkLimitation"):
            raise ValueError(f"Geometry revision or limitation is missing: {board_type}")

        dimensions = fixture.get("canvasDimensions", {})
        if dimensions.get("w", 0) <= 0 or dimensions.get("h", 0) <= 0:
            raise ValueError(f"Canvas dimensions are invalid: {board_type}")

        source = fixture.get("source", {})
        host = urlparse(source.get("uri", "")).hostname
        if host not in OFFICIAL_GEOMETRY_HOSTS:
            raise ValueError(f"Geometry fixture source is not an approved primary host: {board_type}")
        if not source.get("documentRevision") or not source.get("title"):
            raise ValueError(f"Geometry fixture source metadata is incomplete: {board_type}")
        source_hosts.add(host)

        connectors = fixture.get("connectors")
        if not isinstance(connectors, list) or not connectors:
            raise ValueError(f"Geometry fixture connectors are missing: {board_type}")
        connector_ids = [connector.get("id") for connector in connectors]
        if len(connector_ids) != len(set(connector_ids)):
            raise ValueError(f"Geometry fixture connector IDs are duplicated: {board_type}")
        pins = [pin for connector in connectors for pin in connector.get("pins", [])]
        pin_ids = [pin.get("id") for pin in pins]
        if not pins or len(pin_ids) != len(set(pin_ids)):
            raise ValueError(f"Geometry fixture pin IDs are missing or duplicated: {board_type}")
        if any(not pin.get("name") or not pin.get("type") for pin in pins):
            raise ValueError(f"Geometry fixture pin metadata is incomplete: {board_type}")
        legacy_fallback = fixture.get("legacyFallbackPinIds", [])
        if set(legacy_fallback) & set(pin_ids):
            raise ValueError(f"Legacy fallback pins must remain outside the exact fixture: {board_type}")
        pin_count += len(pins)

    return {
        "fixtureCount": len(fixtures),
        "manufacturerVerifiedPinoutCount": len(fixtures),
        "mechanicallyVerifiedArtworkCount": 0,
        "documentedArtworkLimitationCount": len(fixtures),
        "pinCount": pin_count,
        "sourceHosts": sorted(source_hosts),
    }


def verify(
    ui_catalog: Path | None = None,
    ui_component_catalog: Path | None = None,
    ui_board_geometry_fixtures: Path | None = None,
) -> dict[str, Any]:
    report = get_hardware_coverage()
    entries = report["entries"]
    if report["entryCount"] != len(entries):
        raise ValueError("Coverage entry count does not match entries.")
    if len({entry["boardType"] for entry in entries}) != len(entries):
        raise ValueError("Coverage contains duplicate board types.")
    if ui_catalog is not None:
        expected = _read_ui_catalog(ui_catalog)
        actual = [(entry["boardType"], entry["displayName"], entry["family"]) for entry in entries]
        if actual != expected:
            raise ValueError("AI coverage and UI board catalog differ in order, type, name, or family.")
    component_report = get_component_coverage()
    component_entries = component_report["entries"]
    if component_report["entryCount"] != len(component_entries):
        raise ValueError("Component coverage entry count does not match entries.")
    if len({entry["componentType"] for entry in component_entries}) != len(
        component_entries
    ):
        raise ValueError("Component coverage contains duplicate component types.")
    if ui_component_catalog is not None:
        expected_components = set(_read_ui_component_catalog(ui_component_catalog))
        actual_components = {
            (entry["componentType"], entry["displayName"], entry["category"])
            for entry in component_entries
        }
        if actual_components != expected_components:
            missing = sorted(expected_components - actual_components)
            extra = sorted(actual_components - expected_components)
            raise ValueError(
                f"AI coverage and UI component catalog differ: missing={missing}, extra={extra}"
            )
    geometry_summary = None
    if ui_board_geometry_fixtures is not None:
        geometry_summary = _verify_board_geometry_fixtures(
            ui_board_geometry_fixtures,
            entries,
        )
    return {
        "ok": True,
        "reportId": report["reportId"],
        "entryCount": report["entryCount"],
        "summary": report["summary"],
        "reportSha256": report["reportSha256"],
        "componentReportId": component_report["reportId"],
        "componentEntryCount": component_report["entryCount"],
        "componentSummary": component_report["summary"],
        "componentReportSha256": component_report["reportSha256"],
        "boardGeometry": geometry_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-catalog", type=Path)
    parser.add_argument("--ui-component-catalog", type=Path)
    parser.add_argument("--ui-board-geometry-fixtures", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.ui_catalog,
                args.ui_component_catalog,
                args.ui_board_geometry_fixtures,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
