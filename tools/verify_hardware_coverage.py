"""Verify the AI hardware coverage report against the UI board catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_coverage import get_hardware_coverage


BOARD_PATTERN = re.compile(
    r"board\('(?P<type>[A-Z0-9_]+)',\s*'(?P<name>[^']+)',\s*'(?P<family>[^']+)'"
)


def _read_ui_catalog(path: Path) -> list[tuple[str, str, str]]:
    text = path.read_text(encoding="utf-8")
    return [(match.group("type"), match.group("name"), match.group("family")) for match in BOARD_PATTERN.finditer(text)]


def verify(ui_catalog: Path | None = None) -> dict[str, Any]:
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
    return {
        "ok": True,
        "reportId": report["reportId"],
        "entryCount": report["entryCount"],
        "summary": report["summary"],
        "reportSha256": report["reportSha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-catalog", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.ui_catalog), sort_keys=True))


if __name__ == "__main__":
    main()
