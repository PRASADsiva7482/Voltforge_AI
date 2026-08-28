"""Generate or verify the checked-in VoltForge task-record JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from task_schema.schema import TASK_SCHEMA_PATH, build_task_json_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Intentionally refresh the schema file.")
    args = parser.parse_args(argv)
    expected = build_task_json_schema()
    if args.write:
        TASK_SCHEMA_PATH.write_text(
            json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    try:
        actual = json.loads(TASK_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        print(json.dumps({"ok": False, "code": "TASK_SCHEMA_UNAVAILABLE"}))
        return 1
    if actual != expected:
        print(json.dumps({"ok": False, "code": "TASK_SCHEMA_STALE"}))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "schemaVersion": actual["properties"]["schemaVersion"]["const"],
                "contractVersion": actual["properties"]["contractVersion"]["const"],
                "path": str(TASK_SCHEMA_PATH.relative_to(AI_ROOT)).replace("\\", "/"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
