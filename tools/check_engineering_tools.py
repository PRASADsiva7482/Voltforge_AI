"""Write or verify the checked-in VFAI-021 engineering report schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from engineering_tools.schema import (
    ENGINEERING_SCHEMA_PATH,
    build_engineering_json_schema,
    checked_engineering_schema,
    load_engineering_policy,
)


def write_schema(path: Path = ENGINEERING_SCHEMA_PATH) -> None:
    path.write_text(
        json.dumps(build_engineering_json_schema(), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def verify() -> dict[str, object]:
    policy = load_engineering_policy()
    schema = checked_engineering_schema()
    return {
        "ok": True,
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "toolCount": len(policy["tools"]),
        "schemaId": schema["$id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write-schema", "verify"))
    arguments = parser.parse_args()
    if arguments.command == "write-schema":
        write_schema()
        checked_engineering_schema.cache_clear()
    print(json.dumps(verify(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
