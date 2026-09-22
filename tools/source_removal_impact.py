"""Read-only source deletion and retraining impact analyzer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from data_governance.governance import DataGovernanceError, analyze_source_removal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List shards and model artifacts affected by removing one registered source."
    )
    parser.add_argument("source_id", help="Exact sourceId from source-registry.v1.json")
    args = parser.parse_args(argv)
    try:
        report = analyze_source_removal(args.source_id)
    except DataGovernanceError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": error.message}))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
