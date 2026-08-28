"""Audit retained VoltForge corpora and refresh quarantine manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from data_governance.governance import DataGovernanceError, audit_current_corpora


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify complete fail-closed governance coverage for retained datasets."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing manifests without changing files.",
    )
    args = parser.parse_args(argv)
    try:
        report = audit_current_corpora(write=not args.check)
    except DataGovernanceError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": error.message}))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "mode": "check" if args.check else "write",
                "summary": report["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
