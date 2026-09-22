"""Fail when frozen evaluation content appears in training or retrieval data."""

from pathlib import Path
import json
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from evaluation.leakage import scan_corpora


def main() -> int:
    report = scan_corpora()
    print(
        json.dumps(
            {
                "status": report["status"],
                "recordsScanned": report["recordsScanned"],
                "filesScanned": report["filesScanned"],
                "exactAndSemanticCollisions": report["exactAndSemanticCollisions"],
                "collisions": report["collisions"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
