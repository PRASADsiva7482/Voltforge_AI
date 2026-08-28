"""Build or verify the VFAI-009 checked-in synthetic-data release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from synthetic_data.pipeline import check_release, write_pipeline_lock, write_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-lock", action="store_true", help="Refresh only the reviewed pipeline lock")
    mode.add_argument("--write", action="store_true", help="Compile firmware and write all release outputs")
    mode.add_argument("--check", action="store_true", help="No-write deterministic release verification")
    mode.add_argument(
        "--recompile",
        action="store_true",
        help="No-write verification plus fresh compilation using the local pinned toolchain",
    )
    args = parser.parse_args()
    if args.write_lock:
        print(write_pipeline_lock())
        return 0
    report = write_release() if args.write else check_release(recompile=args.recompile)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
