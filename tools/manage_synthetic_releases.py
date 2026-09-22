"""Publish or verify immutable VFAI-FU-015 synthetic releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from synthetic_data.immutable_release import (  # noqa: E402
    ImmutableReleaseError,
    publish_current_release,
    verify_all_releases,
    verify_current_release,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("publish-current")
    commands.add_parser("verify-current")
    commands.add_parser("verify-all")
    args = parser.parse_args(argv)
    try:
        if args.command == "publish-current":
            result: object = publish_current_release()
        elif args.command == "verify-current":
            result = verify_current_release()
        else:
            result = verify_all_releases()
    except ImmutableReleaseError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"ok": True, "command": args.command, "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
