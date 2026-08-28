"""Create or verify the immutable VFAI release-suite manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.leakage import MANIFEST_PATH, build_manifest_data, verify_frozen_suite


def write_manifest(output: Path = MANIFEST_PATH) -> dict[str, object]:
    manifest = build_manifest_data()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create or intentionally replace the freeze manifest.")
    arguments = parser.parse_args(argv)
    manifest = write_manifest() if arguments.write else verify_frozen_suite()
    print(
        json.dumps(
            {
                "status": "locked",
                "suiteId": manifest["suiteId"],
                "suiteVersion": manifest["suiteVersion"],
                "suiteSha256": manifest["suiteSha256"],
                "caseCount": manifest["caseCount"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
