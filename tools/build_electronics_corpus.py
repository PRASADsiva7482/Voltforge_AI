"""Generate or verify the immutable VFAI-008 electronics knowledge packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from electronics_corpus.builder import check_outputs, write_outputs
from electronics_corpus.schema import KNOWLEDGE_SCHEMA_PATH, build_knowledge_json_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Intentionally refresh generated packs.")
    args = parser.parse_args(argv)
    if args.write:
        KNOWLEDGE_SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_SCHEMA_PATH.write_text(
            json.dumps(build_knowledge_json_schema(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    try:
        summary = write_outputs() if args.write else check_outputs()
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
