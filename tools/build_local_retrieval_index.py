"""Write or verify the checked VFAI-022 schemas and lexical index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from local_retrieval.builder import build_index
from local_retrieval.schema import (
    RETRIEVAL_INDEX_PATH,
    RETRIEVAL_INDEX_SCHEMA_PATH,
    RETRIEVAL_RESPONSE_SCHEMA_PATH,
    build_index_json_schema,
    build_response_json_schema,
    load_retrieval_policy,
)


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _outputs() -> dict[Path, bytes]:
    index = build_index().model_dump(mode="json")
    return {
        RETRIEVAL_INDEX_SCHEMA_PATH: _pretty(build_index_json_schema()),
        RETRIEVAL_RESPONSE_SCHEMA_PATH: _pretty(build_response_json_schema()),
        RETRIEVAL_INDEX_PATH: _pretty(index),
    }


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write() -> dict[str, Any]:
    outputs = _outputs()
    for path, content in outputs.items():
        _write(path, content)
    index = json.loads(outputs[RETRIEVAL_INDEX_PATH])
    return _summary(index, "write")


def verify() -> dict[str, Any]:
    outputs = _outputs()
    for path, expected in outputs.items():
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"VFAI-022 checked output is missing: {path}") from error
        if actual != expected:
            raise RuntimeError(f"VFAI-022 checked output is stale: {path}")
    index = json.loads(outputs[RETRIEVAL_INDEX_PATH])
    return _summary(index, "verify")


def _summary(index: dict[str, Any], command: str) -> dict[str, Any]:
    policy = load_retrieval_policy()
    return {
        "ok": True,
        "command": command,
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "indexId": index["indexId"],
        "indexVersion": index["indexVersion"],
        "indexSha256": index["indexSha256"],
        "recordCount": index["recordCount"],
        "chunkCount": index["chunkCount"],
        "termCount": len(index["documentFrequencies"]),
        "embeddingsPresent": index["embeddingsPresent"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    result = write() if arguments.command == "write" else verify()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
