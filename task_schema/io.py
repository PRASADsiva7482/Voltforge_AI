"""Atomic JSONL I/O that validates every task record before shard entry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import TaskContractError, validate_task_record


def write_task_shard(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Validate the complete batch first, then atomically replace the shard."""

    validated = [validate_task_record(record) for record in records]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in validated:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return len(validated)


def read_task_shard(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TaskContractError(
                    "TASK_SHARD_JSON_INVALID", f"Invalid JSON at task shard line {line_number}."
                ) from error
            if not isinstance(value, dict):
                raise TaskContractError(
                    "TASK_SHARD_RECORD_INVALID", f"Task shard line {line_number} is not an object."
                )
            try:
                records.append(validate_task_record(value))
            except TaskContractError as error:
                raise TaskContractError(
                    error.code, f"Task shard line {line_number}: {error.message}"
                ) from error
    return records
