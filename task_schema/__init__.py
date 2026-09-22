"""Versioned VoltForge training and inference task contract."""

from .adapters import (
    evaluation_case_to_task_record,
    legacy_example_to_task_record,
    runtime_request_to_task_record,
    runtime_response_to_task_record,
)
from .compiler import compile_task_record, parse_compiled_sections
from .io import read_task_shard, write_task_shard
from .schema import (
    CONTRACT_VERSION,
    TaskContractError,
    build_task_json_schema,
    validate_task_record,
)

__all__ = [
    "CONTRACT_VERSION",
    "TaskContractError",
    "build_task_json_schema",
    "compile_task_record",
    "evaluation_case_to_task_record",
    "legacy_example_to_task_record",
    "parse_compiled_sections",
    "read_task_shard",
    "runtime_request_to_task_record",
    "runtime_response_to_task_record",
    "validate_task_record",
    "write_task_shard",
]
