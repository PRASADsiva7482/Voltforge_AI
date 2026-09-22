"""Canonical VoltForge task-record models and JSON Schema enforcement."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


CONTRACT_VERSION = "1.0.0"
SCHEMA_VERSION = 1
TASK_SCHEMA_ROOT = Path(__file__).resolve().parent
TASK_SCHEMA_PATH = TASK_SCHEMA_ROOT / "task-record.schema.json"
ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,159}$"
RECORD_ID_PATTERN = r"^vf-task-v1-[a-f0-9]{24}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
HIDDEN_APPLY_KEYS = {
    "autoapply",
    "applyautomatically",
    "executeimmediately",
    "skipconfirmation",
    "bypassconfirmation",
}


class TaskContractError(ValueError):
    """A precise task schema or semantic validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SystemRecord(ContractModel):
    type: Literal["system"] = "system"
    text: str = Field(min_length=1, max_length=16_000)
    policyVersion: str = Field(min_length=1, max_length=80)


class UserRecord(ContractModel):
    type: Literal["user"] = "user"
    text: str = Field(min_length=1, max_length=16_000)


class ProjectContextRecord(ContractModel):
    type: Literal["project-context"] = "project-context"
    projectId: str = Field(min_length=1, max_length=160)
    sourceProjectRevision: str = Field(pattern=ID_PATTERN)
    revisionSource: Literal["client", "derived-snapshot", "synthetic", "evaluation"]
    boardType: str | None = Field(default=None, max_length=80)
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolEvidenceRecord(ContractModel):
    type: Literal["tool-evidence"] = "tool-evidence"
    evidenceId: str = Field(pattern=ID_PATTERN)
    toolName: str = Field(pattern=ID_PATTERN)
    toolVersion: str = Field(min_length=1, max_length=80)
    status: Literal["complete", "failed", "reported", "unavailable"]
    authority: Literal["deterministic", "client-reported", "retrieved"]
    sourceProjectRevision: str = Field(pattern=ID_PATTERN)
    summary: str = Field(min_length=1, max_length=2_000)
    payload: dict[str, Any] = Field(default_factory=dict)


class AssistantTextRecord(ContractModel):
    type: Literal["assistant-text"] = "assistant-text"
    text: str = Field(min_length=1, max_length=200_000)
    evidenceRefs: list[str] = Field(default_factory=list, max_length=100)


class StructuredActionRecord(ContractModel):
    type: Literal["structured-action"] = "structured-action"
    actionId: str = Field(pattern=ID_PATTERN)
    actionKind: Literal[
        "wire-suggestion",
        "component-addition",
        "component-removal",
        "value-change",
        "code-fix",
    ]
    sourceProjectRevision: str = Field(pattern=ID_PATTERN)
    applicationMode: Literal["proposal-only"] = "proposal-only"
    requiresUserConfirmation: Literal[True] = True
    evidenceRefs: list[str] = Field(default_factory=list, max_length=100)
    payload: dict[str, Any]


class CitationRecord(ContractModel):
    type: Literal["citation"] = "citation"
    citationId: str = Field(pattern=ID_PATTERN)
    sourceId: str = Field(pattern=ID_PATTERN)
    sourceRevision: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    locator: str | None = Field(default=None, max_length=2_000)
    contentSha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    evidenceRefs: list[str] = Field(default_factory=list, max_length=100)


class RefusalRecord(ContractModel):
    type: Literal["refusal"] = "refusal"
    reasonCode: Literal[
        "out-of-domain",
        "unsafe-request",
        "malformed-input",
        "capability-unavailable",
        "insufficient-evidence",
    ]
    message: str = Field(min_length=1, max_length=8_000)
    recoverable: bool


class UncertaintyRecord(ContractModel):
    type: Literal["uncertainty"] = "uncertainty"
    level: Literal["low", "medium", "high", "unknown"]
    reason: str = Field(min_length=1, max_length=4_000)
    missingEvidence: list[str] = Field(default_factory=list, max_length=100)


class TaskInput(ContractModel):
    system: SystemRecord
    user: UserRecord
    projectContext: ProjectContextRecord
    toolEvidence: list[ToolEvidenceRecord] = Field(default_factory=list, max_length=100)


class TaskOutput(ContractModel):
    assistantText: AssistantTextRecord | None = None
    structuredActions: list[StructuredActionRecord] = Field(default_factory=list, max_length=100)
    citations: list[CitationRecord] = Field(default_factory=list, max_length=100)
    refusal: RefusalRecord | None = None
    uncertainty: UncertaintyRecord | None = None

    @model_validator(mode="after")
    def has_visible_result(self) -> "TaskOutput":
        if self.assistantText is None and self.refusal is None:
            raise ValueError("output requires assistantText or refusal")
        return self


class GeneratorRecord(ContractModel):
    id: str = Field(pattern=ID_PATTERN)
    version: str = Field(min_length=1, max_length=80)


class TaskMetadata(ContractModel):
    sourceKind: Literal["synthetic", "evaluation", "runtime", "curated"]
    sourceIds: list[str] = Field(default_factory=list, max_length=100)
    generator: GeneratorRecord | None = None


class TaskRecord(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = CONTRACT_VERSION
    recordId: str = Field(pattern=RECORD_ID_PATTERN)
    recordKind: Literal["example", "inference-request", "inference-response"]
    task: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    input: TaskInput
    output: TaskOutput | None = None
    metadata: TaskMetadata

    @model_validator(mode="after")
    def semantic_contract(self) -> "TaskRecord":
        if self.recordKind in {"example", "inference-response"} and self.output is None:
            raise ValueError("example and inference-response records require output")
        if self.recordKind == "inference-request" and self.output is not None:
            raise ValueError("inference-request records cannot contain model output")

        revision = self.input.projectContext.sourceProjectRevision
        evidence_ids = {item.evidenceId for item in self.input.toolEvidence}
        if len(evidence_ids) != len(self.input.toolEvidence):
            raise ValueError("tool evidence IDs must be unique")
        if any(item.sourceProjectRevision != revision for item in self.input.toolEvidence):
            raise ValueError("tool evidence must reference the source project revision")

        if self.output is None:
            return self
        actions = self.output.structuredActions
        action_ids = {item.actionId for item in actions}
        if len(action_ids) != len(actions):
            raise ValueError("structured action IDs must be unique")
        if any(item.sourceProjectRevision != revision for item in actions):
            raise ValueError("structured actions must reference the source project revision")
        for action in actions:
            stack: list[Any] = [action.payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    for key, child in item.items():
                        normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                        if normalized_key in HIDDEN_APPLY_KEYS:
                            raise ValueError(
                                f"structured action payload contains forbidden execution control {key!r}"
                            )
                        stack.append(child)
                elif isinstance(item, list):
                    stack.extend(item)

        referenced = set()
        if self.output.assistantText is not None:
            referenced.update(self.output.assistantText.evidenceRefs)
        for action in actions:
            referenced.update(action.evidenceRefs)
        for citation in self.output.citations:
            referenced.update(citation.evidenceRefs)
        unknown = sorted(referenced - evidence_ids)
        if unknown:
            raise ValueError(f"output references unknown tool evidence IDs: {unknown}")
        citation_ids = {item.citationId for item in self.output.citations}
        if len(citation_ids) != len(self.output.citations):
            raise ValueError("citation IDs must be unique")
        if len(set(self.metadata.sourceIds)) != len(self.metadata.sourceIds):
            raise ValueError("metadata source IDs must be unique")
        return self


def build_task_json_schema() -> dict[str, Any]:
    schema = TaskRecord.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:voltforge-ai:task-record:v1"
    schema["title"] = "VoltForge task record v1"
    return schema


@lru_cache(maxsize=1)
def _checked_schema() -> dict[str, Any]:
    try:
        schema = json.loads(TASK_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TaskContractError(
            "TASK_SCHEMA_UNAVAILABLE", "Checked-in VoltForge task JSON Schema is unavailable."
        ) from error
    if schema != build_task_json_schema():
        raise TaskContractError(
            "TASK_SCHEMA_STALE",
            "Checked-in task-record.schema.json does not match the executable v1 contract.",
        )
    return schema


def _resolve_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise TaskContractError("TASK_SCHEMA_INVALID", "Only local JSON Schema references are allowed.")
    current: Any = root
    for part in reference[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or key not in current:
            raise TaskContractError("TASK_SCHEMA_INVALID", f"Unresolved JSON Schema reference: {reference}")
        current = current[key]
    if not isinstance(current, Mapping):
        raise TaskContractError("TASK_SCHEMA_INVALID", f"JSON Schema reference is not an object: {reference}")
    return current


def _schema_validate(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> None:
    if "$ref" in schema:
        _schema_validate(value, _resolve_ref(root, str(schema["$ref"])), root, path)
        return
    for keyword in ("allOf",):
        for child in schema.get(keyword, []):
            _schema_validate(value, child, root, path)
    if "anyOf" in schema:
        failures = []
        for child in schema["anyOf"]:
            try:
                _schema_validate(value, child, root, path)
                break
            except TaskContractError as error:
                failures.append(error.message)
        else:
            raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} matches no allowed schema: {failures}")
        return
    if "oneOf" in schema:
        matches = 0
        for child in schema["oneOf"]:
            try:
                _schema_validate(value, child, root, path)
                matches += 1
            except TaskContractError:
                pass
        if matches != 1:
            raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} must match exactly one schema")
        return

    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected] if expected else []
    type_checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected_types and not any(type_checks[item](value) for item in expected_types):
        raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} has the wrong JSON type")
    if "const" in schema and value != schema["const"]:
        raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} contains an unsupported value")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} is too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} is too long")
        if "pattern" in schema and re.fullmatch(str(schema["pattern"]), value) is None:
            raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} does not match its pattern")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} contains too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} contains too many items")
        child = schema.get("items")
        if isinstance(child, Mapping):
            for index, item in enumerate(value):
                _schema_validate(item, child, root, f"{path}[{index}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise TaskContractError("TASK_SCHEMA_VALIDATION_FAILED", f"{path} is missing {missing}")
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise TaskContractError(
                    "TASK_SCHEMA_VALIDATION_FAILED", f"{path} has unexpected fields: {unexpected}"
                )
        for name, item in value.items():
            child = properties.get(name)
            if isinstance(child, Mapping):
                _schema_validate(item, child, root, f"{path}.{name}")


def validate_json_schema_document(value: Any, schema: Mapping[str, Any]) -> None:
    """Validate JSON-compatible data against a local, dependency-free schema."""

    _schema_validate(value, schema, schema, "$")


def validate_task_record(value: Mapping[str, Any] | TaskRecord) -> dict[str, Any]:
    """Validate against checked-in JSON Schema and semantic cross-record rules."""

    candidate = value.model_dump(mode="json", exclude_none=True) if isinstance(value, TaskRecord) else dict(value)
    schema = _checked_schema()
    validate_json_schema_document(candidate, schema)
    try:
        model = TaskRecord.model_validate(candidate)
    except Exception as error:
        raise TaskContractError("TASK_SEMANTIC_VALIDATION_FAILED", str(error)) from error
    return model.model_dump(mode="json", exclude_none=True)
