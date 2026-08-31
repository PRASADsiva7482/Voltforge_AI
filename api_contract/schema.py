"""Strict versioned request, response, and SSE contracts for VFAI-026."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


POLICY_ID = "vfai026-fastapi-sse-contract-v1"
POLICY_SHA256 = "bcbdef3deaccdf3e155fcf0209662c8fda3ae62d1989e2c9d1a6a02edcdba27d"
CONTRACT_VERSION = "1.0.0"
SCHEMA_VERSION = 1
PACKAGE_DIR = Path(__file__).resolve().parent
POLICY_PATH = PACKAGE_DIR / "policy.v1.json"
EVENT_SCHEMA_PATH = PACKAGE_DIR / "sse-event.schema.json"
REQUEST_SCHEMA_PATH = PACKAGE_DIR / "chat-request.schema.json"
RESPONSE_SCHEMA_PATH = PACKAGE_DIR / "chat-response.schema.json"
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,159}$"
REVISION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:@/+\-]{0,159}$"


class ApiContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ArtifactIdentity(ContractModel):
    artifactId: str | None = Field(default=None, max_length=160)
    artifactVersion: str = Field(min_length=1, max_length=160)
    registryRevision: int | None = Field(default=None, ge=0)
    runtimeState: str = Field(min_length=1, max_length=80)
    ready: bool


EventType = Literal[
    "start",
    "tool",
    "citation",
    "uncertainty",
    "delta",
    "proposal",
    "complete",
    "error",
]
GenerationMode = Literal[
    "neural-quality-gated", "deterministic-fallback", "unavailable"
]


class SseEvent(ContractModel):
    schemaVersion: Literal[1] = SCHEMA_VERSION
    contractVersion: Literal["1.0.0"] = CONTRACT_VERSION
    type: EventType
    eventId: str = Field(pattern=r"^event:v1:[A-Za-z0-9._:-]{3,160}:[0-9]{1,4}$")
    sequence: int = Field(ge=0, le=383)
    requestId: str = Field(pattern=ID_PATTERN)
    sessionId: str = Field(pattern=ID_PATTERN)
    projectRevision: str = Field(pattern=REVISION_PATTERN)
    model: Literal["voltforge-local-engine-v1"] = "voltforge-local-engine-v1"
    mode: GenerationMode
    artifact: ArtifactIdentity
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_typed_payload(self) -> "SseEvent":
        required = {
            "start": ("readiness", "fallbackUsed"),
            "tool": ("name", "status", "authority"),
            "citation": ("citationId", "evidenceKind"),
            "uncertainty": ("reasonCode",),
            "delta": ("delta",),
            "proposal": ("id", "sourceProjectRevision"),
            "complete": ("reply", "messageId", "taskRecordId"),
            "error": ("code", "message", "retryable"),
        }[self.type]
        missing = [key for key in required if key not in self.payload]
        if missing:
            raise ValueError(f"{self.type} payload is missing: {', '.join(missing)}")
        if self.type == "delta":
            delta = self.payload.get("delta")
            if not isinstance(delta, str) or not 1 <= len(delta) <= 96:
                raise ValueError("delta payload must contain 1..96 characters")
        if self.type == "complete":
            reply = self.payload.get("reply")
            if not isinstance(reply, str) or not 1 <= len(reply) <= 24_000:
                raise ValueError("complete reply must contain 1..24000 characters")
        if self.type == "error" and self.payload.get("reply") is not None:
            raise ValueError("error events cannot carry a reply")
        if self.type == "proposal":
            actions = self.payload.get("structuredActions") or []
            if not isinstance(actions, list) or len(actions) > 25:
                raise ValueError("proposal actions exceed the contract limit")
        return self


class ContractErrorResponse(ContractModel):
    schemaVersion: Literal[1] = SCHEMA_VERSION
    contractVersion: Literal["1.0.0"] = CONTRACT_VERSION
    requestId: str | None = Field(default=None, pattern=ID_PATTERN)
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,99}$")
    message: str = Field(min_length=1, max_length=500)
    retryable: bool
    fieldErrors: list[str] = Field(default_factory=list, max_length=32)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@lru_cache(maxsize=1)
def load_policy() -> dict[str, Any]:
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ApiContractError(
            "API_CONTRACT_POLICY_INVALID", "The API contract policy is unavailable."
        ) from error
    if hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() != POLICY_SHA256:
        raise ApiContractError(
            "API_CONTRACT_POLICY_CHECKSUM_MISMATCH",
            "The API contract policy checksum is invalid.",
        )
    if (
        policy.get("policyId") != POLICY_ID
        or policy.get("contractVersion") != CONTRACT_VERSION
        or policy.get("schemaVersion") != SCHEMA_VERSION
    ):
        raise ApiContractError(
            "API_CONTRACT_POLICY_IDENTITY_MISMATCH",
            "The API contract policy identity is invalid.",
        )
    return policy


def build_event_json_schema() -> dict[str, Any]:
    return SseEvent.model_json_schema()


def checked_event_schema() -> dict[str, Any]:
    return json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))


def checked_request_schema() -> dict[str, Any]:
    return json.loads(REQUEST_SCHEMA_PATH.read_text(encoding="utf-8"))


def checked_response_schema() -> dict[str, Any]:
    return json.loads(RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))
