"""Strict contracts and policy verification for VFAI-025 memory."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


POLICY_ID = "vfai025-bounded-memory-v1"
POLICY_SHA256 = "186045148c7a7a6afbae7aedaafe4aa961f16cd8bdf7f5066fb0dc3f4e92869d"
CONTRACT_VERSION = "1.0.0"
PACKAGE_DIR = Path(__file__).resolve().parent
POLICY_PATH = PACKAGE_DIR / "policy.v1.json"
STATE_SCHEMA_PATH = PACKAGE_DIR / "memory-state.schema.json"
ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,159}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"


class MemoryContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


MemoryKind = Literal["fact", "decision", "summary", "recent-turn"]
MemoryScopeKind = Literal["project", "session"]


class MemoryWriteRequest(ContractModel):
    kind: MemoryKind
    scope: MemoryScopeKind = "project"
    content: str = Field(min_length=1, max_length=1200)
    projectRevision: str = Field(min_length=1, max_length=160)
    expiresInHours: int | None = Field(default=None, ge=1, le=2160)
    approved: Literal[True]

    @model_validator(mode="after")
    def recent_turn_requires_session_scope(self) -> "MemoryWriteRequest":
        if self.kind == "recent-turn" and self.scope != "session":
            raise ValueError("recent-turn memory requires session scope")
        return self


class MemoryCorrectionRequest(ContractModel):
    content: str = Field(min_length=1, max_length=1200)
    projectRevision: str = Field(min_length=1, max_length=160)
    expiresInHours: int | None = Field(default=None, ge=1, le=2160)
    expectedVersion: int = Field(ge=1)
    approved: Literal[True]


class MemoryPreferenceRequest(ContractModel):
    enabled: bool
    clearOnDisable: bool = False


class MemoryEntry(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = CONTRACT_VERSION
    memoryId: str = Field(pattern=r"^memory:v1:[a-f0-9]{24}$")
    kind: MemoryKind
    scope: MemoryScopeKind
    content: str = Field(min_length=1, max_length=1200)
    contentSha256: str = Field(pattern=SHA256_PATTERN)
    projectRevision: str = Field(min_length=1, max_length=160)
    version: int = Field(ge=1)
    source: Literal["user-approved", "recent-turn-summary"]
    createdAt: str = Field(min_length=20, max_length=40)
    updatedAt: str = Field(min_length=20, max_length=40)
    expiresAt: str = Field(min_length=20, max_length=40)
    staleForProjectRevision: bool
    redactionApplied: bool
    authority: Literal["user-memory"] = "user-memory"
    modelEvidenceAllowed: Literal[False] = False
    trainingUseAllowed: Literal[False] = False


class MemoryState(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = CONTRACT_VERSION
    policyId: Literal["vfai025-bounded-memory-v1"] = POLICY_ID
    policySha256: str = Field(pattern=SHA256_PATTERN)
    status: Literal["ready", "disabled", "unavailable"]
    reasonCode: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,99}$")
    enabled: bool
    authenticatedScope: Literal[True] = True
    projectRevision: str | None = Field(default=None, max_length=160)
    entries: list[MemoryEntry] = Field(default_factory=list, max_length=50)
    entryCount: int = Field(ge=0, le=50)
    staleEntryCount: int = Field(ge=0, le=50)
    omittedEntryCount: int = Field(ge=0)
    expiredPurgedCount: int = Field(ge=0)
    evictedCount: int = Field(ge=0)
    totalBytes: int = Field(ge=0, le=24576)
    trainingUseAllowed: Literal[False] = False
    rawIdentifiersStored: Literal[False] = False
    rawProjectContextStored: Literal[False] = False
    hiddenReasoningStored: Literal[False] = False

    @model_validator(mode="after")
    def counts_match(self) -> "MemoryState":
        if self.entryCount != len(self.entries):
            raise ValueError("entryCount does not match entries")
        if self.staleEntryCount != sum(
            item.staleForProjectRevision for item in self.entries
        ):
            raise ValueError("staleEntryCount does not match entries")
        if self.totalBytes != sum(len(item.content.encode("utf-8")) for item in self.entries):
            raise ValueError("totalBytes does not match entries")
        return self


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@lru_cache(maxsize=1)
def load_policy() -> dict[str, Any]:
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MemoryContractError(
            "MEMORY_POLICY_INVALID", "The bounded-memory policy is unreadable."
        ) from error
    if not isinstance(policy, dict) or policy.get("policyId") != POLICY_ID:
        raise MemoryContractError(
            "MEMORY_POLICY_INVALID", "The bounded-memory policy identity is invalid."
        )
    unsigned = dict(policy)
    declared = unsigned.pop("policySha256", None)
    if declared != POLICY_SHA256 or declared != sha256_json(unsigned):
        raise MemoryContractError(
            "MEMORY_POLICY_CHECKSUM_MISMATCH",
            "The bounded-memory policy checksum is invalid.",
        )
    if (
        policy.get("defaultEnabled") is not False
        or policy.get("consent", {}).get("trainingUseAllowed") is not False
        or policy.get("contentPolicy", {}).get("memoryCanSupportHighRiskGroundingClaims")
        is not False
    ):
        raise MemoryContractError(
            "MEMORY_POLICY_INVALID", "A mandatory memory invariant changed."
        )
    return policy


def build_state_json_schema() -> dict[str, Any]:
    schema = MemoryState.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:voltforge-ai:bounded-memory-state:v1"
    schema["title"] = "VoltForge bounded memory state v1"
    return schema


def checked_state_schema() -> dict[str, Any]:
    try:
        checked = json.loads(STATE_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MemoryContractError(
            "MEMORY_SCHEMA_INVALID", "The checked memory schema is unreadable."
        ) from error
    if checked != build_state_json_schema():
        raise MemoryContractError(
            "MEMORY_SCHEMA_DRIFT", "The checked memory schema has drifted."
        )
    return checked
