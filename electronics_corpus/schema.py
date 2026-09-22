"""Strict v1 contract for VoltForge's curated electronics facts."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from task_schema.schema import TaskContractError, validate_json_schema_document


CORPUS_CONTRACT_VERSION = "1.0.0"
# The wire/schema contract remains compatible while each curated release gets
# its own immutable source revision.  VFAI-FU-001 is the first expanded board
# coverage release.
CORPUS_SOURCE_REVISION = "1.1.0"
CORPUS_SCHEMA_VERSION = 1
CORPUS_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_SCHEMA_PATH = CORPUS_ROOT / "knowledge-record.schema.json"
ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,159}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
DATE_PATTERN = r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$"

REQUIRED_CLAIMS: dict[str, set[str]] = {
    "board": {
        "support-status",
        "manufacturer",
        "mcu",
        "architecture",
        "logic-voltage-v",
        "flash-bytes",
        "sram-bytes",
    },
    "pin-map": {"board-record-id", "power-pins", "default-buses", "pin-capabilities"},
    "component": {"support-status", "manufacturer", "part-number", "terminals", "electrical-ratings"},
    "wiring-recipe": {"board-record-id", "component-record-id", "connections", "preconditions"},
    "firmware-api": {"framework", "version-policy", "include", "api-symbols", "supported-board-record-ids"},
    "compiler-diagnostic": {"toolchain", "diagnostic-pattern", "causes", "remediation"},
    "simulation-behavior": {"model", "supported-analyses", "assumptions", "limitations"},
    "safety-constraint": {"severity", "condition", "required-action", "unknown-policy"},
}


class KnowledgeContractError(ValueError):
    """Precise corpus schema, integrity, or lookup-contract failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EffectiveRevision(ContractModel):
    revision: str = Field(pattern=ID_PATTERN)
    validFrom: str = Field(pattern=DATE_PATTERN)
    validTo: str | None = Field(default=None, pattern=DATE_PATTERN)
    supersedes: str | None = Field(default=None, pattern=ID_PATTERN)


class EvidenceReference(ContractModel):
    evidenceId: str = Field(pattern=ID_PATTERN)
    evidenceKind: Literal[
        "manufacturer-documentation",
        "maintainer-documentation",
        "toolchain-documentation",
        "project-engineering-rule",
    ]
    publisher: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    documentRevision: str = Field(min_length=1, max_length=160)
    locator: str = Field(min_length=1, max_length=1_000)
    uri: str | None = Field(default=None, max_length=2_000)
    verifiedAt: str = Field(pattern=DATE_PATTERN)
    extractionPolicy: Literal["normalized-facts-only-no-source-prose"] = (
        "normalized-facts-only-no-source-prose"
    )


class Provenance(ContractModel):
    sourceId: Literal["vf-src-curated-electronics-corpus-v1"] = (
        "vf-src-curated-electronics-corpus-v1"
    )
    sourceRevision: str = Field(pattern=ID_PATTERN)
    curator: Literal["VoltForge"] = "VoltForge"
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=20)


class Subject(ContractModel):
    subjectId: str = Field(pattern=ID_PATTERN)
    familyId: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=300)
    variant: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=50)


class Claim(ContractModel):
    claimId: str = Field(pattern=ID_PATTERN)
    property: str = Field(pattern=r"^[a-z][a-z0-9-]{1,79}$")
    value: Any = None
    unit: str | None = Field(default=None, max_length=80)
    status: Literal["verified", "bounded", "conditional", "unknown"]
    conditions: list[str] = Field(default_factory=list, max_length=30)
    evidenceRefs: list[str] = Field(min_length=1, max_length=20)


class Relation(ContractModel):
    relationType: Literal[
        "describes",
        "applies-to",
        "requires",
        "compatible-with",
        "conflicts-with",
        "supersedes",
    ]
    targetRecordId: str = Field(pattern=ID_PATTERN)


class KnowledgeRecord(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = CORPUS_CONTRACT_VERSION
    recordId: str = Field(pattern=r"^vf-knowledge-v1-[a-z0-9][a-z0-9.-]{2,119}$")
    recordType: Literal[
        "board",
        "pin-map",
        "component",
        "wiring-recipe",
        "firmware-api",
        "compiler-diagnostic",
        "simulation-behavior",
        "safety-constraint",
    ]
    supportStatus: Literal["supported", "variant-required", "reference-only"]
    conflictGroup: str | None = Field(default=None, pattern=ID_PATTERN)
    subject: Subject
    effectiveRevision: EffectiveRevision
    provenance: Provenance
    claims: list[Claim] = Field(min_length=1, max_length=100)
    relations: list[Relation] = Field(default_factory=list, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def semantic_contract(self) -> "KnowledgeRecord":
        evidence_ids = [item.evidenceId for item in self.provenance.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique within a record")
        claim_ids = [item.claimId for item in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim IDs must be unique within a record")
        properties = {item.property for item in self.claims}
        missing = sorted(REQUIRED_CLAIMS[self.recordType] - properties)
        if missing:
            raise ValueError(f"{self.recordType} record is missing required claims: {missing}")
        evidence_set = set(evidence_ids)
        for claim in self.claims:
            unknown_refs = sorted(set(claim.evidenceRefs) - evidence_set)
            if unknown_refs:
                raise ValueError(f"claim {claim.claimId} references unknown evidence: {unknown_refs}")
            if claim.status == "unknown" and claim.value is not None:
                raise ValueError(f"unknown claim {claim.claimId} must have a null value")
            if claim.status != "unknown" and claim.value is None:
                raise ValueError(f"supported claim {claim.claimId} requires a value")
            try:
                json.dumps(claim.value, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(f"claim {claim.claimId} value is not strict JSON") from error
            if isinstance(claim.value, float) and not math.isfinite(claim.value):
                raise ValueError(f"claim {claim.claimId} value must be finite")
        if self.supportStatus == "supported" and all(
            claim.status == "unknown" for claim in self.claims
        ):
            raise ValueError("supported records require at least one evidenced claim")
        if self.effectiveRevision.validTo is not None and (
            self.effectiveRevision.validTo < self.effectiveRevision.validFrom
        ):
            raise ValueError("effective revision validTo cannot precede validFrom")
        if len(self.subject.aliases) != len({item.casefold() for item in self.subject.aliases}):
            raise ValueError("subject aliases must be case-insensitively unique")
        return self


def build_knowledge_json_schema() -> dict[str, Any]:
    schema = KnowledgeRecord.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:voltforge-ai:electronics-knowledge-record:v1"
    schema["title"] = "VoltForge curated electronics knowledge record v1"
    return schema


@lru_cache(maxsize=1)
def checked_knowledge_schema() -> dict[str, Any]:
    try:
        schema = json.loads(KNOWLEDGE_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeContractError(
            "KNOWLEDGE_SCHEMA_UNAVAILABLE",
            "Checked-in electronics knowledge JSON Schema is unavailable.",
        ) from error
    if schema != build_knowledge_json_schema():
        raise KnowledgeContractError(
            "KNOWLEDGE_SCHEMA_STALE",
            "Checked-in knowledge-record.schema.json does not match the executable v1 contract.",
        )
    return schema


def validate_knowledge_record(
    value: Mapping[str, Any] | KnowledgeRecord,
) -> dict[str, Any]:
    """Validate a knowledge record against checked schema and semantic rules."""

    candidate = (
        value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, KnowledgeRecord)
        else dict(value)
    )
    try:
        validate_json_schema_document(candidate, checked_knowledge_schema())
    except TaskContractError as error:
        raise KnowledgeContractError("KNOWLEDGE_SCHEMA_VALIDATION_FAILED", error.message) from error
    try:
        model = KnowledgeRecord.model_validate(candidate)
    except Exception as error:
        raise KnowledgeContractError("KNOWLEDGE_SEMANTIC_VALIDATION_FAILED", str(error)) from error
    return model.model_dump(mode="json", exclude_none=True)
