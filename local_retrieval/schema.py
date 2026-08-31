"""Strict contracts for the VFAI-022 curated local retrieval boundary."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from task_schema.schema import TaskContractError, validate_json_schema_document


RETRIEVAL_CONTRACT_VERSION = "1.0.0"
RETRIEVAL_SOURCE_REVISION = "1.1.0"
RETRIEVAL_INDEX_VERSION = "1.1.0"
RETRIEVAL_POLICY_ID = "vfai022-curated-local-retrieval-v1"
RETRIEVAL_INDEX_ID = "vf-curated-local-index-v1"
RETRIEVAL_ROOT = Path(__file__).resolve().parent
RETRIEVAL_POLICY_PATH = RETRIEVAL_ROOT / "policy.v1.json"
RETRIEVAL_INDEX_PATH = RETRIEVAL_ROOT / "index" / "v1" / "index.json"
RETRIEVAL_INDEX_SCHEMA_PATH = RETRIEVAL_ROOT / "retrieval-index.schema.json"
RETRIEVAL_RESPONSE_SCHEMA_PATH = RETRIEVAL_ROOT / "retrieval-response.schema.json"
ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,199}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
RECORD_TYPES = (
    "board",
    "pin-map",
    "component",
    "wiring-recipe",
    "firmware-api",
    "compiler-diagnostic",
    "simulation-behavior",
    "safety-constraint",
)
RecordType = Literal[
    "board",
    "pin-map",
    "component",
    "wiring-recipe",
    "firmware-api",
    "compiler-diagnostic",
    "simulation-behavior",
    "safety-constraint",
]


class RetrievalContractError(ValueError):
    """Content-free local-index contract or integrity failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RetrievalQuery(ContractModel):
    text: str = Field(min_length=1, max_length=2_000)
    maximumResults: int = Field(default=5, ge=1, le=8)
    boardFilters: list[str] = Field(default_factory=list, max_length=10)
    componentFilters: list[str] = Field(default_factory=list, max_length=20)
    recordTypes: list[RecordType] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def unique_filters(self) -> "RetrievalQuery":
        for name in ("boardFilters", "componentFilters", "recordTypes"):
            values = getattr(self, name)
            if len(values) != len({item.casefold() for item in values}):
                raise ValueError(f"{name} must be case-insensitively unique")
        return self


class IndexedFact(ContractModel):
    factId: str = Field(pattern=ID_PATTERN)
    claimId: str = Field(pattern=ID_PATTERN)
    property: str = Field(pattern=r"^[a-z][a-z0-9-]{1,79}$")
    pointer: str = Field(max_length=500)
    value: Any = None
    unit: str | None = Field(default=None, max_length=80)
    status: Literal["verified", "bounded", "conditional", "unknown"]
    conditions: list[str] = Field(default_factory=list, max_length=30)
    evidenceRefs: list[str] = Field(min_length=1, max_length=20)


class RetrievalEvidence(ContractModel):
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
    verifiedAt: str = Field(pattern=r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$")


class RetrievalSubject(ContractModel):
    subjectId: str = Field(pattern=ID_PATTERN)
    familyId: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=300)
    variant: str = Field(min_length=1, max_length=200)


class RetrievalSource(ContractModel):
    sourceId: Literal["vf-src-curated-electronics-corpus-v1"]
    sourceRevision: Literal[RETRIEVAL_SOURCE_REVISION]
    recordRevision: str = Field(pattern=ID_PATTERN)
    validFrom: str = Field(pattern=r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$")
    recordId: str = Field(pattern=r"^vf-knowledge-v1-[a-z0-9][a-z0-9.-]{2,119}$")


class RetrievalIndexChunk(ContractModel):
    chunkId: str = Field(pattern=r"^vf-retrieval-chunk-v1-[a-f0-9]{24}$")
    recordId: str = Field(pattern=r"^vf-knowledge-v1-[a-z0-9][a-z0-9.-]{2,119}$")
    recordType: RecordType
    supportStatus: Literal["supported", "variant-required", "reference-only"]
    subject: RetrievalSubject
    source: RetrievalSource
    facts: list[IndexedFact] = Field(min_length=1, max_length=8)
    evidence: list[RetrievalEvidence] = Field(min_length=1, max_length=20)
    boardKeys: list[str] = Field(default_factory=list, max_length=100)
    componentKeys: list[str] = Field(default_factory=list, max_length=100)
    termFrequencies: dict[str, int] = Field(min_length=1, max_length=1_000)
    documentLength: int = Field(ge=1, le=20_000)
    contentSha256: str = Field(pattern=SHA256_PATTERN)


class RetrievalIndex(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = RETRIEVAL_CONTRACT_VERSION
    indexId: Literal["vf-curated-local-index-v1"] = RETRIEVAL_INDEX_ID
    indexVersion: Literal[RETRIEVAL_INDEX_VERSION] = RETRIEVAL_INDEX_VERSION
    policyId: Literal["vfai022-curated-local-retrieval-v1"] = RETRIEVAL_POLICY_ID
    policySha256: str = Field(pattern=SHA256_PATTERN)
    algorithm: Literal["bm25-lexical-v1"] = "bm25-lexical-v1"
    sourceId: Literal["vf-src-curated-electronics-corpus-v1"]
    sourceRevision: Literal[RETRIEVAL_SOURCE_REVISION]
    sourceEntrySha256: str = Field(pattern=SHA256_PATTERN)
    sourceCatalogSha256: str = Field(pattern=SHA256_PATTERN)
    corpusContentSha256: str = Field(pattern=SHA256_PATTERN)
    builderSha256: str = Field(pattern=SHA256_PATTERN)
    recordCount: int = Field(ge=1)
    chunkCount: int = Field(ge=1)
    averageDocumentLength: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+)?$")
    documentFrequencies: dict[str, int] = Field(min_length=1, max_length=20_000)
    chunks: list[RetrievalIndexChunk] = Field(min_length=1, max_length=10_000)
    rawSourceDocumentsStored: Literal[False] = False
    embeddingsPresent: Literal[False] = False
    indexSha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def index_consistency(self) -> "RetrievalIndex":
        if self.chunkCount != len(self.chunks):
            raise ValueError("chunkCount does not match chunks")
        ids = [item.chunkId for item in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieval chunk IDs must be unique")
        if self.recordCount != len({item.recordId for item in self.chunks}):
            raise ValueError("recordCount does not match indexed record IDs")
        for term, frequency in self.documentFrequencies.items():
            if not term or frequency < 1 or frequency > self.chunkCount:
                raise ValueError("document frequency is outside the index boundary")
        return self


class RetrievalCitation(ContractModel):
    citationId: str = Field(pattern=r"^citation:local:[a-f0-9]{24}$")
    sourceId: str = Field(pattern=ID_PATTERN)
    sourceRevision: str = Field(min_length=1, max_length=160)
    recordId: str = Field(pattern=r"^vf-knowledge-v1-[a-z0-9][a-z0-9.-]{2,119}$")
    title: str = Field(min_length=1, max_length=500)
    locator: str = Field(min_length=1, max_length=1_000)
    contentSha256: str = Field(pattern=SHA256_PATTERN)


class RetrievalResult(ContractModel):
    resultId: str = Field(pattern=r"^retrieval:local:[a-f0-9]{24}$")
    rank: int = Field(ge=1, le=8)
    score: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+)?$")
    matchedTerms: list[str] = Field(min_length=1, max_length=24)
    chunkId: str = Field(pattern=r"^vf-retrieval-chunk-v1-[a-f0-9]{24}$")
    recordId: str = Field(pattern=r"^vf-knowledge-v1-[a-z0-9][a-z0-9.-]{2,119}$")
    recordType: RecordType
    supportStatus: Literal["supported", "variant-required", "reference-only"]
    subject: RetrievalSubject
    source: RetrievalSource
    facts: list[IndexedFact] = Field(min_length=1, max_length=8)
    evidence: list[RetrievalEvidence] = Field(min_length=1, max_length=12)
    contentSha256: str = Field(pattern=SHA256_PATTERN)
    citation: RetrievalCitation


class RetrievalResponse(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = RETRIEVAL_CONTRACT_VERSION
    policyId: Literal["vfai022-curated-local-retrieval-v1"] = RETRIEVAL_POLICY_ID
    policySha256: str = Field(pattern=SHA256_PATTERN)
    responseId: str = Field(pattern=r"^vf-retrieval-response-v1-[a-f0-9]{24}$")
    status: Literal["complete", "no-results", "unavailable"]
    reasonCode: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,99}$")
    querySha256: str = Field(pattern=SHA256_PATTERN)
    indexId: Literal["vf-curated-local-index-v1"]
    indexVersion: Literal[RETRIEVAL_INDEX_VERSION]
    indexSha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    sourceId: Literal["vf-src-curated-electronics-corpus-v1"]
    sourceRevision: Literal[RETRIEVAL_SOURCE_REVISION]
    results: list[RetrievalResult] = Field(default_factory=list, max_length=8)
    candidateCount: int = Field(default=0, ge=0)
    returnedCount: int = Field(default=0, ge=0, le=8)
    filtersApplied: dict[str, list[str]] = Field(default_factory=dict)
    degraded: bool = False
    embeddingsUsed: Literal[False] = False
    networkAccessed: Literal[False] = False
    rawQueryStored: Literal[False] = False
    rawProjectContextStored: Literal[False] = False

    @model_validator(mode="after")
    def response_consistency(self) -> "RetrievalResponse":
        if self.returnedCount != len(self.results):
            raise ValueError("returnedCount does not match results")
        if self.status == "complete" and not self.results:
            raise ValueError("complete retrieval requires at least one result")
        if self.status != "complete" and self.results:
            raise ValueError("non-complete retrieval cannot expose results")
        if self.status == "unavailable" and not self.degraded:
            raise ValueError("unavailable retrieval must be explicitly degraded")
        if self.status != "unavailable" and self.degraded:
            raise ValueError("only unavailable retrieval may be degraded")
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
def load_retrieval_policy() -> dict[str, Any]:
    try:
        policy = json.loads(RETRIEVAL_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalContractError(
            "RETRIEVAL_POLICY_INVALID", "The local retrieval policy is unreadable."
        ) from error
    if not isinstance(policy, dict) or policy.get("policyId") != RETRIEVAL_POLICY_ID:
        raise RetrievalContractError(
            "RETRIEVAL_POLICY_INVALID", "The local retrieval policy identity is invalid."
        )
    declared = policy.get("policySha256")
    unsigned = dict(policy)
    unsigned.pop("policySha256", None)
    if declared != sha256_json(unsigned):
        raise RetrievalContractError(
            "RETRIEVAL_POLICY_CHECKSUM_MISMATCH",
            "The local retrieval policy checksum is invalid.",
        )
    if policy.get("algorithm", {}).get("embeddingsEnabled") is not False:
        raise RetrievalContractError(
            "RETRIEVAL_POLICY_INVALID", "Unapproved retrieval embeddings are enabled."
        )
    return policy


def build_index_json_schema() -> dict[str, Any]:
    schema = RetrievalIndex.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:voltforge-ai:curated-local-retrieval-index:v1"
    schema["title"] = "VoltForge curated local retrieval index v1"
    return schema


def build_response_json_schema() -> dict[str, Any]:
    schema = RetrievalResponse.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:voltforge-ai:curated-local-retrieval-response:v1"
    schema["title"] = "VoltForge curated local retrieval response v1"
    return schema


def _checked_schema(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    try:
        checked = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalContractError(
            "RETRIEVAL_SCHEMA_UNAVAILABLE", "A checked local retrieval schema is unavailable."
        ) from error
    if checked != dict(expected):
        raise RetrievalContractError(
            "RETRIEVAL_SCHEMA_STALE",
            "A checked local retrieval schema does not match the executable contract.",
        )
    return checked


@lru_cache(maxsize=1)
def checked_index_schema() -> dict[str, Any]:
    return _checked_schema(RETRIEVAL_INDEX_SCHEMA_PATH, build_index_json_schema())


@lru_cache(maxsize=1)
def checked_response_schema() -> dict[str, Any]:
    return _checked_schema(RETRIEVAL_RESPONSE_SCHEMA_PATH, build_response_json_schema())


def validate_retrieval_index(value: Mapping[str, Any] | RetrievalIndex) -> RetrievalIndex:
    candidate = value.model_dump(mode="json") if isinstance(value, RetrievalIndex) else dict(value)
    try:
        validate_json_schema_document(candidate, checked_index_schema())
        return RetrievalIndex.model_validate(candidate)
    except (TaskContractError, ValueError) as error:
        raise RetrievalContractError(
            "RETRIEVAL_INDEX_SCHEMA_INVALID", "The local retrieval index contract is invalid."
        ) from error


def validate_retrieval_response(
    value: Mapping[str, Any] | RetrievalResponse,
) -> RetrievalResponse:
    candidate = (
        value.model_dump(mode="json") if isinstance(value, RetrievalResponse) else dict(value)
    )
    try:
        validate_json_schema_document(candidate, checked_response_schema())
        return RetrievalResponse.model_validate(candidate)
    except (TaskContractError, ValueError) as error:
        raise RetrievalContractError(
            "RETRIEVAL_RESPONSE_SCHEMA_INVALID",
            "The local retrieval response contract is invalid.",
        ) from error
