"""Strict contracts for VFAI-024 claim grounding metadata."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


POLICY_ID = "vfai024-claim-grounding-v1"
POLICY_SHA256 = "1497847cdadb41aeaf0f52ed7e36d9fa0712c805a8b26318291dd50c102c25ab"
CONTRACT_VERSION = "1.0.0"
PACKAGE_DIR = Path(__file__).resolve().parent
POLICY_PATH = PACKAGE_DIR / "policy.v1.json"
REPORT_SCHEMA_PATH = PACKAGE_DIR / "grounding-report.schema.json"
ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,159}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"


class GroundingContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


EvidenceKind = Literal["project", "deterministic", "local", "internet"]
ClaimType = Literal["datasheet-rating", "pin-capability", "library-api", "current-web"]


class GroundingCitation(ContractModel):
    citationId: str = Field(pattern=ID_PATTERN)
    evidenceKind: EvidenceKind
    authority: Literal["project-state", "deterministic", "retrieved"]
    sourceId: str = Field(pattern=ID_PATTERN)
    sourceRevision: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=500)
    locator: str | None = Field(default=None, max_length=2_000)
    url: str | None = Field(default=None, max_length=2_000)
    snippet: str | None = Field(default=None, max_length=800)
    contentSha256: str = Field(pattern=SHA256_PATTERN)
    modelPayloadSha256: str = Field(pattern=SHA256_PATTERN)
    evidenceRefs: list[str] = Field(default_factory=list, max_length=4)
    claimIds: list[str] = Field(default_factory=list, max_length=24)
    supportStatus: Literal["supported", "conflicted", "reference-only"]
    sourceTimestamp: str | None = Field(default=None, max_length=64)
    exactModelEvidence: Literal[True] = True
    untrustedContent: bool = False


class GroundedClaim(ContractModel):
    claimId: str = Field(pattern=r"^claim:grounding:[a-f0-9]{24}$")
    claimTypes: list[ClaimType] = Field(min_length=1, max_length=4)
    supportStatus: Literal["supported", "unsupported", "conflicted", "unknown"]
    reasonCode: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,99}$")
    citationIds: list[str] = Field(default_factory=list, max_length=2)
    evidenceRefs: list[str] = Field(default_factory=list, max_length=8)
    visibleTextTransformed: bool = False


class EvidenceConflict(ContractModel):
    conflictId: str = Field(pattern=r"^conflict:grounding:[a-f0-9]{24}$")
    claimType: ClaimType
    subject: str = Field(min_length=1, max_length=200)
    property: str = Field(min_length=1, max_length=160)
    valueHashes: list[str] = Field(min_length=2, max_length=8)
    citationIds: list[str] = Field(min_length=2, max_length=8)
    evidenceRefs: list[str] = Field(default_factory=list, max_length=16)
    resolution: Literal["visible-uncertainty-required"] = "visible-uncertainty-required"


class GroundingUncertainty(ContractModel):
    level: Literal["none", "medium", "high"]
    reasonCode: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,99}$")
    missingEvidence: list[str] = Field(default_factory=list, max_length=12)


class GroundingReport(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = CONTRACT_VERSION
    policyId: Literal["vfai024-claim-grounding-v1"] = POLICY_ID
    policySha256: str = Field(pattern=SHA256_PATTERN)
    reportId: str = Field(pattern=r"^vf-grounding-report-v1-[a-f0-9]{24}$")
    status: Literal["grounded", "uncertain", "conflicted", "no-high-risk-claims"]
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=24)
    conflicts: list[EvidenceConflict] = Field(default_factory=list, max_length=12)
    claimCount: int = Field(ge=0, le=24)
    supportedClaimCount: int = Field(ge=0, le=24)
    unsupportedClaimCount: int = Field(ge=0, le=24)
    conflictedClaimCount: int = Field(ge=0, le=24)
    citationCount: int = Field(ge=0, le=25)
    usedEvidenceRefs: list[str] = Field(default_factory=list, max_length=100)
    uncertainty: GroundingUncertainty
    maximumConfidence: float = Field(ge=0.0, le=1.0)
    allCitationIdsResolved: Literal[True] = True
    exactModelPayloadHashesVerified: Literal[True] = True
    rawPromptStored: Literal[False] = False
    rawProjectContextStored: Literal[False] = False
    rawModelOutputStored: Literal[False] = False
    claimTextStoredInMetadata: Literal[False] = False

    @model_validator(mode="after")
    def report_consistency(self) -> "GroundingReport":
        if self.claimCount != len(self.claims):
            raise ValueError("claimCount does not match claims")
        statuses = [item.supportStatus for item in self.claims]
        if self.supportedClaimCount != statuses.count("supported"):
            raise ValueError("supportedClaimCount does not match claims")
        if self.unsupportedClaimCount != statuses.count("unsupported"):
            raise ValueError("unsupportedClaimCount does not match claims")
        if self.conflictedClaimCount != statuses.count("conflicted"):
            raise ValueError("conflictedClaimCount does not match claims")
        if self.status == "grounded" and any(
            item in {"unsupported", "conflicted"} for item in statuses
        ):
            raise ValueError("grounded report contains ungrounded claims")
        if self.status == "conflicted" and not self.conflictedClaimCount:
            raise ValueError("conflicted report requires a conflicted claim")
        if self.status == "uncertain" and not self.unsupportedClaimCount:
            raise ValueError("uncertain report requires an unsupported claim")
        if self.status == "no-high-risk-claims" and self.claims:
            raise ValueError("no-high-risk-claims report cannot contain claims")
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
        raise GroundingContractError(
            "GROUNDING_POLICY_INVALID", "The claim-grounding policy is unreadable."
        ) from error
    if not isinstance(policy, dict) or policy.get("policyId") != POLICY_ID:
        raise GroundingContractError(
            "GROUNDING_POLICY_INVALID", "The claim-grounding policy identity is invalid."
        )
    unsigned = dict(policy)
    declared = unsigned.pop("policySha256", None)
    if declared != POLICY_SHA256 or declared != sha256_json(unsigned):
        raise GroundingContractError(
            "GROUNDING_POLICY_CHECKSUM_MISMATCH",
            "The claim-grounding policy checksum is invalid.",
        )
    if (
        policy.get("citationPolicy", {}).get("citationMustResolveToModelEvidence")
        is not True
        or policy.get("authority", {}).get("deterministicEngineeringCanBeOverridden")
        is not False
        or policy.get("authority", {}).get("retrievalCanAuthorizeStructuredActions")
        is not False
    ):
        raise GroundingContractError(
            "GROUNDING_POLICY_INVALID", "A mandatory grounding invariant changed."
        )
    return policy


def build_report_json_schema() -> dict[str, Any]:
    schema = GroundingReport.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:voltforge-ai:claim-grounding-report:v1"
    schema["title"] = "VoltForge claim grounding report v1"
    return schema


def checked_report_schema() -> dict[str, Any]:
    try:
        checked = json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundingContractError(
            "GROUNDING_SCHEMA_INVALID", "The checked grounding schema is unreadable."
        ) from error
    if checked != build_report_json_schema():
        raise GroundingContractError(
            "GROUNDING_SCHEMA_DRIFT", "The checked grounding schema has drifted."
        )
    return checked


def grounding_health() -> dict[str, Any]:
    try:
        policy = load_policy()
        checked_report_schema()
    except GroundingContractError as error:
        return {
            "ready": False,
            "status": "unavailable",
            "code": error.code,
            "policyId": POLICY_ID,
            "contractVersion": CONTRACT_VERSION,
        }
    return {
        "ready": True,
        "status": "ready",
        "code": "GROUNDING_READY",
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "contractVersion": CONTRACT_VERSION,
        "claimTypes": sorted(policy["claimTypes"]),
        "evidenceKinds": ["project", "deterministic", "local", "internet"],
    }
