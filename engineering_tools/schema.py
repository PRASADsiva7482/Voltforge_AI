"""Strict typed contract for authoritative VoltForge engineering checks."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from task_schema.schema import TaskContractError, validate_json_schema_document


ENGINEERING_CONTRACT_VERSION = "1.0.0"
ENGINEERING_POLICY_ID = "vfai021-authoritative-engineering-tools-v1"
ENGINEERING_ROOT = Path(__file__).resolve().parent
ENGINEERING_POLICY_PATH = ENGINEERING_ROOT / "policy.v1.json"
ENGINEERING_SCHEMA_PATH = ENGINEERING_ROOT / "engineering-report.schema.json"
ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,199}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"


class EngineeringContractError(ValueError):
    """Precise engineering policy or report contract failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EngineeringEvidence(ContractModel):
    evidenceId: str = Field(pattern=ID_PATTERN)
    evidenceKind: Literal[
        "curated-knowledge",
        "project-state",
        "firmware-source",
        "compiler-diagnostic",
        "simulation-state",
        "deterministic-calculation",
    ]
    authority: Literal["deterministic", "client-reported"]
    sourceId: str = Field(pattern=ID_PATTERN)
    sourceRevision: str = Field(min_length=1, max_length=200)
    locator: str = Field(min_length=1, max_length=500)
    contentSha256: str = Field(pattern=SHA256_PATTERN)
    facts: dict[str, Any] = Field(default_factory=dict)
    rawContentStored: Literal[False] = False

    @model_validator(mode="after")
    def strict_json_facts(self) -> "EngineeringEvidence":
        try:
            json.dumps(self.facts, allow_nan=False, sort_keys=True, default=_json_default)
        except (TypeError, ValueError) as error:
            raise ValueError("evidence facts must be finite strict JSON") from error
        return self


class ReviewableFix(ContractModel):
    fixId: str = Field(pattern=ID_PATTERN)
    summary: str = Field(min_length=1, max_length=1_000)
    actionKind: Literal[
        "wire-suggestion",
        "component-addition",
        "component-removal",
        "value-change",
        "code-fix",
    ] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    applicationMode: Literal["proposal-only"] = "proposal-only"
    requiresUserConfirmation: Literal[True] = True
    machineApplicable: bool = False
    evidenceRefs: list[str] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def machine_action_contract(self) -> "ReviewableFix":
        if self.machineApplicable != (self.actionKind is not None):
            raise ValueError("machine-applicable fixes require exactly one typed action kind")
        return self


class EngineeringFinding(ContractModel):
    findingId: str = Field(pattern=ID_PATTERN)
    ruleId: str = Field(pattern=ID_PATTERN)
    ruleVersion: str = Field(min_length=1, max_length=80)
    category: Literal[
        "circuit-netlist",
        "board-pin",
        "firmware-static",
        "compiler-feedback",
        "simulation",
        "units-calculation",
        "component-support",
    ]
    severity: Literal["CRITICAL", "HIGH", "WARNING", "INFO", "UNKNOWN"]
    decision: Literal["violation", "warning", "unknown"]
    summary: str = Field(min_length=1, max_length=2_000)
    affectedProjectIds: list[str] = Field(min_length=1, max_length=100)
    evidenceRefs: list[str] = Field(min_length=1, max_length=100)
    fix: ReviewableFix
    blocking: bool
    modelOverridePolicy: Literal["prohibited", "not-applicable"]

    @model_validator(mode="after")
    def authority_contract(self) -> "EngineeringFinding":
        if len(self.affectedProjectIds) != len(set(self.affectedProjectIds)):
            raise ValueError("affected project IDs must be unique")
        if len(self.evidenceRefs) != len(set(self.evidenceRefs)):
            raise ValueError("finding evidence references must be unique")
        if not set(self.fix.evidenceRefs).issubset(self.evidenceRefs):
            raise ValueError("fix evidence must be a subset of finding evidence")
        if self.severity == "CRITICAL" and (
            not self.blocking or self.modelOverridePolicy != "prohibited"
        ):
            raise ValueError("critical findings must block and prohibit model override")
        if self.blocking and self.modelOverridePolicy != "prohibited":
            raise ValueError("blocking findings must prohibit model override")
        return self


class EngineeringCalculation(ContractModel):
    calculationId: str = Field(pattern=ID_PATTERN)
    calculationKind: Literal["ohms-law", "electrical-power"]
    affectedProjectIds: list[str] = Field(min_length=1, max_length=100)
    inputs: dict[str, str]
    result: dict[str, str]
    evidenceRefs: list[str] = Field(min_length=1, max_length=100)
    exactDecimal: Literal[True] = True


class ApprovedAction(ContractModel):
    actionKind: Literal[
        "wire-suggestion",
        "component-addition",
        "component-removal",
        "value-change",
        "code-fix",
    ]
    payload: dict[str, Any]


class EngineeringToolRun(ContractModel):
    toolId: str = Field(pattern=ID_PATTERN)
    toolVersion: str = Field(min_length=1, max_length=80)
    category: Literal[
        "circuit-netlist",
        "board-pin",
        "firmware-static",
        "compiler-feedback",
        "simulation",
        "units-calculation",
        "component-support",
    ]
    status: Literal["complete", "not-applicable", "unavailable"]
    summary: str = Field(min_length=1, max_length=2_000)
    findings: list[EngineeringFinding] = Field(default_factory=list, max_length=50)
    evidence: list[EngineeringEvidence] = Field(default_factory=list, max_length=100)
    calculations: list[EngineeringCalculation] = Field(default_factory=list, max_length=100)
    approvedActions: list[ApprovedAction] = Field(default_factory=list, max_length=50)
    compiled: bool = False
    rawContentStored: Literal[False] = False

    @model_validator(mode="after")
    def resolved_evidence_contract(self) -> "EngineeringToolRun":
        evidence_ids = [item.evidenceId for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("tool evidence IDs must be unique")
        available = set(evidence_ids)
        finding_ids = [item.findingId for item in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("tool finding IDs must be unique")
        for finding in self.findings:
            if not set(finding.evidenceRefs).issubset(available):
                raise ValueError(f"finding {finding.findingId} has unresolved evidence")
        for calculation in self.calculations:
            if not set(calculation.evidenceRefs).issubset(available):
                raise ValueError(
                    f"calculation {calculation.calculationId} has unresolved evidence"
                )
        expected_actions = [
            ApprovedAction(actionKind=finding.fix.actionKind, payload=finding.fix.payload)
            for finding in self.findings
            if finding.fix.machineApplicable and finding.fix.actionKind is not None
        ]
        if self.approvedActions != expected_actions:
            raise ValueError("approved actions must exactly match machine-applicable fixes")
        if self.status == "not-applicable" and (
            self.findings or self.evidence or self.calculations or self.approvedActions
        ):
            raise ValueError("not-applicable tools cannot claim evidence or results")
        return self


class EngineeringReportSummary(ContractModel):
    toolRuns: int = Field(ge=0)
    applicableToolRuns: int = Field(ge=0)
    findings: int = Field(ge=0)
    blockingFindings: int = Field(ge=0)
    criticalFindings: int = Field(ge=0)
    unknownFindings: int = Field(ge=0)
    calculations: int = Field(ge=0)
    status: Literal["pass", "review-required", "blocked"]


class EngineeringAuthorityReport(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = ENGINEERING_CONTRACT_VERSION
    policyId: Literal["vfai021-authoritative-engineering-tools-v1"] = (
        ENGINEERING_POLICY_ID
    )
    policySha256: str = Field(pattern=SHA256_PATTERN)
    reportId: str = Field(pattern=r"^vf-engineering-report-v1-[a-f0-9]{24}$")
    sourceProjectRevision: str = Field(min_length=1, max_length=200)
    revisionSource: Literal["client", "derived-snapshot", "synthetic", "evaluation"]
    boardType: str = Field(min_length=1, max_length=80)
    toolRuns: list[EngineeringToolRun] = Field(min_length=7, max_length=7)
    blockingFindingIds: list[str] = Field(default_factory=list, max_length=350)
    summary: EngineeringReportSummary
    rawProjectContentStored: Literal[False] = False

    @model_validator(mode="after")
    def aggregate_contract(self) -> "EngineeringAuthorityReport":
        tool_ids = [item.toolId for item in self.toolRuns]
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("engineering tool IDs must be unique")
        findings = [finding for run in self.toolRuns for finding in run.findings]
        finding_ids = [item.findingId for item in findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("report finding IDs must be unique")
        blocking = sorted(item.findingId for item in findings if item.blocking)
        if self.blockingFindingIds != blocking:
            raise ValueError("blocking finding index does not match report findings")
        expected = EngineeringReportSummary(
            toolRuns=len(self.toolRuns),
            applicableToolRuns=sum(item.status != "not-applicable" for item in self.toolRuns),
            findings=len(findings),
            blockingFindings=len(blocking),
            criticalFindings=sum(item.severity == "CRITICAL" for item in findings),
            unknownFindings=sum(item.decision == "unknown" for item in findings),
            calculations=sum(len(item.calculations) for item in self.toolRuns),
            status=(
                "blocked"
                if blocking
                else "review-required"
                if findings
                else "pass"
            ),
        )
        if self.summary != expected:
            raise ValueError("engineering report summary does not match tool results")
        return self


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite numbers are forbidden")
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@lru_cache(maxsize=1)
def load_engineering_policy() -> dict[str, Any]:
    try:
        policy = json.loads(ENGINEERING_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EngineeringContractError(
            "ENGINEERING_POLICY_UNAVAILABLE", "Engineering policy is unavailable."
        ) from error
    if not isinstance(policy, dict):
        raise EngineeringContractError(
            "ENGINEERING_POLICY_INVALID", "Engineering policy must be an object."
        )
    expected = policy.get("policySha256")
    unsigned = dict(policy)
    unsigned.pop("policySha256", None)
    actual = sha256_json(unsigned)
    if not isinstance(expected, str) or not re.fullmatch(SHA256_PATTERN, expected):
        raise EngineeringContractError(
            "ENGINEERING_POLICY_INVALID", "Engineering policy checksum is malformed."
        )
    if actual != expected:
        raise EngineeringContractError(
            "ENGINEERING_POLICY_CHECKSUM_MISMATCH",
            "Engineering policy checksum does not match its content.",
        )
    if (
        policy.get("schemaVersion") != 1
        or policy.get("policyId") != ENGINEERING_POLICY_ID
        or policy.get("contractVersion") != ENGINEERING_CONTRACT_VERSION
        or len(policy.get("tools") or []) != 7
    ):
        raise EngineeringContractError(
            "ENGINEERING_POLICY_UNSUPPORTED", "Engineering policy contract is unsupported."
        )
    return policy


def validate_engineering_report(
    value: Mapping[str, Any] | EngineeringAuthorityReport,
) -> dict[str, Any]:
    candidate = (
        value.model_dump(mode="json")
        if isinstance(value, EngineeringAuthorityReport)
        else dict(value)
    )
    try:
        validate_json_schema_document(candidate, checked_engineering_schema())
    except TaskContractError as error:
        raise EngineeringContractError(
            "ENGINEERING_REPORT_SCHEMA_INVALID", error.message
        ) from error
    try:
        model = EngineeringAuthorityReport.model_validate(candidate)
    except Exception as error:
        raise EngineeringContractError(
            "ENGINEERING_REPORT_INVALID", str(error)
        ) from error
    return model.model_dump(mode="json")


def build_engineering_json_schema() -> dict[str, Any]:
    schema = EngineeringAuthorityReport.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:voltforge-ai:authoritative-engineering-report:v1"
    schema["title"] = "VoltForge authoritative engineering report v1"
    return schema


@lru_cache(maxsize=1)
def checked_engineering_schema() -> dict[str, Any]:
    try:
        schema = json.loads(ENGINEERING_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EngineeringContractError(
            "ENGINEERING_SCHEMA_UNAVAILABLE",
            "Checked-in engineering report schema is unavailable.",
        ) from error
    if schema != build_engineering_json_schema():
        raise EngineeringContractError(
            "ENGINEERING_SCHEMA_STALE",
            "Checked-in engineering report schema does not match the executable contract.",
        )
    return schema
