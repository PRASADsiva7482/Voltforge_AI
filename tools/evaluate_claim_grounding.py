"""Generate or verify the content-free VFAI-024 claim-grounding receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from api.schemas import ChatRequest
from grounding import (
    POLICY_ID,
    POLICY_SHA256,
    GroundingReport,
    build_evidence_catalog,
    build_report_json_schema,
    checked_report_schema,
    enforce_response_grounding,
    grounding_health,
    load_policy,
)
from grounding.schema import REPORT_SCHEMA_PATH, canonical_json, sha256_json
from task_schema.adapters import (
    runtime_request_to_task_record,
    runtime_response_to_task_record,
)


DEFAULT_REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "claim-grounding-v1.json"
LOCAL_CITATION = "citation:local:mpu6050-v1"
PIN_CITATION = "citation:local:uno-pin-d3-v1"
API_CITATION = "citation:local:servo-api-v1"
WEB_CITATION = "citation:internet:voltforge-status-v1"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(
    citation_id: str,
    title: str,
    snippet: str,
    source_id: str,
    *,
    fact: Mapping[str, object] | None = None,
    url: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "citationId": citation_id,
        "title": title,
        "subject": title,
        "snippet": snippet,
        "sourceId": source_id,
        "sourceRevision": "1.0.0",
        "contentSha256": sha256_json({"title": title, "snippet": snippet}),
    }
    if fact is not None:
        item["fact"] = json.dumps(fact, sort_keys=True)
    if url is not None:
        item["sourceUrl"] = url
        item["retrievedAt"] = "2026-08-30T08:30:00Z"
    return item


def _event(
    name: str,
    authority: str,
    policy_id: str | None,
    results: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    evidence: dict[str, object] = {"results": list(results)}
    if policy_id:
        evidence["policyId"] = policy_id
    return {
        "name": name,
        "version": "1.0.0",
        "status": "complete",
        "authority": authority,
        "summary": f"Bounded {name} evidence.",
        "evidence": evidence,
    }


def _record(*events: Mapping[str, object]) -> dict[str, Any]:
    request = ChatRequest(
        message="Evaluate exact grounding.",
        projectId="evaluation-project",
        projectRevision="evaluation-revision-24",
        boardType="ARDUINO_UNO",
    )
    return runtime_request_to_task_record(
        request,
        project_payload={"boardType": "ARDUINO_UNO"},
        tool_events=list(events),
    )


def _fixture_record() -> dict[str, Any]:
    local = [
        _result(
            LOCAL_CITATION,
            "MPU6050",
            "MPU6050 operating voltage is 3.3V.",
            "vf-knowledge-mpu6050",
            fact={"property": "operating-voltage", "value": "3.3v"},
        ),
        _result(
            PIN_CITATION,
            "Arduino UNO pin map",
            "Pin D3 supports PWM on Arduino UNO.",
            "vf-knowledge-uno-pins",
            fact={"property": "pin-d3-capability", "value": "PWM"},
        ),
        _result(
            API_CITATION,
            "Servo library API",
            "The Servo library method attach() is available.",
            "vf-knowledge-servo-api",
            fact={"property": "api-method", "value": "attach()"},
        ),
    ]
    internet = [
        _result(
            WEB_CITATION,
            "VoltForge release status",
            "Currently, the VoltForge release status is stable.",
            "web-provider:status",
            url="https://example.com/voltforge-status",
        )
    ]
    return _record(
        _event("engineering-authority", "deterministic", None),
        _event(
            "curated-local-retrieval",
            "retrieved",
            "vfai022-curated-local-retrieval-v1",
            local,
        ),
        _event(
            "secure-internet-evidence",
            "retrieved",
            "vfai023-secure-internet-evidence-v1",
            internet,
        ),
    )


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evaluate() -> tuple[dict[str, bool], dict[str, int]]:
    policy = load_policy()
    record = _fixture_record()
    catalog = build_evidence_catalog(record)
    by_id = {item.citation.citationId: item.citation for item in catalog}
    supported = {
        "datasheet": enforce_response_grounding(
            "The MPU6050 operating voltage is 3.3V.", record
        ),
        "pin": enforce_response_grounding(
            "Pin D3 supports PWM on Arduino UNO.", record
        ),
        "api": enforce_response_grounding(
            "The Servo library method attach() is available.", record
        ),
        "web": enforce_response_grounding(
            "Currently, the VoltForge release status is stable.", record
        ),
    }
    empty_record = _record(_event("engineering-authority", "deterministic", None))
    unsupported = {
        key: enforce_response_grounding(text, empty_record)
        for key, text in {
            "datasheet": "The MPU6050 operating voltage is 3.3V.",
            "pin": "Pin D3 supports PWM.",
            "api": "The Servo library method attach() is available.",
            "web": "Currently, the VoltForge release status is stable.",
        }.items()
    }
    conflict_record = _record(
        _event(
            "curated-local-retrieval",
            "retrieved",
            "vfai022-curated-local-retrieval-v1",
            [
                _result(
                    LOCAL_CITATION,
                    "MPU6050",
                    "MPU6050 operating voltage is 3.3V.",
                    "vf-knowledge-mpu6050",
                    fact={"property": "operating-voltage", "value": "3.3v"},
                )
            ],
        ),
        _event(
            "secure-internet-evidence",
            "retrieved",
            "vfai023-secure-internet-evidence-v1",
            [
                _result(
                    "citation:internet:mpu6050-voltage",
                    "MPU6050",
                    "MPU6050 operating voltage is 5V.",
                    "web-provider:mpu6050",
                    fact={"property": "operating-voltage", "value": "5v"},
                    url="https://example.com/mpu6050",
                )
            ],
        ),
    )
    conflict = enforce_response_grounding(
        "The MPU6050 operating voltage is 3.3V.", conflict_record
    )
    forged = enforce_response_grounding(
        "The MPU6050 operating voltage is 3.3V. [citation:unknown:forged]",
        record,
    )
    no_claim = enforce_response_grounding("Bounded checks completed.", record)
    response_record = runtime_response_to_task_record(
        record,
        response_text=supported["pin"].text,
        metadata={
            "confidence": supported["pin"].report.maximumConfidence,
            "citations": [
                item.model_dump(mode="json") for item in supported["pin"].citations
            ],
            "grounding": supported["pin"].report.model_dump(mode="json"),
        },
    )
    ui_types = (WORKSPACE_ROOT / "Voltforge_UI" / "src" / "types" / "domain.ts").read_text(encoding="utf-8")
    ui_panel = (WORKSPACE_ROOT / "Voltforge_UI" / "src" / "features" / "ai" / "AiChatPanel.tsx").read_text(encoding="utf-8")
    backend_dto = (WORKSPACE_ROOT / "Voltforge_BL" / "src" / "main" / "java" / "in" / "voltforge" / "api" / "ai" / "dto" / "AiChatResponse.java").read_text(encoding="utf-8")
    kinds = {item.evidenceKind for item in by_id.values()}
    chat_source = (AI_ROOT / "api" / "chat.py").read_text(encoding="utf-8")
    checks = {
        "policyChecksumPinned": policy["policySha256"] == POLICY_SHA256,
        "checkedSchemaMatchesExecutableContract": checked_report_schema() == build_report_json_schema(),
        "groundingHealthReady": grounding_health()["ready"] is True,
        "projectEvidenceTyped": "project" in kinds,
        "deterministicEvidenceTyped": "deterministic" in kinds,
        "localEvidenceTyped": "local" in kinds,
        "internetEvidenceTyped": "internet" in kinds,
        "allCitationsResolveSelectedEvidence": all(item.citationId in by_id for result in supported.values() for item in result.citations),
        "allCitationsCarryExactPayloadHash": all(item.exactModelEvidence and len(item.modelPayloadSha256) == 64 for item in by_id.values()),
        "datasheetClaimGrounded": supported["datasheet"].report.status == "grounded" and LOCAL_CITATION in supported["datasheet"].text,
        "pinCapabilityGrounded": supported["pin"].report.status == "grounded" and PIN_CITATION in supported["pin"].text,
        "libraryApiGrounded": supported["api"].report.status == "grounded" and API_CITATION in supported["api"].text,
        "currentWebClaimGrounded": supported["web"].report.status == "grounded" and WEB_CITATION in supported["web"].text,
        "datasheetWithoutEvidenceRejected": unsupported["datasheet"].report.status == "uncertain",
        "pinWithoutEvidenceRejected": unsupported["pin"].report.status == "uncertain",
        "apiWithoutEvidenceRejected": unsupported["api"].report.status == "uncertain",
        "webWithoutEvidenceRejected": unsupported["web"].report.status == "uncertain",
        "unsupportedClaimsRemovedFromVisibleText": all("cannot verify" in item.text.casefold() for item in unsupported.values()),
        "unsupportedConfidenceCapped": all(item.report.maximumConfidence <= 0.6 for item in unsupported.values()),
        "conflictDetectedAcrossSources": conflict.report.status == "conflicted" and conflict.report.conflictedClaimCount == 1,
        "conflictForcesVisibleWording": "evidence conflicts" in conflict.text.casefold(),
        "conflictConfidenceCapped": conflict.report.maximumConfidence <= 0.5,
        "forgedCitationRemoved": "citation:unknown:forged" not in forged.text,
        "referenceCatalogAvailableWithoutClaim": no_claim.report.status == "no-high-risk-claims" and no_claim.report.citationCount == len(no_claim.citations),
        "taskRecordUsesExactGroundedRefs": response_record["output"]["assistantText"]["evidenceRefs"] == supported["pin"].report.usedEvidenceRefs,
        "taskRecordPreservesCitationIds": {item["citationId"] for item in response_record["output"]["citations"]} == {item.citationId for item in supported["pin"].citations},
        "internetEvidenceMarkedUntrusted": by_id[WEB_CITATION].untrustedContent is True,
        "retrievalCannotAuthorizeActions": policy["authority"]["retrievalCanAuthorizeStructuredActions"] is False,
        "rawContentNotStoredInReport": all(not item for item in (supported["pin"].report.rawPromptStored, supported["pin"].report.rawProjectContextStored, supported["pin"].report.rawModelOutputStored, supported["pin"].report.claimTextStoredInMetadata)),
        "backendPreservesStructuredGrounding": "Map<String, Object> grounding" in backend_dto and "List<Map<String, Object>> citations" in backend_dto,
        "uiTypesAllEvidenceClasses": all(f"'{kind}'" in ui_types for kind in ("project", "deterministic", "local", "internet")),
        "uiRendersEvidenceClassLabels": all(label in ui_panel for label in ("Project evidence", "Deterministic check", "Local knowledge", "Internet source")),
        "uiRendersConflictAndUncertainty": "Evidence conflict" in ui_panel and "Evidence unavailable" in ui_panel,
        "typedCitationEventPrecedesDeltas": chat_source.index('sse_event("citation"') < chat_source.index("for delta in _text_chunks"),
    }
    metrics = {
        "claimClassCount": len(policy["claimTypes"]),
        "evidenceClassCount": len(kinds),
        "supportedFixtureCount": len(supported),
        "unsupportedFixtureCount": len(unsupported),
        "conflictCount": len(conflict.report.conflicts),
    }
    GroundingReport.model_validate(supported["pin"].report.model_dump(mode="json"))
    return checks, metrics


def build_report() -> dict[str, Any]:
    checks, metrics = _evaluate()
    report = {
        "schemaVersion": 1,
        "reportId": "vfai024-claim-grounding-v1",
        "generatedOn": "2026-08-30",
        "policyId": POLICY_ID,
        "policySha256": POLICY_SHA256,
        "contractVersion": "1.0.0",
        "checkCount": len(checks),
        "passedCheckCount": sum(bool(value) for value in checks.values()),
        "checks": checks,
        "metrics": metrics,
        "networkFixtureOnly": True,
        "liveInternetRequiredForEvaluation": False,
        "rawPromptStored": False,
        "rawProjectContextStored": False,
        "rawModelOutputStored": False,
        "claimTextStored": False,
        "evaluatorSha256": _sha_file(Path(__file__).resolve()),
        "reportSchemaSha256": sha256_json(build_report_json_schema()),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def evaluate(path: Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    _write_json(build_report_json_schema(), REPORT_SCHEMA_PATH)
    report = build_report()
    if not all(report["checks"].values()):
        failed = [key for key, value in report["checks"].items() if not value]
        raise RuntimeError(f"VFAI-024 evaluation failed: {failed}")
    _write_json(report, path)
    return report


def verify(path: Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    checked = json.loads(path.read_text(encoding="utf-8"))
    generated = build_report()
    if checked != generated or checked.get("reportSha256") != _receipt_digest(checked):
        raise RuntimeError("VFAI-024 evaluation report is stale or invalid")
    if not all(checked["checks"].values()):
        raise RuntimeError("VFAI-024 evaluation report contains a failed check")
    return checked


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = evaluate(arguments.output.resolve()) if arguments.command == "evaluate" else verify(arguments.output.resolve())
    print(json.dumps({
        "ok": True,
        "command": arguments.command,
        "reportId": report["reportId"],
        "reportSha256": report["reportSha256"],
        "checks": report["checkCount"],
        "passed": report["passedCheckCount"],
        "liveInternetRequiredForEvaluation": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
