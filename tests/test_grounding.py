from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.chat import stream_chat_sse
from api.schemas import ChatRequest
from grounding import (
    GroundingReport,
    NeuralGroundingError,
    build_evidence_catalog,
    build_report_json_schema,
    checked_report_schema,
    classify_claim,
    enforce_response_grounding,
    load_policy,
    validate_neural_claim,
)
from grounding.schema import sha256_json
from task_schema.adapters import (
    runtime_request_to_task_record,
    runtime_response_to_task_record,
)
from tools.evaluate_claim_grounding import evaluate, verify


LOCAL_CITATION = "citation:local:mpu6050-v1"
PIN_CITATION = "citation:local:uno-pin-d3-v1"
API_CITATION = "citation:local:servo-api-v1"
WEB_CITATION = "citation:internet:voltforge-status-v1"


def _result(
    citation_id: str,
    title: str,
    snippet: str,
    *,
    source_id: str,
    fact: dict[str, object] | None = None,
    url: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "citationId": citation_id,
        "title": title,
        "subject": title,
        "snippet": snippet,
        "sourceId": source_id,
        "sourceRevision": "1.0.0",
        "contentSha256": sha256_json({"title": title, "snippet": snippet}),
    }
    if fact is not None:
        result["fact"] = json.dumps(fact, sort_keys=True)
    if url is not None:
        result["sourceUrl"] = url
        result["retrievedAt"] = "2026-08-30T08:30:00Z"
    return result


def _local_event(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "name": "curated-local-retrieval",
        "version": "1.0.0",
        "status": "complete",
        "authority": "retrieved",
        "summary": "Selected curated local evidence.",
        "evidence": {
            "policyId": "vfai022-curated-local-retrieval-v1",
            "results": results,
        },
    }


def _internet_event(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "name": "secure-internet-evidence",
        "version": "1.0.0",
        "status": "complete",
        "authority": "retrieved",
        "summary": "Selected untrusted internet evidence.",
        "evidence": {
            "policyId": "vfai023-secure-internet-evidence-v1",
            "results": results,
        },
    }


def _deterministic_event() -> dict[str, object]:
    return {
        "name": "engineering-authority",
        "version": "1.0.0",
        "status": "complete",
        "authority": "deterministic",
        "summary": "Deterministic circuit checks passed for the bounded project.",
        "evidence": {"status": "pass", "findingCount": 0},
    }


def _record(*events: dict[str, object]) -> dict[str, object]:
    request = ChatRequest(
        message="Check the bounded electronics facts.",
        projectId="grounding-project",
        projectRevision="revision-24",
        boardType="ARDUINO_UNO",
        components=[{"id": "mcu1", "type": "arduino-uno"}],
    )
    return runtime_request_to_task_record(
        request,
        project_payload={"boardType": "ARDUINO_UNO", "componentCount": 1},
        tool_events=list(events),
    )


def _all_evidence_record() -> dict[str, object]:
    return _record(
        _deterministic_event(),
        _local_event(
            [
                _result(
                    LOCAL_CITATION,
                    "MPU6050",
                    "MPU6050 operating voltage is 3.3V.",
                    source_id="vf-knowledge-mpu6050",
                    fact={"property": "operating-voltage", "value": "3.3v"},
                ),
                _result(
                    PIN_CITATION,
                    "Arduino UNO pin map",
                    "Pin D3 supports PWM on Arduino UNO.",
                    source_id="vf-knowledge-uno-pins",
                    fact={"property": "pin-d3-capability", "value": "PWM"},
                ),
                _result(
                    API_CITATION,
                    "Servo library API",
                    "The Servo library method attach() is available.",
                    source_id="vf-knowledge-servo-api",
                    fact={"property": "api-method", "value": "attach()"},
                ),
            ]
        ),
        _internet_event(
            [
                _result(
                    WEB_CITATION,
                    "VoltForge release status",
                    "Currently, the VoltForge release status is stable.",
                    source_id="web-provider:status",
                    url="https://example.com/voltforge-status",
                )
            ]
        ),
    )


def test_policy_checksum_and_required_claim_classes_are_pinned() -> None:
    policy = load_policy()
    unsigned = dict(policy)
    declared = unsigned.pop("policySha256")

    assert declared == sha256_json(unsigned)
    assert set(policy["claimTypes"]) == {
        "datasheet-rating",
        "pin-capability",
        "library-api",
        "current-web",
    }
    assert policy["authority"]["internetIsUntrusted"] is True
    assert policy["authority"]["retrievalCanAuthorizeStructuredActions"] is False


def test_checked_schema_matches_executable_contract() -> None:
    assert checked_report_schema() == build_report_json_schema()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The MPU6050 operating voltage is 3.3V.", "datasheet-rating"),
        ("Pin D3 supports PWM.", "pin-capability"),
        ("The Servo library method attach() is available.", "library-api"),
        ("Currently, the VoltForge release status is stable.", "current-web"),
    ],
)
def test_required_claim_classes_are_detected(text: str, expected: str) -> None:
    assert expected in classify_claim(text)


def test_electrical_current_is_not_misclassified_as_temporal_web_claim() -> None:
    kinds = classify_claim("Do not bypass the current-limiting resistor.")
    assert "current-web" not in kinds


def test_catalog_contains_four_visibly_distinct_exact_evidence_classes() -> None:
    record = _all_evidence_record()
    catalog = build_evidence_catalog(record)

    assert {item.citation.evidenceKind for item in catalog} == {
        "project",
        "deterministic",
        "local",
        "internet",
    }
    for entry in catalog:
        assert entry.citation.exactModelEvidence is True
        assert len(entry.citation.modelPayloadSha256) == 64
        assert len(entry.citation.contentSha256) == 64
    internet = next(
        item.citation for item in catalog if item.citation.citationId == WEB_CITATION
    )
    assert internet.untrustedContent is True
    assert internet.url == "https://example.com/voltforge-status"


def test_only_results_in_selected_model_evidence_can_be_cited() -> None:
    record = _record(
        _local_event(
            [
                _result(
                    LOCAL_CITATION,
                    "MPU6050",
                    "MPU6050 operating voltage is 3.3V.",
                    source_id="vf-knowledge-mpu6050",
                )
            ]
        )
    )
    unselected = {
        "citationId": "citation:local:unselected-result",
        "title": "Unselected source",
        "url": "https://example.com/unselected",
    }
    catalog = build_evidence_catalog(record, [unselected])

    assert "citation:local:unselected-result" not in {
        item.citation.citationId for item in catalog
    }


@pytest.mark.parametrize(
    ("claim", "citation_id", "claim_type"),
    [
        ("The MPU6050 operating voltage is 3.3V.", LOCAL_CITATION, "datasheet-rating"),
        ("Pin D3 supports PWM on Arduino UNO.", PIN_CITATION, "pin-capability"),
        ("The Servo library method attach() is available.", API_CITATION, "library-api"),
        (
            "Currently, the VoltForge release status is stable.",
            WEB_CITATION,
            "current-web",
        ),
    ],
)
def test_supported_high_risk_claims_bind_exact_citation_ids(
    claim: str, citation_id: str, claim_type: str
) -> None:
    result = enforce_response_grounding(claim, _all_evidence_record())
    grounded_claim = result.report.claims[0]

    assert result.report.status == "grounded"
    assert grounded_claim.supportStatus == "supported"
    assert claim_type in grounded_claim.claimTypes
    assert citation_id in grounded_claim.citationIds
    assert f"[{citation_id}]" in result.text
    assert grounded_claim.evidenceRefs == result.report.usedEvidenceRefs


@pytest.mark.parametrize(
    ("claim", "reason_code"),
    [
        ("The MPU6050 operating voltage is 3.3V.", "DATASHEET_EVIDENCE_REQUIRED"),
        ("Pin D3 supports PWM.", "PIN_CAPABILITY_EVIDENCE_REQUIRED"),
        ("The Servo library method attach() is available.", "LIBRARY_API_EVIDENCE_REQUIRED"),
        ("Currently, the VoltForge release status is stable.", "CURRENT_WEB_EVIDENCE_REQUIRED"),
    ],
)
def test_unsupported_high_risk_claims_are_replaced_with_visible_uncertainty(
    claim: str, reason_code: str
) -> None:
    result = enforce_response_grounding(claim, _record(_deterministic_event()))

    assert result.report.status == "uncertain"
    assert result.report.claims[0].supportStatus == "unsupported"
    assert result.report.claims[0].visibleTextTransformed is True
    assert reason_code in result.report.uncertainty.missingEvidence
    assert claim not in result.text
    assert "cannot verify" in result.text.casefold()
    assert result.report.maximumConfidence <= 0.6


def test_current_claim_cannot_use_local_evidence_in_place_of_internet() -> None:
    result = enforce_response_grounding(
        "Currently, the VoltForge release status is stable.",
        _record(
            _local_event(
                [
                    _result(
                        "citation:local:release-status",
                        "VoltForge release status",
                        "Currently, the VoltForge release status is stable.",
                        source_id="vf-knowledge-release-status",
                    )
                ]
            )
        ),
    )

    assert result.report.status == "uncertain"
    assert result.report.claims[0].reasonCode == "CURRENT_WEB_EVIDENCE_REQUIRED"


def test_pin_capability_cannot_use_untrusted_internet_as_authority() -> None:
    result = enforce_response_grounding(
        "Pin D3 supports PWM on Arduino UNO.",
        _record(
            _internet_event(
                [
                    _result(
                        "citation:internet:uno-pin-d3",
                        "Arduino UNO pin map",
                        "Pin D3 supports PWM on Arduino UNO.",
                        source_id="web-provider:pin-map",
                        url="https://example.com/uno-pin-map",
                    )
                ]
            )
        ),
    )

    assert result.report.status == "uncertain"
    assert result.report.claims[0].reasonCode == "PIN_CAPABILITY_EVIDENCE_REQUIRED"


def test_conflicting_sources_force_conflict_wording_and_confidence_cap() -> None:
    record = _record(
        _local_event(
            [
                _result(
                    LOCAL_CITATION,
                    "MPU6050",
                    "MPU6050 operating voltage is 3.3V.",
                    source_id="vf-knowledge-mpu6050",
                    fact={"property": "operating-voltage", "value": "3.3v"},
                )
            ]
        ),
        _internet_event(
            [
                _result(
                    "citation:internet:mpu6050-voltage",
                    "MPU6050",
                    "MPU6050 operating voltage is 5V.",
                    source_id="web-provider:mpu6050",
                    fact={"property": "operating-voltage", "value": "5v"},
                    url="https://example.com/mpu6050",
                )
            ]
        ),
    )
    result = enforce_response_grounding(
        "The MPU6050 operating voltage is 3.3V.", record
    )

    assert result.report.status == "conflicted"
    assert result.report.conflictedClaimCount == 1
    assert set(result.report.conflicts[0].citationIds) == {
        LOCAL_CITATION,
        "citation:internet:mpu6050-voltage",
    }
    assert "evidence conflicts" in result.text.casefold()
    assert "operating voltage is 3.3V" not in result.text
    assert result.report.maximumConfidence <= 0.5


def test_unknown_citation_marker_is_never_preserved() -> None:
    result = enforce_response_grounding(
        "The MPU6050 operating voltage is 3.3V. [citation:unknown:forged]",
        _all_evidence_record(),
    )

    assert "citation:unknown:forged" not in result.text
    assert f"[{LOCAL_CITATION}]" in result.text


def test_existing_visible_uncertainty_remains_uncertain_and_caps_confidence() -> None:
    result = enforce_response_grounding(
        "I cannot verify the MPU6050 operating voltage is 3.3V.",
        _record(),
    )

    assert result.report.status == "uncertain"
    assert result.report.claims[0].reasonCode == "VISIBLE_UNCERTAINTY_PRESENT"
    assert result.report.maximumConfidence <= 0.6


def test_claim_limit_never_allows_overflow_claims_to_bypass_the_gate() -> None:
    text = "\n".join(
        f"Device {index} operating voltage is 3.3V." for index in range(30)
    )
    result = enforce_response_grounding(text, _record())

    assert result.report.status == "uncertain"
    assert result.report.claimCount == 24
    assert result.report.claims[-1].reasonCode == "CLAIM_LIMIT_EXCEEDED"
    assert "operating voltage is 3.3V" not in result.text


def test_non_high_risk_response_exposes_reference_catalog_without_claim_binding() -> None:
    result = enforce_response_grounding(
        "The bounded engineering checks completed.", _all_evidence_record()
    )

    assert result.report.status == "no-high-risk-claims"
    assert result.report.usedEvidenceRefs == []
    assert result.report.citationCount == len(result.citations)
    assert {item.evidenceKind for item in result.citations} == {
        "project",
        "deterministic",
        "local",
        "internet",
    }


def test_task_response_binds_only_evidence_used_by_supported_claims() -> None:
    request_record = _all_evidence_record()
    grounded = enforce_response_grounding(
        "Pin D3 supports PWM on Arduino UNO.", request_record
    )
    response_record = runtime_response_to_task_record(
        request_record,
        response_text=grounded.text,
        metadata={
            "confidence": grounded.report.maximumConfidence,
            "citations": [item.model_dump(mode="json") for item in grounded.citations],
            "grounding": grounded.report.model_dump(mode="json"),
        },
    )

    assert response_record["output"]["assistantText"]["evidenceRefs"] == (
        grounded.report.usedEvidenceRefs
    )
    assert {item["citationId"] for item in response_record["output"]["citations"]} == {
        item.citationId for item in grounded.citations
    }


def test_grounding_report_rejects_unknown_fields() -> None:
    result = enforce_response_grounding("A bounded summary.", _record())
    payload = result.report.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        GroundingReport.model_validate(payload)


def test_grounding_evaluation_receipt_is_reproducible(tmp_path: Path) -> None:
    report_path = tmp_path / "claim-grounding-v1.json"

    generated = evaluate(report_path)
    checked = verify(report_path)

    assert generated == checked
    assert generated["checkCount"] == 34
    assert generated["passedCheckCount"] == 34
    assert generated["liveInternetRequiredForEvaluation"] is False


@pytest.mark.parametrize(
    ("claim", "events", "expected_code"),
    [
        (
            "The MPU6050 operating voltage is 3.3V.",
            [_deterministic_event()],
            "QG_DATASHEET_EVIDENCE_REQUIRED",
        ),
        (
            "Pin D3 supports PWM on Arduino UNO.",
            [_internet_event([_result("citation:internet:pin", "Arduino UNO pin map", "Pin D3 supports PWM on Arduino UNO.", source_id="web-provider:pins", url="https://example.com/pins")])],
            "QG_PIN_CAPABILITY_EVIDENCE_REQUIRED",
        ),
        (
            "The Servo library method attach() is available.",
            [_deterministic_event()],
            "QG_LIBRARY_API_EVIDENCE_REQUIRED",
        ),
        (
            "Currently, the VoltForge release status is stable.",
            [_local_event([_result("citation:local:status", "VoltForge release status", "Currently, the VoltForge release status is stable.", source_id="vf-knowledge-status")])],
            "QG_CURRENT_WEB_EVIDENCE_REQUIRED",
        ),
    ],
)
def test_neural_quality_gate_rejects_wrong_source_class(
    claim: str, events: list[dict[str, object]], expected_code: str
) -> None:
    record = _record(*events)
    evidence = {
        item["evidenceId"]: item for item in record["input"]["toolEvidence"]
    }

    with pytest.raises(NeuralGroundingError) as caught:
        validate_neural_claim(claim, list(evidence), evidence)
    assert caught.value.code == expected_code


def test_sse_emits_typed_citations_before_response_deltas() -> None:
    events = asyncio.run(
        _collect_stream(
            ChatRequest(
                message="What MCU architecture and logic voltage does Arduino UNO R3 use?",
                boardType="ARDUINO_UNO",
                projectRevision="grounding-stream-revision",
            )
        )
    )
    event_names = [
        line.removeprefix("event: ")
        for event in events
        for line in event.splitlines()
        if line.startswith("event: ")
    ]

    assert "citation" in event_names
    assert event_names.index("citation") < event_names.index("delta")


async def _collect_stream(request: ChatRequest) -> list[str]:
    return [item async for item in stream_chat_sse(request)]
