from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import socket

import pytest
import requests

import context_compiler
from api.copilot import prepare_grounded_context
from api.routes import engineering_check, validate_circuit
from api.schemas import ChatRequest, ValidateRequest
from engineering_tools import (
    engineering_tools_health,
    run_authoritative_engineering_checks,
    tool_events_from_report,
    validate_engineering_report,
)
from task_schema.adapters import runtime_request_to_task_record
from tools.evaluate_engineering_tools import evaluate, verify


def rich_request() -> ChatRequest:
    return ChatRequest(
        message="Review this project without overriding safety checks.",
        projectId="project-engineering-21",
        projectRevision="revision-21",
        boardType="ESP32_DEVKITC_V4_WROOM32E_N4",
        components=[
            {"id": "board-1", "type": "ESP32_DEVKITC_V4_WROOM32E_N4"},
            {"id": "led-1", "type": "LED"},
            {"id": "motor-1", "type": "DC_MOTOR"},
            {
                "id": "load-1",
                "type": "RESISTOR",
                "properties": {
                    "voltageV": "5 V",
                    "currentA": "20 mA",
                    "resistanceOhm": "100 ohm",
                    "powerW": "0.1 W",
                },
            },
            {"id": "unknown-1", "type": "UNLISTED_SENSOR_21"},
        ],
        wires=[
            {
                "id": "wire-flash",
                "fromNodeId": "board-1",
                "fromPinId": "GPIO6",
                "toNodeId": "motor-1",
                "toPinId": "positive",
            },
            {
                "id": "wire-led",
                "fromNodeId": "board-1",
                "fromPinId": "GPIO21",
                "toNodeId": "led-1",
                "toPinId": "anode",
            },
        ],
        code=(
            "#include <Arduino.h>\n"
            "#define LED_PIN 6\n"
            "#define MOTOR_PIN 6\n"
            "void setup() {}\n"
            "void loop() { digitalWrite(LED_PIN, HIGH); delay(1000); }"
        ),
        diagnostics=[
            {
                "message": "fatal error: PrivateMissing21.h: No such file or directory",
                "severity": "error",
                "compiled": False,
            }
        ],
        simulationState={
            "isSimulating": False,
            "solverConverged": False,
            "nodeVoltages": {"net-led": 3.2},
        },
    )


def all_findings(report) -> list:
    return [finding for run in report.toolRuns for finding in run.findings]


def test_all_seven_tools_emit_one_valid_evidence_bound_report() -> None:
    report = run_authoritative_engineering_checks(rich_request())
    validated = validate_engineering_report(report)

    assert validated == report.model_dump(mode="json")
    assert len(report.toolRuns) == 7
    assert {run.category for run in report.toolRuns} == {
        "circuit-netlist",
        "board-pin",
        "firmware-static",
        "compiler-feedback",
        "simulation",
        "units-calculation",
        "component-support",
    }
    rules = {finding.ruleId for finding in all_findings(report)}
    assert {
        "safety.inductive-suppression-unverified",
        "board-pin.flash-reserved-connected",
        "firmware.pin-definition-collision",
        "compiler.missing-header",
        "simulation.not-converged",
        "calculation.ohms-law-inconsistent",
        "component.unsupported-or-missing-evidence",
    } <= rules
    assert report.summary.status == "blocked"
    assert report.summary.calculations == 2


def test_every_finding_has_severity_affected_ids_evidence_and_reviewable_fix() -> None:
    report = run_authoritative_engineering_checks(rich_request())

    for run in report.toolRuns:
        evidence_ids = {item.evidenceId for item in run.evidence}
        for finding in run.findings:
            assert finding.severity in {"CRITICAL", "HIGH", "WARNING", "INFO", "UNKNOWN"}
            assert finding.affectedProjectIds
            assert set(finding.evidenceRefs) <= evidence_ids
            assert finding.fix.summary
            assert finding.fix.applicationMode == "proposal-only"
            assert finding.fix.requiresUserConfirmation is True
            if finding.blocking:
                assert finding.modelOverridePolicy == "prohibited"
            if finding.severity == "CRITICAL":
                assert finding.blocking is True
                assert finding.modelOverridePolicy == "prohibited"


def test_semantically_reordered_project_produces_identical_report() -> None:
    request = rich_request()
    reordered = request.model_copy(
        update={
            "components": list(reversed(request.components)),
            "wires": list(reversed(request.wires)),
            "diagnostics": list(reversed(request.diagnostics)),
        }
    )

    first = run_authoritative_engineering_checks(request)
    second = run_authoritative_engineering_checks(reordered)

    assert first.reportId == second.reportId
    assert first.model_dump_json() == second.model_dump_json()


def test_exact_variant_and_pin_capability_checks_fail_closed() -> None:
    unknown = run_authoritative_engineering_checks(
        ChatRequest(message="Check pins", boardType="ESP32")
    )
    unknown_finding = next(
        finding
        for finding in all_findings(unknown)
        if finding.ruleId == "board-pin.exact-variant-required"
    )
    assert unknown_finding.severity == "CRITICAL"
    assert unknown_finding.decision == "unknown"

    exact = run_authoritative_engineering_checks(rich_request())
    reserved = next(
        finding
        for finding in all_findings(exact)
        if finding.ruleId == "board-pin.flash-reserved-connected"
    )
    assert {"wire-flash", "board-1", "GPIO6"} <= set(
        reserved.affectedProjectIds
    )


def test_compiler_and_simulation_interpretation_store_hashes_not_raw_inputs() -> None:
    request = rich_request()
    report = run_authoritative_engineering_checks(request)
    encoded = report.model_dump_json()

    assert "PrivateMissing21.h" not in encoded
    compiler = next(run for run in report.toolRuns if run.category == "compiler-feedback")
    assert compiler.findings[0].ruleId == "compiler.missing-header"
    assert all(item.rawContentStored is False for item in compiler.evidence)
    simulation = next(run for run in report.toolRuns if run.category == "simulation")
    assert any(item.authority == "client-reported" for item in simulation.evidence)
    assert simulation.compiled is False


def test_nonfinite_simulation_and_bad_units_are_blocking() -> None:
    request = ChatRequest(
        message="Check numeric inputs",
        components=[
            {
                "id": "r-invalid",
                "type": "RESISTOR",
                "properties": {"resistanceOhm": "12 bananas"},
            }
        ],
        simulationState={
            "solverConverged": True,
            "nodeVoltages": {"broken": math.nan},
        },
    )
    report = run_authoritative_engineering_checks(request)
    rules = {finding.ruleId for finding in all_findings(report)}

    assert "simulation.nonfinite-result" in rules
    assert "units.invalid-or-incompatible-unit" in rules
    assert report.summary.status == "blocked"


def test_tool_event_index_preserves_authority_and_versions_in_task_evidence() -> None:
    request = rich_request()
    report = run_authoritative_engineering_checks(request)
    events = tool_events_from_report(report)
    index = events[0]

    assert index["name"] == "engineering-authority-index"
    assert index["version"] == "1.0.0"
    assert index["evidence"]["modelOverrideAllowed"] is False
    assert index["evidence"]["blockingFindingIds"]
    assert len(events) <= 8

    record = runtime_request_to_task_record(request, tool_events=events)
    typed = record["input"]["toolEvidence"][0]
    assert typed["toolVersion"] == "1.0.0"
    assert typed["authority"] == "deterministic"


def test_grounded_context_and_compatibility_route_use_authoritative_report() -> None:
    request = rich_request()
    grounded = prepare_grounded_context(request)

    assert grounded.engineering_report["policyId"] == (
        "vfai021-authoritative-engineering-tools-v1"
    )
    assert grounded.response_metadata["engineeringAuthorityActive"] is True
    assert grounded.response_metadata["engineeringAuthority"]["status"] == "blocked"
    assert grounded.proposal is None

    direct = engineering_check(request)
    assert direct["reportId"] == grounded.engineering_report["reportId"]
    compatible = validate_circuit(
        ValidateRequest(
            boardType=request.boardType,
            components=request.components,
            wires=request.wires,
            code=request.code,
            compilerDiagnostics=[item["message"] for item in request.diagnostics],
            simulationState=request.simulationState,
        )
    )
    assert compatible["isValid"] is False
    assert compatible["engineeringAuthority"]["criticalModelOverrideAllowed"] is False
    assert all(item["affectedProjectIds"] for item in compatible["issues"])


def test_engineering_checks_have_no_network_dependency(monkeypatch) -> None:
    def denied(*_args, **_kwargs):
        raise AssertionError("authoritative engineering checks attempted network access")

    monkeypatch.setattr(requests.sessions.Session, "request", denied)
    monkeypatch.setattr(socket, "create_connection", denied)

    report = run_authoritative_engineering_checks(rich_request())

    assert report.summary.findings > 0


def test_engineering_checks_survive_context_compiler_unavailability(monkeypatch) -> None:
    def unavailable(_request):
        raise context_compiler.ContextCompilerError(
            "TEST_CONTEXT_UNAVAILABLE", "context compiler unavailable"
        )

    monkeypatch.setattr(context_compiler, "resolve_project_revision", unavailable)

    report = run_authoritative_engineering_checks(rich_request())

    assert report.summary.toolRuns == 7
    assert report.sourceProjectRevision
    assert report.summary.status == "blocked"


def test_report_tampering_and_policy_health_are_fail_closed() -> None:
    report = run_authoritative_engineering_checks(rich_request()).model_dump(mode="json")
    tampered = deepcopy(report)
    tampered["toolRuns"][0]["findings"][0]["affectedProjectIds"] = []

    with pytest.raises(Exception):
        validate_engineering_report(tampered)

    health = engineering_tools_health()
    assert health["ready"] is True
    assert health["toolCount"] == 7
    assert health["criticalModelOverrideAllowed"] is False
    assert health["rawProjectContentStored"] is False


def test_committed_evaluation_is_reproducible(tmp_path: Path) -> None:
    receipt_path = tmp_path / "engineering-tools-receipt.json"

    generated = evaluate(receipt_path)
    checked = verify(receipt_path)

    assert generated == checked
    assert len(checked["checks"]) == 22
    assert all(checked["checks"].values())
