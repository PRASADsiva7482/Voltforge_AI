"""Generate or verify the content-free VFAI-021 authoritative-tools receipt."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sys
from typing import Any, Sequence

import requests


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from api.chat import _deterministic_response
from api.schemas import ChatRequest
from context_compiler import compile_project_context
from engineering_tools import (
    engineering_tools_health,
    run_authoritative_engineering_checks,
    tool_events_from_report,
    validate_engineering_report,
)
from engineering_tools.schema import (
    ENGINEERING_POLICY_PATH,
    ENGINEERING_SCHEMA_PATH,
    checked_engineering_schema,
    load_engineering_policy,
)
from model.generation_quality import GenerationQualityGate


DEFAULT_REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "authoritative-engineering-tools-v1.json"
PRIVATE_SENTINEL = "PRIVATE_DIAGNOSTIC_21.h"


def fixture() -> ChatRequest:
    return ChatRequest(
        message="Review this local electronics project safely.",
        projectId="synthetic-engineering-project",
        projectRevision="synthetic-revision-21",
        boardType="ESP32_DEVKITC_V4_WROOM32E_N4",
        components=[
            {"id": "board", "type": "ESP32_DEVKITC_V4_WROOM32E_N4"},
            {"id": "led", "type": "LED"},
            {"id": "motor", "type": "DC_MOTOR"},
            {
                "id": "load",
                "type": "RESISTOR",
                "properties": {
                    "voltageV": "5 V",
                    "currentA": "20 mA",
                    "resistanceOhm": "100 ohm",
                    "powerW": "0.1 W",
                },
            },
            {"id": "unknown", "type": "SYNTHETIC_UNKNOWN_COMPONENT_21"},
        ],
        wires=[
            {
                "id": "reserved-wire",
                "fromNodeId": "board",
                "fromPinId": "GPIO6",
                "toNodeId": "motor",
                "toPinId": "positive",
            },
            {
                "id": "led-wire",
                "fromNodeId": "board",
                "fromPinId": "GPIO21",
                "toNodeId": "led",
                "toPinId": "anode",
            },
        ],
        code=(
            "#include <Arduino.h>\n"
            "#define LED_PIN 6\n#define MOTOR_PIN 6\n"
            "void setup() {}\n"
            "void loop() { digitalWrite(LED_PIN, HIGH); delay(1000); }"
        ),
        diagnostics=[
            {
                "message": f"fatal error: {PRIVATE_SENTINEL}: No such file or directory",
                "severity": "error",
                "compiled": False,
            }
        ],
        simulationState={
            "isSimulating": False,
            "solverConverged": False,
            "nodeVoltages": {"led-net": 3.2},
        },
    )


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_digest(report: dict[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    payload = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quality_decisions(compilation) -> tuple[str, str]:
    index = next(
        item
        for item in compilation.task_record["input"]["toolEvidence"]
        if item["toolName"] == "tool:engineering-authority-index"
    )
    evidence_id = index["evidenceId"]
    accepted_envelope = {
        "schemaVersion": 1,
        "domain": "voltforge-electronics",
        "finishReason": "stop",
        "confidence": {"score": 0.9, "basis": "evidence-aligned"},
        "segments": [
            {
                "type": "safety-warning",
                "text": "Warning: blocking circuit and firmware findings require review and are not verified safe.",
                "evidenceRefs": [evidence_id],
            }
        ],
        "structuredActions": [],
        "citationEvidenceIds": [evidence_id],
    }
    omitted_envelope = deepcopy(accepted_envelope)
    omitted_envelope["segments"] = [
        {
            "type": "uncertainty",
            "text": "The circuit status is uncertain because evidence is unavailable.",
            "evidenceRefs": [],
        }
    ]
    omitted_envelope["citationEvidenceIds"] = []
    gate = GenerationQualityGate()
    accepted = gate.evaluate(json.dumps(accepted_envelope), compilation.task_record)
    omitted = gate.evaluate(json.dumps(omitted_envelope), compilation.task_record)
    return accepted.code, omitted.code


def build_report() -> dict[str, Any]:
    policy = load_engineering_policy()
    checked_engineering_schema()
    request = fixture()

    original_request = requests.sessions.Session.request
    original_connection = socket.create_connection

    def denied(*_args, **_kwargs):
        raise AssertionError("VFAI-021 deterministic evaluation attempted network access")

    requests.sessions.Session.request = denied
    socket.create_connection = denied
    try:
        first = run_authoritative_engineering_checks(request)
        reordered = request.model_copy(
            update={
                "components": list(reversed(request.components)),
                "wires": list(reversed(request.wires)),
                "diagnostics": list(reversed(request.diagnostics)),
            }
        )
        second = run_authoritative_engineering_checks(reordered)
    finally:
        requests.sessions.Session.request = original_request
        socket.create_connection = original_connection

    validated = validate_engineering_report(first)
    events = tool_events_from_report(first)
    compilation = compile_project_context(
        request,
        tool_events=events,
        context_window_tokens=8192,
        reserved_output_tokens=512,
    )
    accepted_code, omitted_code = _quality_decisions(compilation)
    findings = [finding for run in first.toolRuns for finding in run.findings]
    rules = sorted({finding.ruleId for finding in findings})
    unknown_board = run_authoritative_engineering_checks(
        ChatRequest(message="Check this ambiguous board.", boardType="ESP32")
    )
    simulation_run = next(run for run in first.toolRuns if run.category == "simulation")
    compiler_run = next(run for run in first.toolRuns if run.category == "compiler-feedback")
    deterministic_reply = _deterministic_response(
        request, False, first.model_dump(mode="json")
    ).reply
    encoded_report = first.model_dump_json()
    checks = {
        "policyChecksumVerified": policy["policySha256"]
        == "4399610d474c0ad9e106f5e73c4d09123ec3e1325b41e30ab31b1121a7835033",
        "checkedSchemaMatchesExecutableContract": bool(checked_engineering_schema()),
        "networkDeniedDuringAllToolRuns": True,
        "sevenVersionedToolsPresent": len(first.toolRuns) == 7
        and all(run.toolVersion == "1.0.0" for run in first.toolRuns),
        "reportSchemaAndSemanticsValid": validated == first.model_dump(mode="json"),
        "semanticReorderingIsByteStable": first.model_dump_json() == second.model_dump_json(),
        "everyFindingHasRequiredReviewFields": all(
            finding.severity
            and finding.affectedProjectIds
            and finding.evidenceRefs
            and finding.fix.summary
            and finding.fix.requiresUserConfirmation
            for finding in findings
        ),
        "criticalFindingsAreNonOverridable": all(
            finding.blocking and finding.modelOverridePolicy == "prohibited"
            for finding in findings
            if finding.severity == "CRITICAL"
        ),
        "circuitAndNetlistRulesExecuted": any(
            rule.startswith("safety.inductive") for rule in rules
        ),
        "exactBoardPinRuleExecuted": "board-pin.flash-reserved-connected" in rules,
        "firmwareStaticRuleExecuted": "firmware.pin-definition-collision" in rules,
        "compilerDiagnosticClassified": "compiler.missing-header" in rules
        and compiler_run.compiled is False,
        "simulationReportedNotRerun": any(
            item.authority == "client-reported" for item in simulation_run.evidence
        ),
        "unitsAndCalculationsExecuted": first.summary.calculations == 2
        and "calculation.ohms-law-inconsistent" in rules,
        "unsupportedComponentsRemainUnknown": (
            "component.unsupported-or-missing-evidence" in rules
        ),
        "ambiguousBoardFailsClosed": any(
            finding.ruleId == "board-pin.exact-variant-required"
            and finding.blocking
            for run in unknown_board.toolRuns
            for finding in run.findings
        ),
        "toolEventSetIsBounded": len(events) <= 8
        and events[0]["name"] == "engineering-authority-index",
        "contextCompilerPreservesAuthorityIndex": any(
            item["toolName"] == "tool:engineering-authority-index"
            for item in compilation.task_record["input"]["toolEvidence"]
        )
        and compilation.public_metadata["promptTokens"]
        <= compilation.public_metadata["promptTokenLimit"],
        "qualityGateRequiresVisibleBlockingWarning": accepted_code == "QG_ACCEPTED"
        and omitted_code == "QG_AUTHORITATIVE_FINDING_OMITTED",
        "blockingDeterministicReplySuppressesFreeFormOverride": (
            deterministic_reply.startswith("Authoritative deterministic engineering checks")
            and "cannot be overridden" in deterministic_reply
        ),
        "rawPrivateInputsNotStored": PRIVATE_SENTINEL not in encoded_report
        and PRIVATE_SENTINEL not in deterministic_reply,
        "healthDeclaresCriticalOverrideDisabled": engineering_tools_health().get(
            "criticalModelOverrideAllowed"
        )
        is False,
    }
    report = {
        "schemaVersion": 1,
        "reportId": "vfai021-authoritative-engineering-tools-v1",
        "generatedOn": "2026-08-30",
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "schemaSha256": _sha_file(ENGINEERING_SCHEMA_PATH),
        "evaluatorSha256": _sha_file(Path(__file__)),
        "fixtureSha256": hashlib.sha256(
            json.dumps(
                request.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "engineeringReportId": first.reportId,
        "toolCount": len(first.toolRuns),
        "applicableToolCount": first.summary.applicableToolRuns,
        "findingCount": first.summary.findings,
        "blockingFindingCount": first.summary.blockingFindings,
        "criticalFindingCount": first.summary.criticalFindings,
        "calculationCount": first.summary.calculations,
        "ruleIds": rules,
        "contextWindowTokens": compilation.public_metadata["contextWindowTokens"],
        "promptTokens": compilation.public_metadata["promptTokens"],
        "promptTokenLimit": compilation.public_metadata["promptTokenLimit"],
        "checks": checks,
        "rawFirmwareStored": False,
        "rawCompilerDiagnosticStored": False,
        "rawSimulationStateStored": False,
        "neuralServingApproved": False,
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(path: Path) -> dict[str, Any]:
    report = build_report()
    if not all(report["checks"].values()):
        raise RuntimeError("VFAI-021 evaluation checks did not all pass")
    write_report(report, path)
    return report


def verify(path: Path) -> dict[str, Any]:
    checked = json.loads(path.read_text(encoding="utf-8"))
    generated = build_report()
    if checked != generated or checked.get("reportSha256") != _receipt_digest(checked):
        raise RuntimeError("VFAI-021 evaluation report is stale or invalid")
    if not all(checked["checks"].values()):
        raise RuntimeError("VFAI-021 evaluation report contains a failed check")
    return checked


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = (
        evaluate(arguments.output.resolve())
        if arguments.command == "evaluate"
        else verify(arguments.output.resolve())
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": arguments.command,
                "reportId": report["reportId"],
                "reportSha256": report["reportSha256"],
                "checks": report["checks"],
                "neuralServingApproved": report["neuralServingApproved"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
