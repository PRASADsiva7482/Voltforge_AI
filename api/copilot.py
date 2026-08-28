"""Bounded deterministic evidence and proposals for streamed AI chat."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import uuid
from typing import Any

from api.schemas import ChatRequest
from circuit_verifier import ElectricalVerifier
from config import AiSettings, get_settings
from engine.firmware_analyzer import FirmwareAnalyzer
from task_schema.adapters import runtime_request_to_task_record
from task_schema.compiler import compile_task_record


@dataclass(frozen=True)
class GroundedContext:
    prompt_context: str
    tool_events: list[dict[str, object]]
    task_record: dict[str, object]
    response_metadata: dict[str, object] = field(default_factory=dict)
    proposal: dict[str, object] | None = None


def prepare_grounded_context(
    request: ChatRequest,
    settings: AiSettings | None = None,
) -> GroundedContext:
    """Run allow-listed checks and construct model context from untrusted project data."""
    configured = settings or get_settings()
    tool_events: list[dict[str, object]] = []
    context: list[str] = ["<voltforge_project_data>"]
    metadata: dict[str, object] = {
        "wireSuggestions": [],
        "additions": [],
        "removals": [],
        "valueChanges": [],
        "codeFixes": [],
        "citations": [],
    }

    board_type = request.boardType or "ARDUINO_UNO"
    context.append(f"Board selected by the project: {_clean(board_type, 80)}")
    project_metadata = _project_metadata(request.context or request.canvasContext or "")
    if project_metadata:
        context.append("Project metadata: " + _json(project_metadata, 2_000))

    if request.components or request.wires:
        circuit_summary = _circuit_summary(request.components, request.wires)
        context.append("Circuit canvas summary: " + _json(circuit_summary, 18_000))
        try:
            validation = ElectricalVerifier.verify_circuit(
                board_type=board_type,
                components=request.components,
                wires=request.wires,
                code=request.code or "",
            )
            issues = list(validation.get("issues") or [])[:25]
            safety_score = int(validation.get("safetyScore", 0))
            blocking = any(str(issue.get("severity", "")).upper() == "CRITICAL" for issue in issues)
            evidence = {
                "safetyScore": safety_score,
                "blockingIssues": blocking,
                "issueCount": len(validation.get("issues") or []),
                "issues": issues,
            }
            tool_events.append(
                {
                    "name": "validate_circuit",
                    "status": "complete",
                    "summary": (
                        f"Circuit validation found {len(validation.get('issues') or [])} issue(s); "
                        f"safety score {safety_score}/100."
                    ),
                    "evidence": evidence,
                }
            )
            for key in ("wireSuggestions", "additions", "removals", "valueChanges", "codeFixes"):
                metadata[key] = list(validation.get(key) or [])[:25]
        except Exception as error:
            tool_events.append(
                {
                    "name": "validate_circuit",
                    "status": "failed",
                    "summary": "Circuit validation could not complete for the supplied canvas.",
                    "evidence": {"errorType": type(error).__name__},
                }
            )
    else:
        context.append("No circuit canvas was supplied. Do not claim to have checked project wiring.")

    firmware_sources = list(request.files)
    if not firmware_sources and request.code:
        # Backward-compatible current-editor payload.
        from api.schemas import FirmwareSource

        firmware_sources = [
            FirmwareSource(filename="active-source.ino", language="cpp", content=request.code)
        ]
    if firmware_sources:
        firmware_issues: list[dict[str, object]] = []
        firmware_fixes: list[dict[str, object]] = []
        file_summaries: list[dict[str, object]] = []
        for source in firmware_sources[:10]:
            analysis = FirmwareAnalyzer.analyze(
                source.content[:40_000], request.components, board_type
            )
            firmware_issues.extend(list(analysis.get("issues") or [])[:15])
            firmware_fixes.extend(list(analysis.get("codeFixes") or [])[:10])
            file_summaries.append(
                {
                    "filename": _clean(source.filename, 255),
                    "language": _clean(source.language, 32),
                    "score": analysis.get("score"),
                    "charactersReviewed": min(len(source.content), 40_000),
                }
            )
        firmware_evidence = {
            "compiled": False,
            "files": file_summaries,
            "issues": firmware_issues[:30],
        }
        tool_events.append(
            {
                "name": "review_firmware",
                "status": "complete",
                "summary": (
                    f"Static review checked {len(file_summaries)} firmware file(s) and found "
                    f"{len(firmware_issues)} issue(s); code was not compiled."
                ),
                "evidence": firmware_evidence,
            }
        )
        if firmware_fixes:
            metadata["codeFixes"] = [
                *list(metadata.get("codeFixes") or []),
                *firmware_fixes,
            ][:25]

    if request.simulationState:
        simulation_evidence = _simulation_summary(request.simulationState)
        tool_events.append(
            {
                "name": "inspect_simulation_state",
                "status": "reported",
                "summary": (
                    "Inspected the browser-reported simulation snapshot; the AI service did not rerun the solver."
                ),
                "evidence": simulation_evidence,
            }
        )

    if request.netlist:
        context.append("Client-reported netlist: " + _json(request.netlist, 8_000))

    context.append("</voltforge_project_data>")
    bounded_summary = "\n".join(context)[: max(1_000, configured.max_context_characters // 2)]
    task_record = runtime_request_to_task_record(
        request,
        tool_events=tool_events,
        project_payload={"boundedProjectSummary": bounded_summary},
    )
    prompt_context = compile_task_record(task_record)
    if len(prompt_context) > configured.max_context_characters:
        compact_events = [
            {
                **event,
                "evidence": {
                    "truncated": True,
                    "sha256": hashlib.sha256(
                        _json(event.get("evidence"), 100_000).encode("utf-8")
                    ).hexdigest(),
                },
            }
            for event in tool_events
        ]
        task_record = runtime_request_to_task_record(
            request,
            tool_events=compact_events,
            project_payload={
                "boundedProjectSummary": bounded_summary[: max(500, configured.max_context_characters // 4)],
                "contextTruncated": True,
            },
        )
        prompt_context = compile_task_record(task_record)
    revision = str(task_record["input"]["projectContext"]["sourceProjectRevision"])
    evidence_refs = [
        str(item["evidenceId"]) for item in task_record["input"]["toolEvidence"]
    ]
    proposal = _proposal(metadata, revision, evidence_refs)
    return GroundedContext(
        prompt_context=prompt_context,
        tool_events=tool_events,
        task_record=task_record,
        response_metadata=metadata,
        proposal=proposal,
    )


def _circuit_summary(
    components: list[dict[str, Any]], wires: list[dict[str, Any]]
) -> dict[str, object]:
    component_rows = []
    for component in components[:150]:
        pins = component.get("pins")
        component_rows.append(
            {
                "id": _clean(component.get("id") or component.get("componentId"), 80),
                "type": _clean(component.get("type"), 80),
                "name": _clean(component.get("name"), 120),
                "properties": component.get("properties") or {},
                "pins": [
                    _clean(pin.get("id") or pin.get("name"), 40)
                    for pin in pins[:40]
                    if isinstance(pin, dict)
                ]
                if isinstance(pins, list)
                else [],
            }
        )
    wire_rows = []
    for wire in wires[:300]:
        wire_rows.append(
            {
                "id": _clean(wire.get("id"), 80),
                "fromComponent": _clean(
                    wire.get("fromComponent") or wire.get("fromNodeId"), 80
                ),
                "fromPin": _clean(wire.get("fromPin") or wire.get("fromPinId"), 40),
                "toComponent": _clean(
                    wire.get("toComponent") or wire.get("toNodeId"), 80
                ),
                "toPin": _clean(wire.get("toPin") or wire.get("toPinId"), 40),
            }
        )
    return {
        "componentCount": len(components),
        "wireCount": len(wires),
        "components": component_rows,
        "wires": wire_rows,
        "truncated": len(components) > len(component_rows) or len(wires) > len(wire_rows),
    }


def _project_metadata(raw_context: str) -> dict[str, object]:
    if not raw_context or len(raw_context) > 100_000:
        return {}
    try:
        value = json.loads(raw_context)
    except json.JSONDecodeError:
        return {"description": _clean(raw_context, 1_000)} if raw_context.strip() else {}
    if not isinstance(value, dict):
        return {}
    allowed = ("projectName", "boardType", "selectedNodeId", "selectedWireId", "activeFile")
    return {key: value[key] for key in allowed if key in value}


def _simulation_summary(state: dict[str, Any]) -> dict[str, object]:
    summary: dict[str, object] = {
        "source": "client-reported",
        "isSimulating": bool(state.get("isSimulating")),
        "solverConverged": bool(state.get("solverConverged")),
    }
    for key in ("nodeVoltages", "branchCurrents", "componentPower", "pinStates"):
        value = state.get(key)
        if isinstance(value, dict):
            summary[key] = dict(list(value.items())[:80])
    logs = state.get("serialBuffer")
    if isinstance(logs, list):
        summary["serialBuffer"] = logs[-10:]
    return summary


def _proposal(
    metadata: dict[str, object],
    source_project_revision: str,
    evidence_refs: list[str],
) -> dict[str, object] | None:
    action_keys = ("wireSuggestions", "additions", "removals", "valueChanges", "codeFixes")
    count = sum(len(metadata.get(key) or []) for key in action_keys)  # type: ignore[arg-type]
    if not count:
        return None
    action_kinds = {
        "wireSuggestions": "wire-suggestion",
        "additions": "component-addition",
        "removals": "component-removal",
        "valueChanges": "value-change",
        "codeFixes": "code-fix",
    }
    structured_actions = []
    for key, action_kind in action_kinds.items():
        for item in metadata.get(key) or []:  # type: ignore[union-attr]
            structured_actions.append(
                {
                    "type": "structured-action",
                    "actionId": f"action:{uuid.uuid4()}",
                    "actionKind": action_kind,
                    "sourceProjectRevision": source_project_revision,
                    "applicationMode": "proposal-only",
                    "requiresUserConfirmation": True,
                    "evidenceRefs": evidence_refs,
                    "payload": item if isinstance(item, dict) else {"value": item},
                }
            )
    return {
        "id": str(uuid.uuid4()),
        "source": "deterministic-validation",
        "summary": f"{count} reviewable project action(s) are available; none have been applied.",
        "sourceProjectRevision": source_project_revision,
        "applicationMode": "proposal-only",
        "requiresUserConfirmation": True,
        "structuredActions": structured_actions,
        **{key: metadata.get(key) or [] for key in action_keys},
    }


def _clean(value: object, maximum: int) -> str:
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()[:maximum]


def _json(value: object, maximum: int) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, default=str)[:maximum]
