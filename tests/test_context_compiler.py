from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from api.schemas import ChatRequest
from context_compiler import (
    ContextCompilerError,
    ProjectContextCompiler,
    compile_project_context,
    context_compiler_health,
)
from model.tokenizer import DEFAULT_TOKENIZER_RELEASE_PATH, VoltForgeTokenizer
from task_schema.compiler import parse_compiled_sections
from tools.evaluate_context_compiler import run_evaluation, verify_evidence


def safety_event() -> dict:
    return {
        "name": "validate_circuit",
        "status": "complete",
        "summary": "Circuit validation found one critical LED current issue.",
        "evidence": {
            "safetyScore": 20,
            "blockingIssues": True,
            "issues": [
                {
                    "severity": "CRITICAL",
                    "code": "LED_NO_RESISTOR",
                    "message": "The selected LED has no current-limiting resistor.",
                }
            ],
            "approvedActions": [
                {
                    "actionKind": "component-addition",
                    "payload": {"componentType": "RESISTOR", "value": "220 Ohm"},
                }
            ],
        },
    }


def rich_request(*, reverse: bool = False) -> ChatRequest:
    components = [
        {
            "id": "led-1",
            "type": "LED",
            "name": "Status LED",
            "pins": [{"id": "A"}, {"id": "K"}],
            "properties": {"color": "red"},
        },
        *[
            {
                "id": f"resistor-{index:03d}",
                "type": "RESISTOR",
                "name": f"R{index}",
                "pins": [{"id": "1"}, {"id": "2"}],
                "properties": {"resistance": f"{200 + index} Ohm"},
            }
            for index in range(45)
        ],
    ]
    wires = [
        {
            "id": "wire-selected",
            "fromComponent": "led-1",
            "fromPin": "A",
            "toComponent": "resistor-000",
            "toPin": "1",
        },
        *[
            {
                "id": f"wire-{index:03d}",
                "fromComponent": f"resistor-{index:03d}",
                "fromPin": "2",
                "toComponent": f"resistor-{index + 1:03d}",
                "toPin": "1",
            }
            for index in range(30)
        ],
    ]
    nets = [
        {"id": "net-led", "pins": ["led-1/A", "resistor-000/1"]},
        *[
            {
                "id": f"net-{index:03d}",
                "pins": [f"resistor-{index:03d}/2", f"resistor-{index + 1:03d}/1"],
            }
            for index in range(30)
        ],
    ]
    if reverse:
        components.reverse()
        wires.reverse()
        nets.reverse()
    context = json.dumps(
        {
            "projectName": "Compiler fixture",
            "selectedNodeId": "led-1",
            "selectedWireId": "wire-selected",
            "activeFile": {"filename": "main.ino", "language": "cpp"},
            "viewport": {"x": 10, "y": 20, "zoom": 1.2},
        },
        sort_keys=reverse,
    )
    return ChatRequest(
        message="Review the selected LED and firmware safely.",
        projectId="project-context-fixture",
        boardType="ARDUINO_UNO",
        context=context,
        canvasContext=context,
        components=components,
        wires=wires,
        netlist={"nets": nets},
        files=[
            {
                "filename": "main.ino",
                "language": "cpp",
                "content": (
                    "void setup() { pinMode(13, OUTPUT); }\n"
                    "void loop() { digitalWrite(13, HIGH); delay(100); }\n"
                )
                * 18,
            },
            {
                "filename": "helpers.h",
                "language": "cpp",
                "content": "#pragma once\nconst int LED_PIN = 13;\n" * 12,
            },
        ],
        diagnostics=[
            {
                "severity": "ERROR",
                "code": "COMPILE_FIXTURE",
                "message": "digitalWrite requires the selected output pin configuration.",
            }
        ],
        simulationState={
            "solverConverged": True,
            "nodeVoltages": {"net-led": 4.8},
            "pinStates": {"13": "HIGH"},
        },
        history=[
            {"role": "user", "content": f"Earlier project question {index}."}
            for index in range(12)
        ],
        memory=[
            {"id": f"memory-{index}", "fact": f"User-approved fact {index}."}
            for index in range(8)
        ],
        retrievedEvidence=[
            {
                "id": f"retrieved-{index}",
                "title": f"Untrusted client retrieval {index}",
                "content": "A client supplied this text; it is not authoritative.",
            }
            for index in range(8)
        ],
    )


def project_sections(compilation) -> list[dict]:
    return compilation.task_record["input"]["projectContext"]["payload"]["sections"]


def test_compiler_uses_verified_owned_tokenizer_and_exact_budget() -> None:
    compilation = compile_project_context(ChatRequest(message="Review this board safely."))
    tokenizer = VoltForgeTokenizer()
    tokenizer.load(DEFAULT_TOKENIZER_RELEASE_PATH)

    exact_tokens = len(tokenizer.encode(compilation.prompt, add_bos=True))

    assert exact_tokens == compilation.public_metadata["promptTokens"]
    assert exact_tokens <= compilation.public_metadata["promptTokenLimit"]
    assert compilation.public_metadata["remainingPromptTokens"] == (
        compilation.public_metadata["promptTokenLimit"] - exact_tokens
    )
    assert compilation.public_metadata["truncated"] is False
    assert context_compiler_health()["ready"] is True
    assert parse_compiled_sections(compilation.prompt)


def test_semantically_reordered_project_compiles_byte_identically() -> None:
    first = compile_project_context(rich_request(), tool_events=[safety_event()])
    second = compile_project_context(rich_request(reverse=True), tool_events=[safety_event()])

    assert first.prompt == second.prompt
    assert first.task_record["recordId"] == second.task_record["recordId"]
    assert (
        first.task_record["input"]["projectContext"]["sourceProjectRevision"]
        == second.task_record["input"]["projectContext"]["sourceProjectRevision"]
    )
    assert first.public_metadata["promptSha256"] == second.public_metadata["promptSha256"]


def test_truncation_keeps_task_safety_selection_relevant_net_and_active_source() -> None:
    compilation = ProjectContextCompiler().compile(
        rich_request(),
        tool_events=[
            safety_event(),
            {
                "name": "inspect_simulation_state",
                "status": "reported",
                "summary": "Client-reported simulation state was inspected.",
                "evidence": {"solverConverged": True},
            },
        ],
        context_window_tokens=2600,
        reserved_output_tokens=256,
    )
    categories = {item["category"] for item in project_sections(compilation)}
    evidence = compilation.task_record["input"]["toolEvidence"]

    assert compilation.public_metadata["truncated"] is True
    assert compilation.public_metadata["promptTokens"] <= 2344
    assert compilation.task_record["input"]["user"]["text"].startswith("Review the selected LED")
    assert "active-selection" in categories
    assert "relevant-net" in categories
    assert "active-firmware" in categories
    assert any(item["toolName"] == "tool:validate_circuit" for item in evidence)
    assert any(
        issue.get("severity") == "CRITICAL"
        for item in evidence
        for issue in item["payload"].get("issues", [])
        if isinstance(issue, dict)
    )
    assert compilation.public_metadata["allMandatoryContentIncluded"] is True
    assert all(item["source"]["untrustedData"] is True for item in project_sections(compilation))


def test_priority_omits_low_authority_context_before_high_priority_diagnostics() -> None:
    compilation = ProjectContextCompiler().compile(
        rich_request(),
        tool_events=[safety_event()],
        context_window_tokens=3000,
        reserved_output_tokens=256,
    )
    selected = compilation.public_metadata["selectedCategoryCounts"]
    omitted = compilation.public_metadata["omittedCategoryCounts"]

    assert selected.get("diagnostic") == 1
    assert omitted.get("retrieved-evidence", 0) > 0
    assert omitted.get("memory", 0) > 0


def test_firmware_chunks_preserve_complete_source_boundaries() -> None:
    compilation = compile_project_context(rich_request(), tool_events=[safety_event()])
    firmware = [
        item
        for item in project_sections(compilation)
        if item["category"] in {"active-firmware", "firmware"}
    ]

    assert firmware
    assert firmware[0]["payload"]["filename"] == "main.ino"
    assert all(item["payload"]["startLine"] <= item["payload"]["endLine"] for item in firmware)
    assert all(len(item["payload"]["fullContentSha256"]) == 64 for item in firmware)
    assert all(item["payload"]["content"] for item in firmware)
    assert all(len(item["payload"]["content"]) <= 1200 for item in firmware)


def test_current_128_token_artifact_fails_closed_without_private_details() -> None:
    sentinel = "PRIVATE_PROJECT_CONTEXT_020"
    request = ChatRequest(message="Review safely", context=sentinel)

    with pytest.raises(ContextCompilerError) as captured:
        ProjectContextCompiler().compile(
            request,
            context_window_tokens=128,
            reserved_output_tokens=64,
        )

    assert captured.value.code == "CONTEXT_WINDOW_TOO_SMALL"
    assert sentinel not in json.dumps(captured.value.details)
    assert captured.value.details["minimumSupportedContextWindowTokens"] == 768


def test_public_metadata_contains_no_raw_project_prompt_or_section_ids() -> None:
    sentinel = "PRIVATE_CONTEXT_SENTINEL_020"
    request = rich_request()
    request.context = json.dumps(
        {
            "projectName": sentinel,
            "selectedNodeId": "led-1",
            "selectedWireId": "wire-selected",
            "activeFile": {"filename": "main.ino"},
        }
    )

    compilation = compile_project_context(request, tool_events=[safety_event()])
    metadata_text = json.dumps(compilation.public_metadata, sort_keys=True)

    assert sentinel not in metadata_text
    assert "led-1" not in metadata_text
    assert compilation.public_metadata["rawPromptStored"] is False
    assert compilation.public_metadata["rawProjectContextStored"] is False
    assert len(compilation.public_metadata["selectedSectionSetSha256"]) == 64


def test_client_revision_is_preserved_exactly_across_context_selection() -> None:
    request = rich_request()
    request.projectRevision = "revision-020"
    compilation = ProjectContextCompiler().compile(
        request,
        tool_events=[safety_event()],
        context_window_tokens=2600,
        reserved_output_tokens=256,
    )
    project = compilation.task_record["input"]["projectContext"]

    assert project["sourceProjectRevision"] == "client:revision-020"
    assert project["revisionSource"] == "client"


def test_policy_tampering_is_rejected_before_compilation() -> None:
    compiler = ProjectContextCompiler()
    changed = deepcopy(compiler.policy)
    changed["budget"]["defaultContextWindowTokens"] = 8192

    with pytest.raises(ContextCompilerError) as captured:
        ProjectContextCompiler(policy=changed)

    assert captured.value.code == "CONTEXT_POLICY_CHECKSUM_MISMATCH"


def test_context_evaluation_and_snapshot_are_reproducible(tmp_path: Path) -> None:
    report_path = tmp_path / "project-context-compiler-v1.json"
    snapshot_path = tmp_path / "project-context-snapshot-v1.json"

    generated = run_evaluation(report_path, snapshot_path)
    verified = verify_evidence(report_path, snapshot_path)

    assert generated["reportSha256"] == verified["reportSha256"]
    assert all(generated["checks"].values())
    assert generated["currentArtifactCompatibility"] == "incompatible-context-window"
    assert generated["neuralServingApproved"] is False


def test_board_type_and_canvas_components_in_compiled_context() -> None:
    req = ChatRequest(
        message="What board am I using and what components do I have?",
        boardType="ESP32",
        components=[
            {"id": "led-1", "type": "LED", "name": "Status LED", "pins": [{"id": "A"}, {"id": "K"}]},
            {"id": "res-1", "type": "RESISTOR", "name": "R1", "pins": [{"id": "1"}, {"id": "2"}], "properties": {"resistance": "220 Ohm"}},
        ],
        wires=[
            {"id": "w1", "fromComponent": "led-1", "fromPin": "K", "toComponent": "res-1", "toPin": "1"},
        ],
    )
    compilation = compile_project_context(req)
    sections = project_sections(compilation)
    metadata_sec = next((s for s in sections if s["sectionId"] == "project:metadata"), None)
    assert metadata_sec is not None
    assert metadata_sec["payload"]["boardType"] == "ESP32"

    comp_ids = {s["source"]["sourceId"] for s in sections if s["category"] == "component"}
    assert "led-1" in comp_ids
    assert "res-1" in comp_ids
    wire_ids = {s["source"]["sourceId"] for s in sections if s["category"] == "wire"}
    assert "w1" in wire_ids
