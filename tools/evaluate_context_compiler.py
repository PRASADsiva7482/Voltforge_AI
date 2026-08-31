"""Evaluate and verify the content-free VFAI-020 context-compiler evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from api.schemas import ChatRequest  # noqa: E402
from context_compiler import (  # noqa: E402
    ContextCompilerError,
    ProjectContextCompiler,
)
from model.tokenizer import DEFAULT_TOKENIZER_RELEASE_PATH, VoltForgeTokenizer  # noqa: E402
from task_schema.compiler import parse_compiled_sections  # noqa: E402


REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "project-context-compiler-v1.json"
SNAPSHOT_PATH = AI_ROOT / "evaluation" / "snapshots" / "project-context-compiler-v1.json"
REGISTRY_PATH = AI_ROOT / "model" / "registry" / "active_model.json"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fixture_request(*, reverse: bool = False, sentinel: str = "SYNTHETIC_VFAI020") -> ChatRequest:
    components = [
        {
            "id": "led-selected",
            "type": "LED",
            "pins": [{"id": "A"}, {"id": "K"}],
            "properties": {"color": "red"},
        },
        *[
            {
                "id": f"resistor-{index:02d}",
                "type": "RESISTOR",
                "pins": [{"id": "1"}, {"id": "2"}],
                "properties": {"resistance": f"{220 + index} Ohm"},
            }
            for index in range(24)
        ],
    ]
    wires = [
        {
            "id": "wire-selected",
            "fromComponent": "led-selected",
            "fromPin": "A",
            "toComponent": "resistor-00",
            "toPin": "1",
        },
        *[
            {
                "id": f"wire-{index:02d}",
                "fromComponent": f"resistor-{index:02d}",
                "fromPin": "2",
                "toComponent": f"resistor-{index + 1:02d}",
                "toPin": "1",
            }
            for index in range(18)
        ],
    ]
    nets = [
        {"id": "net-selected", "pins": ["led-selected/A", "resistor-00/1"]},
        *[
            {
                "id": f"net-{index:02d}",
                "pins": [f"resistor-{index:02d}/2", f"resistor-{index + 1:02d}/1"],
            }
            for index in range(18)
        ],
    ]
    if reverse:
        components.reverse()
        wires.reverse()
        nets.reverse()
    context = json.dumps(
        {
            "projectName": sentinel,
            "selectedNodeId": "led-selected",
            "selectedWireId": "wire-selected",
            "activeFile": {"filename": "main.ino", "language": "cpp"},
        },
        sort_keys=reverse,
    )
    return ChatRequest(
        message="Review the selected LED circuit and firmware safely.",
        projectId="vfai020-evaluation-project",
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
                    "void setup(){pinMode(13,OUTPUT);}\n"
                    "void loop(){digitalWrite(13,HIGH);delay(100);}\n"
                )
                * 16,
            }
        ],
        diagnostics=[
            {
                "severity": "ERROR",
                "code": "VFAI020_FIXTURE",
                "message": "The selected pin configuration must be reviewed.",
            }
        ],
        simulationState={"solverConverged": True, "nodeVoltages": {"net-selected": 4.7}},
        history=[
            {"role": "user", "content": f"Earlier bounded turn {index}."}
            for index in range(10)
        ],
        memory=[{"id": f"memory-{index}", "fact": "User-approved fixture fact."} for index in range(6)],
        retrievedEvidence=[
            {
                "id": f"retrieval-{index}",
                "title": "Client supplied retrieval fixture",
                "content": "This bounded client record is untrusted.",
            }
            for index in range(6)
        ],
    )


def fixture_evidence() -> list[dict[str, Any]]:
    return [
        {
            "name": "validate_circuit",
            "status": "complete",
            "summary": "A critical missing LED current-limiting resistor was found.",
            "evidence": {
                "safetyScore": 20,
                "blockingIssues": True,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "code": "LED_NO_RESISTOR",
                        "message": "The selected LED requires current limiting.",
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
    ]


def run_evaluation(
    report_path: Path = REPORT_PATH,
    snapshot_path: Path = SNAPSHOT_PATH,
) -> dict[str, Any]:
    compiler = ProjectContextCompiler()
    request = fixture_request()
    reordered = fixture_request(reverse=True)
    original_connect = socket.socket.connect

    def deny_connect(_socket, _address):
        raise AssertionError("VFAI-020 context compilation attempted network access")

    socket.socket.connect = deny_connect
    try:
        first = compiler.compile(
            request,
            tool_events=fixture_evidence(),
            context_window_tokens=3400,
            reserved_output_tokens=256,
        )
        second = compiler.compile(
            reordered,
            tool_events=fixture_evidence(),
            context_window_tokens=3400,
            reserved_output_tokens=256,
        )
        tiny_code = None
        try:
            compiler.compile(
                request,
                tool_events=fixture_evidence(),
                context_window_tokens=128,
                reserved_output_tokens=64,
            )
        except ContextCompilerError as error:
            tiny_code = error.code
    finally:
        socket.socket.connect = original_connect

    tokenizer = VoltForgeTokenizer()
    tokenizer.load(DEFAULT_TOKENIZER_RELEASE_PATH)
    exact_tokens = len(tokenizer.encode(first.prompt, add_bos=True))
    project = first.task_record["input"]["projectContext"]
    sections = project["payload"]["sections"]
    categories = {item["category"] for item in sections}
    evidence = first.task_record["input"]["toolEvidence"]
    parsed = parse_compiled_sections(first.prompt)
    registry = read_json(REGISTRY_PATH)
    minimum_context = compiler.policy["budget"]["minimumSupportedContextWindowTokens"]
    artifact_contexts = {
        item["artifactId"]: item["contextLength"] for item in registry["artifacts"]
    }
    sentinel = "SYNTHETIC_VFAI020"
    metadata_text = json.dumps(first.public_metadata, sort_keys=True)
    snapshot: dict[str, Any] = {
        "schemaVersion": 1,
        "snapshotId": "vfai020-project-context-snapshot-v1",
        "fixtureId": "vfai020-synthetic-selected-led-v1",
        "policyId": compiler.policy["policyId"],
        "policySha256": compiler.policy["policySha256"],
        "compilerVersion": compiler.policy["compilerVersion"],
        "tokenizerId": compiler.tokenizer.tokenizer_id,
        "tokenizerVersion": compiler.tokenizer.version,
        "contextWindowTokens": first.public_metadata["contextWindowTokens"],
        "reservedOutputTokens": first.public_metadata["reservedOutputTokens"],
        "promptTokenLimit": first.public_metadata["promptTokenLimit"],
        "promptTokens": first.public_metadata["promptTokens"],
        "promptSha256": first.public_metadata["promptSha256"],
        "selectedSectionSetSha256": first.public_metadata["selectedSectionSetSha256"],
        "selectedSectionCount": first.public_metadata["selectedSectionCount"],
        "omittedSectionCount": first.public_metadata["omittedSectionCount"],
        "selectedCategoryCounts": first.public_metadata["selectedCategoryCounts"],
        "omittedCategoryCounts": first.public_metadata["omittedCategoryCounts"],
        "selectedToolEvidenceCount": first.public_metadata["selectedToolEvidenceCount"],
        "sourceProjectRevision": project["sourceProjectRevision"],
        "compiledSectionTypes": [item["sectionType"] for item in parsed],
        "rawPromptStored": False,
        "rawProjectContextStored": False,
    }
    snapshot["snapshotSha256"] = sha256_value(snapshot)
    write_json(snapshot_path, snapshot)

    checks = {
        "policyChecksumVerified": len(compiler.policy["policySha256"]) == 64,
        "networkDeniedDuringCompilation": True,
        "exactTokenizerBudgetRespected": exact_tokens == first.public_metadata["promptTokens"]
        and exact_tokens <= first.public_metadata["promptTokenLimit"],
        "semanticReorderingIsByteStable": first.prompt == second.prompt,
        "taskPreserved": first.task_record["input"]["user"]["text"] == request.message,
        "safetyEvidencePreserved": any(
            item["toolName"] == "tool:validate_circuit"
            and any(
                issue.get("severity") == "CRITICAL"
                for issue in item["payload"].get("issues", [])
                if isinstance(issue, dict)
            )
            for item in evidence
        ),
        "activeSelectionPreserved": "active-selection" in categories,
        "relevantNetPreserved": "relevant-net" in categories,
        "activeFirmwareBoundaryPreserved": "active-firmware" in categories,
        "strictPriorityTruncationObserved": first.public_metadata["truncated"] is True
        and first.public_metadata["selectedCategoryCounts"].get("diagnostic") == 1
        and first.public_metadata["omittedCategoryCounts"].get("retrieved-evidence", 0) > 0,
        "lengthAndChecksumFramingVerified": bool(parsed),
        "currentArtifactFailsClosed": tiny_code == "CONTEXT_WINDOW_TOO_SMALL",
        "allCatalogedArtifactsBelowMinimum": all(
            value < minimum_context for value in artifact_contexts.values()
        ),
        "activeNeuralArtifactRemainsNull": registry.get("activeArtifactId") is None,
        "publicMetadataContainsNoRawFixture": sentinel not in metadata_text,
        "rawContentNotStored": first.public_metadata["rawPromptStored"] is False
        and first.public_metadata["rawProjectContextStored"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"VFAI-020 context evaluation failed: {checks}")
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai020-project-context-compiler-v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policyId": compiler.policy["policyId"],
        "policySha256": compiler.policy["policySha256"],
        "snapshotPath": snapshot_path.relative_to(AI_ROOT).as_posix()
        if snapshot_path.is_relative_to(AI_ROOT)
        else str(snapshot_path),
        "snapshotSha256": snapshot["snapshotSha256"],
        "artifactContextWindows": artifact_contexts,
        "minimumSupportedContextWindowTokens": minimum_context,
        "currentArtifactCompatibility": "incompatible-context-window",
        "checks": checks,
        "rawPromptStored": False,
        "rawProjectContextStored": False,
        "neuralServingApproved": False,
        "blockingReason": (
            "The current 128-token experimental artifacts cannot hold the mandatory typed "
            "prompt. A future governed model revision needs at least the policy minimum and "
            "must independently pass training, generation-quality, and release gates."
        ),
    }
    report["reportSha256"] = sha256_value(report)
    write_json(report_path, report)
    return report


def verify_evidence(
    report_path: Path = REPORT_PATH,
    snapshot_path: Path = SNAPSHOT_PATH,
) -> dict[str, Any]:
    report = read_json(report_path)
    snapshot = read_json(snapshot_path)
    report_digest = report.get("reportSha256")
    report_unsigned = dict(report)
    report_unsigned.pop("reportSha256", None)
    snapshot_digest = snapshot.get("snapshotSha256")
    snapshot_unsigned = dict(snapshot)
    snapshot_unsigned.pop("snapshotSha256", None)
    if report_digest != sha256_value(report_unsigned):
        raise RuntimeError("VFAI-020 report digest is invalid")
    if snapshot_digest != sha256_value(snapshot_unsigned):
        raise RuntimeError("VFAI-020 snapshot digest is invalid")
    if report.get("snapshotSha256") != snapshot_digest:
        raise RuntimeError("VFAI-020 report references a different snapshot")
    compiler = ProjectContextCompiler()
    if report.get("policySha256") != compiler.policy["policySha256"]:
        raise RuntimeError("VFAI-020 evidence references a different policy")
    if not all((report.get("checks") or {}).values()):
        raise RuntimeError("VFAI-020 evidence contains a failed check")
    if report.get("neuralServingApproved") is not False:
        raise RuntimeError("VFAI-020 may not approve neural serving")
    forbidden = {"rawPrompt", "rawProjectContext", "promptText", "firmwareContent"}
    if forbidden.intersection(report) or forbidden.intersection(snapshot):
        raise RuntimeError("VFAI-020 evidence stores forbidden raw content")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "verify"))
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()
    report = (
        run_evaluation(args.report, args.snapshot)
        if args.command == "evaluate"
        else verify_evidence(args.report, args.snapshot)
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": args.command,
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
