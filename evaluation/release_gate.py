"""Release-blocking evaluator for the frozen VoltForge AI held-out suite."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Sequence

from benchmarking.runtime_baseline import offline_runtime, validate_schema_contract
from evaluation.leakage import (
    FIXTURE_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    load_cases,
    scan_corpora,
    verify_frozen_suite,
)
from task_schema.adapters import evaluation_case_to_task_record
from task_schema.compiler import compile_task_record


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = EVALUATION_ROOT / "reports" / "current-system-baseline.json"
REPORT_SCHEMA_VERSION = 1
CASE_SCHEMA_PATH = EVALUATION_ROOT / "evaluation-case.schema.json"
REPORT_SCHEMA_PATH = EVALUATION_ROOT / "evaluation-report.schema.json"


def _endpoint(orchestrator: Any) -> str:
    endpoints = orchestrator.performance_metrics.get_summary().get("endpoints", {})
    return next(reversed(endpoints), "unknown") if endpoints else "unknown"


def _chat(case_input: dict[str, Any]) -> dict[str, Any]:
    from engine.reasoning import ElectronicsReasoningOrchestrator

    orchestrator = ElectronicsReasoningOrchestrator(internet_retrieval_enabled=False)
    response = orchestrator.process_chat(
        message=str(case_input["message"]),
        board_type=str(case_input.get("boardType") or "ARDUINO_UNO"),
        components=list(case_input.get("components") or []),
        wires=list(case_input.get("wires") or []),
        code=str(case_input.get("code") or ""),
        context={"history": list(case_input.get("history") or [])},
    )
    dispatch = _endpoint(orchestrator)
    output = response.model_dump()
    output["dispatch"] = dispatch
    return {
        "status": "completed",
        "executionClass": "local-deterministic",
        "fallbackUsed": dispatch in {"fallback", "clarify", "unknown"},
        "output": output,
    }


def _circuit_validation(case_input: dict[str, Any]) -> dict[str, Any]:
    from circuit_verifier import ElectricalVerifier

    output = ElectricalVerifier.verify_circuit(
        board_type=str(case_input.get("boardType") or "ARDUINO_UNO"),
        components=list(case_input.get("components") or []),
        wires=list(case_input.get("wires") or []),
        code=str(case_input.get("code") or ""),
    )
    return _tool_outcome(output, "deterministic-circuit-tool")


def _wiring(case_input: dict[str, Any]) -> dict[str, Any]:
    from engine.pin_router import PinRouter

    output = PinRouter.resolve_connections(
        str(case_input["componentName"]), str(case_input.get("boardType") or "ARDUINO_UNO")
    )
    return _tool_outcome(output, "deterministic-pin-tool")


def _board_pins(case_input: dict[str, Any]) -> dict[str, Any]:
    from engine.pin_router import PinRouter

    output = PinRouter.get_board_pinout(str(case_input.get("boardType") or "ARDUINO_UNO"))
    return _tool_outcome(output, "deterministic-pin-tool")


def _firmware_review(case_input: dict[str, Any]) -> dict[str, Any]:
    from engine.firmware_analyzer import FirmwareAnalyzer

    output = FirmwareAnalyzer.analyze(
        str(case_input.get("code") or ""),
        list(case_input.get("components") or []),
        str(case_input.get("boardType") or "ARDUINO_UNO"),
    )
    return _tool_outcome(output, "deterministic-firmware-review")


def _firmware_generation(case_input: dict[str, Any]) -> dict[str, Any]:
    from engine.code_generator import FirmwareCodeGenerator

    output = FirmwareCodeGenerator.generate(
        list(case_input.get("components") or []),
        list(case_input.get("wires") or []),
        str(case_input.get("boardType") or "ARDUINO_UNO"),
        str(case_input.get("instructions") or ""),
    )
    return _tool_outcome(output, "deterministic-firmware-generator")


def _compiler_repair(_case_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "executionClass": "missing-capability",
        "fallbackUsed": False,
        "reasonCode": "COMPILER_REPAIR_NOT_IMPLEMENTED",
        "output": {},
    }


def _simulation_interpretation(case_input: dict[str, Any]) -> dict[str, Any]:
    from api.copilot import prepare_grounded_context
    from api.schemas import ChatRequest
    from config import get_settings
    from engine.reasoning import ElectronicsReasoningOrchestrator

    request = ChatRequest.model_validate(case_input)
    grounded = prepare_grounded_context(request, get_settings())
    simulation_event = next(
        (event for event in grounded.tool_events if event.get("name") == "inspect_simulation_state"),
        None,
    )
    orchestrator = ElectronicsReasoningOrchestrator(internet_retrieval_enabled=False)
    response = orchestrator.process_chat(
        message=request.message,
        board_type=request.boardType or "ARDUINO_UNO",
        components=request.components,
        wires=request.wires,
        code=request.code or "",
        context={"history": [], "simulationState": request.simulationState},
    )
    dispatch = _endpoint(orchestrator)
    return {
        "status": "completed",
        "executionClass": "local-deterministic-with-client-evidence",
        "fallbackUsed": dispatch in {"fallback", "clarify", "unknown"},
        "output": {
            "reply": response.reply,
            "dispatch": dispatch,
            "groundingEvidence": (simulation_event or {}).get("evidence"),
            "groundingStatus": (simulation_event or {}).get("status"),
        },
    }


def _search_grounding(case_input: dict[str, Any]) -> dict[str, Any]:
    from web_search_engine import WebSearchEngine

    cache_path = EVALUATION_ROOT / ".offline-evaluation-cache-disabled.json"
    output = WebSearchEngine(
        cache_file=str(cache_path), internet_enabled=False
    ).get_component_info(str(case_input["component"]))
    return _tool_outcome(output, "local-retrieval")


def _memory_isolation(case_input: dict[str, Any]) -> dict[str, Any]:
    from api.chat import _deterministic_response, _get_local_orchestrator
    from api.schemas import ChatRequest

    _get_local_orchestrator.cache_clear()
    first = _deterministic_response(
        ChatRequest(message=str(case_input["messageA"]), sessionId=str(case_input["sessionA"])),
        False,
    )
    second = _deterministic_response(
        ChatRequest(message=str(case_input["messageB"]), sessionId=str(case_input["sessionB"])),
        False,
    )
    _get_local_orchestrator.cache_clear()
    canary = str(case_input["canary"])
    output = {
        "sessionAReplyDigest": hashlib.sha256(first.reply.encode("utf-8")).hexdigest(),
        "sessionBReply": second.reply,
        "canaryLeaked": canary.casefold() in second.reply.casefold(),
    }
    return _tool_outcome(output, "local-session-boundary")


REPEAT_MARKER = re.compile(r"^<repeat:(\d+):(.+)>$")


def _expand_repeat_markers(value: Any) -> Any:
    if isinstance(value, str):
        match = REPEAT_MARKER.fullmatch(value)
        if match:
            return match.group(2) * int(match.group(1))
        return value
    if isinstance(value, list):
        return [_expand_repeat_markers(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_repeat_markers(item) for key, item in value.items()}
    return value


def _request_validation(case_input: dict[str, Any]) -> dict[str, Any]:
    from api.schemas import ChatRequest
    from pydantic import ValidationError

    payload = _expand_repeat_markers(case_input.get("payload", {}))
    try:
        ChatRequest.model_validate(payload)
    except ValidationError as error:
        output = {
            "rejected": True,
            "errorCount": error.error_count(),
            "errorTypes": [item["type"] for item in error.errors(include_url=False)],
        }
    else:
        output = {"rejected": False, "errorCount": 0, "errorTypes": []}
    return _tool_outcome(output, "typed-request-validation")


def _grounding_guard(case_input: dict[str, Any]) -> dict[str, Any]:
    from api.copilot import prepare_grounded_context
    from api.schemas import ChatRequest
    from config import get_settings

    grounded = prepare_grounded_context(ChatRequest.model_validate(case_input), get_settings())
    output = {
        "promptContext": grounded.prompt_context,
        "proposal": grounded.proposal,
        "toolEvents": grounded.tool_events,
    }
    return _tool_outcome(output, "deterministic-grounding-guard")


def _tool_outcome(output: Any, execution_class: str) -> dict[str, Any]:
    return {
        "status": "completed",
        "executionClass": execution_class,
        "fallbackUsed": False,
        "output": output,
    }


ADAPTERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "chat": _chat,
    "circuit_validation": _circuit_validation,
    "wiring": _wiring,
    "board_pins": _board_pins,
    "firmware_review": _firmware_review,
    "firmware_generation": _firmware_generation,
    "compiler_repair": _compiler_repair,
    "simulation_interpretation": _simulation_interpretation,
    "search_grounding": _search_grounding,
    "memory_isolation": _memory_isolation,
    "request_validation": _request_validation,
    "grounding_guard": _grounding_guard,
}


def _resolve_path(value: Any, path: str) -> Any:
    if path in {"", "$"}:
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(path)
    return current


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _assertion_result(output: Any, assertion: dict[str, Any]) -> tuple[bool, str]:
    assertion_type = assertion.get("type")
    path = str(assertion.get("path") or "$")
    try:
        actual = _resolve_path(output, path)
    except KeyError:
        return False, f"missing path {path}"
    expected = assertion.get("value")
    if assertion_type == "path_equals":
        passed = actual == expected
    elif assertion_type == "path_not_empty":
        passed = bool(actual)
    elif assertion_type == "path_is_empty":
        passed = not bool(actual)
    elif assertion_type == "path_text_contains":
        passed = str(expected).casefold() in _as_text(actual).casefold()
    elif assertion_type == "path_text_contains_any":
        text = _as_text(actual).casefold()
        passed = any(str(value).casefold() in text for value in assertion.get("values", []))
    elif assertion_type == "path_text_excludes":
        passed = str(expected).casefold() not in _as_text(actual).casefold()
    elif assertion_type == "path_number_min":
        passed = isinstance(actual, (int, float)) and actual >= expected
    elif assertion_type == "path_number_max":
        passed = isinstance(actual, (int, float)) and actual <= expected
    elif assertion_type in {"list_item_field_equals", "list_item_field_text_contains"}:
        field = assertion.get("field")
        if not isinstance(actual, list):
            passed = False
        elif assertion_type == "list_item_field_equals":
            passed = any(isinstance(item, dict) and item.get(field) == expected for item in actual)
        else:
            passed = any(
                isinstance(item, dict)
                and str(expected).casefold() in _as_text(item.get(field)).casefold()
                for item in actual
            )
    else:
        return False, f"unsupported assertion type {assertion_type!r}"
    return passed, "passed" if passed else f"assertion {assertion_type} failed at {path}"


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    # The immutable fixture remains in its suite schema, but every executable
    # case must also normalize into the shared training/inference contract.
    compile_task_record(evaluation_case_to_task_record(case))
    adapter_name = str(case.get("adapter") or "")
    adapter = ADAPTERS.get(adapter_name)
    started = time.perf_counter()
    if adapter is None:
        outcome = {
            "status": "unsupported",
            "executionClass": "missing-adapter",
            "fallbackUsed": False,
            "reasonCode": "EVALUATION_ADAPTER_NOT_IMPLEMENTED",
            "output": {},
        }
    else:
        try:
            outcome = adapter(dict(case.get("input") or {}))
        except Exception as error:
            outcome = {
                "status": "error",
                "executionClass": "evaluation-error",
                "fallbackUsed": False,
                "reasonCode": type(error).__name__,
                "output": {},
            }
    elapsed_ms = round((time.perf_counter() - started) * 1_000, 3)
    assertion_results = []
    for assertion in case.get("assertions", []):
        passed, detail = _assertion_result(outcome.get("output"), assertion)
        assertion_results.append(
            {
                "type": assertion.get("type"),
                "path": assertion.get("path", "$"),
                "passed": passed,
                "detail": detail,
            }
        )
    passed_assertions = sum(1 for result in assertion_results if result["passed"])
    raw_score = passed_assertions / len(assertion_results) if assertion_results else 0.0
    credit_eligible = outcome.get("status") == "completed" and not outcome.get("fallbackUsed")
    credited_score = raw_score if credit_eligible else 0.0
    passed = credit_eligible and credited_score == 1.0
    output_text = _as_text(outcome.get("output"))
    return {
        "id": case["id"],
        "task": case["task"],
        "metricId": case["metricId"],
        "critical": bool(case.get("critical")),
        "status": outcome.get("status"),
        "executionClass": outcome.get("executionClass"),
        "fallbackUsed": bool(outcome.get("fallbackUsed")),
        "reasonCode": outcome.get("reasonCode"),
        "rawAssertionScore": round(raw_score, 6),
        "creditedScore": round(credited_score, 6),
        "passed": passed,
        "durationMs": elapsed_ms,
        "assertions": assertion_results,
        "outputSha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "outputExcerpt": output_text[:500],
    }


def _metric_scorecards(
    policy: dict[str, Any], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scorecards = []
    for metric in policy["metrics"]:
        metric_results = [result for result in results if result["metricId"] == metric["id"]]
        score = (
            sum(result["creditedScore"] for result in metric_results) / len(metric_results)
            if metric_results
            else 0.0
        )
        threshold = float(metric["minimumScore"])
        scorecards.append(
            {
                **metric,
                "caseCount": len(metric_results),
                "passedCases": sum(1 for result in metric_results if result["passed"]),
                "failedCases": sum(1 for result in metric_results if not result["passed"]),
                "unsupportedCases": sum(1 for result in metric_results if result["status"] == "unsupported"),
                "fallbackCases": sum(1 for result in metric_results if result["fallbackUsed"]),
                "score": round(score, 6),
                "thresholdMet": score >= threshold,
            }
        )
    return scorecards


def build_evaluation_report() -> dict[str, Any]:
    manifest = verify_frozen_suite(MANIFEST_PATH)
    policy = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cases = load_cases(FIXTURE_PATH)
    for case in cases:
        validate_schema_contract(case, CASE_SCHEMA_PATH)
    leakage = scan_corpora()
    with offline_runtime():
        results = [_run_case(case) for case in cases]

    metrics = _metric_scorecards(policy, results)
    overall_score = sum(metric["score"] for metric in metrics) / len(metrics)
    critical_case_failures = [
        result["id"] for result in results if result["critical"] and not result["passed"]
    ]
    failed_metric_ids = [metric["id"] for metric in metrics if not metric["thresholdMet"]]
    blocking_reasons = []
    if leakage["status"] != "pass":
        blocking_reasons.append("HELD_OUT_LEAKAGE_DETECTED")
    if critical_case_failures:
        blocking_reasons.append("CRITICAL_CASE_FAILURE")
    if failed_metric_ids:
        blocking_reasons.append("METRIC_THRESHOLD_NOT_MET")
    if overall_score < float(policy["overallMinimumScore"]):
        blocking_reasons.append("OVERALL_THRESHOLD_NOT_MET")

    from model.artifact_registry import get_artifact_health

    model_health = get_artifact_health()
    status_counts = Counter(result["status"] for result in results)
    report = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "reportId": "vfai-current-system-baseline-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "suite": {
            "id": manifest["suiteId"],
            "version": manifest["suiteVersion"],
            "sha256": manifest["suiteSha256"],
            "caseCount": manifest["caseCount"],
            "metricCount": manifest["metricCount"],
            "freezeStatus": manifest["freezeStatus"],
            "evaluatorSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "target": {
            "system": "VoltForge AI local runtime",
            "generationMode": "local-deterministic",
            "networkAccessAllowed": False,
            "neuralModelReady": bool(model_health["ready"]),
            "neuralArtifactId": model_health["artifactId"],
            "neuralModelCode": model_health["code"],
            "missingNeuralMetricsReceiveCredit": False,
        },
        "scoringPolicy": {
            "policyId": policy["policyId"],
            "overallMinimumScore": policy["overallMinimumScore"],
            "unsupportedScore": policy["unsupportedScore"],
            "errorScore": policy["errorScore"],
            "fallbackScore": policy["fallbackScore"],
            "globalCriticalFailureRules": policy["globalCriticalFailureRules"],
        },
        "leakage": leakage,
        "summary": {
            "releaseDecision": "pass" if not blocking_reasons else "blocked",
            "blockingReasons": blocking_reasons,
            "overallScore": round(overall_score, 6),
            "overallThresholdMet": overall_score >= float(policy["overallMinimumScore"]),
            "cases": len(results),
            "passedCases": sum(1 for result in results if result["passed"]),
            "failedCases": sum(1 for result in results if not result["passed"]),
            "criticalCaseFailures": critical_case_failures,
            "failedMetricIds": failed_metric_ids,
            "statusCounts": dict(sorted(status_counts.items())),
            "fallbackCases": sum(1 for result in results if result["fallbackUsed"]),
            "unsupportedCases": status_counts.get("unsupported", 0),
            "errorCases": status_counts.get("error", 0),
        },
        "metrics": metrics,
        "cases": results,
    }
    json.dumps(report, allow_nan=False)
    validate_schema_contract(report, REPORT_SCHEMA_PATH)
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return exit code 2 when any release gate is blocked.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = build_evaluation_report()
    output = arguments.output.resolve()
    write_report(report, output)
    print(
        json.dumps(
            {
                "reportId": report["reportId"],
                "releaseDecision": report["summary"]["releaseDecision"],
                "overallScore": report["summary"]["overallScore"],
                "passedCases": report["summary"]["passedCases"],
                "cases": report["summary"]["cases"],
                "output": str(output),
            }
        )
    )
    if arguments.require_pass and report["summary"]["releaseDecision"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
