"""Evaluate the privacy and capacity boundaries of VFAI-031 telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from observability import ObservabilityRegistry  # noqa: E402


REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "ai-observability-v1.json"
SENTINEL = "VFAI031_PRIVATE_PROMPT_AND_PROJECT_CONTENT"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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


def run_evaluation(report_path: Path = REPORT_PATH) -> dict[str, Any]:
    registry = ObservabilityRegistry(
        latency_alert_ms=5,
        active_request_alert=1,
        fallback_alert_count=2,
        retrieval_error_alert_count=1,
    )
    request = registry.begin_request(SENTINEL, f"/voltForge-ai/api/v1/model/chat/{SENTINEL}")
    registry.finish_request(request, status_code=200, duration_ms=20)
    registry.record_generation(
        mode="deterministic-fallback",
        fallback_used=True,
        fallback_reason="NO_APPROVED_MODEL_ARTIFACT",
        neural_attempted=True,
        tool_grounded=False,
        artifact_id="deterministic-tools-v1",
        input_tokens=12,
    )
    registry.record_generation(
        mode="deterministic-fallback",
        fallback_used=True,
        fallback_reason="NEURAL_GENERATION_NOT_ENABLED",
        neural_attempted=False,
        tool_grounded=True,
        artifact_id="deterministic-tools-v1",
    )
    registry.record_generation(
        mode="neural",
        fallback_used=False,
        fallback_reason=None,
        neural_attempted=True,
        tool_grounded=False,
        artifact_id="vfdlm-g1-edge-v0.1.0",
        input_tokens=18,
        output_tokens=9,
    )
    stream = registry.start_stream("stream-vfai031")
    registry.observe_stream_event(stream, "start", 100)
    registry.observe_stream_event(stream, "delta", 120)
    registry.observe_stream_event(stream, "complete", 180)
    registry.finish_stream(stream, outcome="success", output_characters=42)
    cancelled = registry.start_stream("stream-vfai031-cancelled")
    registry.observe_stream_event(cancelled, "start", 100)
    registry.finish_stream(cancelled, outcome="cancelled")
    registry.record_retrieval(status="complete", duration_ms=2)
    registry.record_retrieval(status="degraded", failure_code="PROVIDER_REQUEST_FAILED", duration_ms=3)
    registry.record_model_load_failure("NO_APPROVED_MODEL_ARTIFACT")

    snapshot = registry.snapshot()
    serialized = json.dumps(snapshot, sort_keys=True)
    alerts = snapshot["alerts"]["state"]
    paths = snapshot["generation"]["paths"]
    checks = {
        "rawContentNeverRetained": SENTINEL not in serialized
        and snapshot["rawPromptStored"] is False
        and snapshot["rawProjectContentStored"] is False
        and snapshot["rawModelOutputStored"] is False,
        "generationPathsDistinguished": paths.get("deterministic-fallback", 0) >= 1
        and paths.get("tool-grounded", 0) >= 1
        and paths.get("neural", 0) >= 1,
        "streamFirstTokenAndCancellationRecorded": snapshot["streams"]["firstTokenLatencyMs"]["count"] == 1
        and snapshot["streams"]["outcomes"].get("cancelled") == 1,
        "queueAndResourceMetricsBounded": snapshot["requests"]["queueTimeMs"]["count"] == 1
        and snapshot["resources"]["sampleCount"] >= 2,
        "alertsCoverRequiredFailures": alerts["modelLoadFailure"]
        and alerts["latencySaturation"]
        and alerts["repeatedFallback"]
        and alerts["retrievalErrors"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"VFAI-031 observability evaluation failed: {checks}")

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai031-ai-observability-v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checks": checks,
        "observabilitySnapshot": snapshot,
        "telemetryContract": {
            "rawPromptStored": False,
            "rawProjectContentStored": False,
            "rawModelOutputStored": False,
            "requestIdLogging": "safe-bounded-only",
            "latencyWindowSize": 256,
        },
    }
    report["reportSha256"] = sha256_value(report)
    write_json(report_path, report)
    return report


def verify_report(report_path: Path = REPORT_PATH) -> dict[str, Any]:
    report = read_json(report_path)
    digest = report.get("reportSha256")
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    if digest != sha256_value(unsigned):
        raise RuntimeError("VFAI-031 observability report digest is invalid")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise RuntimeError("VFAI-031 observability report contains a failed check")
    forbidden = {"rawPrompt", "rawProjectContent", "rawModelOutput", "promptText", "responseText"}
    if forbidden.intersection(report):
        raise RuntimeError("VFAI-031 observability report contains forbidden raw-content fields")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "verify"))
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run_evaluation(args.report) if args.command == "evaluate" else verify_report(args.report)
    print(json.dumps({"reportId": report["reportId"], "reportSha256": report.get("reportSha256"), "checks": report.get("checks")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
