"""Offline lifecycle smoke for the signed VFAI-018 optimized artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import time


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from gen1_optimization.benchmark import (  # noqa: E402
    REPORT_PATH as BENCHMARK_REPORT_PATH,
    validate_report,
)
from model.gen1.runtime import GenerationOptions, LocalGen1Runtime  # noqa: E402
from model.registry_manager import (  # noqa: E402
    DEFAULT_TRUST_STORE_PATH,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    verify_artifact_directory,
    write_json,
)


ARTIFACT_ID = "vfdlm-g1-edge-v0.1.2-optimized"
ARTIFACT_ROOT = AI_ROOT / "model" / "registry" / "artifacts" / ARTIFACT_ID
REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "gen1-optimization-smoke-v1.json"


def run_smoke(report_path: Path = REPORT_PATH) -> dict:
    benchmark = read_json(BENCHMARK_REPORT_PATH, "OPTIMIZATION_REPORT_INVALID")
    validate_report(benchmark)
    artifact = verify_artifact_directory(
        ARTIFACT_ROOT, trust_store_path=DEFAULT_TRUST_STORE_PATH
    )
    runtime = LocalGen1Runtime(
        trust_store_path=DEFAULT_TRUST_STORE_PATH,
        requested_device="cpu",
    )
    original_connect = socket.socket.connect

    def deny_connect(_socket, _address):
        raise AssertionError("optimized inference attempted network access")

    socket.socket.connect = deny_connect
    try:
        load_started = time.perf_counter()
        runtime.load(ARTIFACT_ROOT, allow_experimental=True)
        load_ms = (time.perf_counter() - load_started) * 1_000
        health = runtime.health()
        first = runtime.generate(
            "Review an LED resistor and GPIO safety constraint.",
            options=GenerationOptions(max_new_tokens=8, mode="greedy"),
        )
        second = runtime.generate(
            "Review an LED resistor and GPIO safety constraint.",
            options=GenerationOptions(max_new_tokens=8, mode="greedy"),
        )
        runtime.unload()
        unloaded = runtime.health()
    finally:
        socket.socket.connect = original_connect

    selected = benchmark["selection"]["selectedProfile"]
    checks = {
        "signedArtifactVerified": artifact.manifest_sha256
        == "fc97eae9340d725d0e6511524e8e2c81b199eef01d35d774ecf185df39288c77",
        "networkDeniedDuringLifecycle": True,
        "selectedProfileApplied": health["optimizationProfile"] == selected,
        "noOptimizationFallbackUsed": health["optimizationFallbackCode"] is None,
        "experimentalStateRemainsDegraded": health["state"] == "degraded"
        and health["ready"] is False,
        "boundedGenerationCompleted": first.completion_tokens <= 8,
        "greedyGenerationReproducible": first.token_ids == second.token_ids,
        "rawPromptNotInHealth": "LED resistor" not in json.dumps(health),
        "unloadCompleted": unloaded["state"] == "unavailable",
        "serviceActivationRemainsDisabled": artifact.activation_eligible is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"optimized runtime smoke failed: {checks}")
    report = {
        "schemaVersion": 1,
        "reportId": "vfai018-gen1-optimization-smoke-v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifactId": artifact.artifact_id,
        "artifactManifestSha256": artifact.manifest_sha256,
        "benchmarkReportSha256": benchmark["reportSha256"],
        "selectedProfileId": selected["profileId"],
        "fallbackProfileId": benchmark["selection"]["fallbackProfile"]["profileId"],
        "device": health["device"],
        "dtype": health["dtype"],
        "loadLatencyMs": round(load_ms, 6),
        "generation": {
            "promptTokens": first.prompt_tokens,
            "completionTokens": first.completion_tokens,
            "finishReason": first.finish_reason,
            "firstTokenMs": first.first_token_ms,
            "elapsedMs": first.elapsed_ms,
            "outputTokenIdsSha256": sha256_bytes(canonical_json_bytes(first.token_ids)),
            "rawPromptStored": False,
            "rawOutputStored": False,
        },
        "checks": checks,
    }
    report["reportSha256"] = sha256_bytes(canonical_json_bytes(report))
    write_json(report_path, report)
    return report


def main() -> int:
    report = run_smoke()
    print(
        json.dumps(
            {
                "ok": True,
                "artifactId": report["artifactId"],
                "selectedProfileId": report["selectedProfileId"],
                "reportSha256": report["reportSha256"],
                "checks": report["checks"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
