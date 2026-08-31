"""Run the VFAI-017 offline lifecycle smoke test and publish its safe receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.gen1.runtime import (  # noqa: E402
    CancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationOptions,
    LocalGen1Runtime,
    LocalRuntimeError,
)
from model.registry_manager import (  # noqa: E402
    DEFAULT_TRUST_STORE_PATH,
    canonical_json_bytes,
    sha256_bytes,
    utc_now,
    verify_artifact_directory,
    write_json,
)


ARTIFACT_ID = "vfdlm-g1-edge-v0.1.1-runtime"
ARTIFACT_ROOT = PROJECT_ROOT / "model" / "registry" / "artifacts" / ARTIFACT_ID
REPORT_PATH = PROJECT_ROOT / "evaluation" / "reports" / "gen1-runtime-smoke-v1.json"
SMOKE_PROMPT = "Explain why an LED needs a current-limiting resistor."


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def run_smoke(*, device: str = "cpu", report_path: Path = REPORT_PATH) -> dict:
    artifact = verify_artifact_directory(
        ARTIFACT_ROOT,
        trust_store_path=DEFAULT_TRUST_STORE_PATH,
    )
    runtime = LocalGen1Runtime(
        trust_store_path=DEFAULT_TRUST_STORE_PATH,
        requested_device=device,
    )
    greedy_options = GenerationOptions(max_new_tokens=8, mode="greedy")
    sample_options = GenerationOptions(
        max_new_tokens=8,
        mode="sample",
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        seed=17017,
    )

    with mock.patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("Gen1 runtime attempted outbound network access"),
    ):
        first_load_started = time.perf_counter()
        runtime.load(ARTIFACT_ROOT, allow_experimental=True)
        first_load_ms = (time.perf_counter() - first_load_started) * 1_000
        loaded_health = runtime.health()

        first = runtime.generate(SMOKE_PROMPT, options=greedy_options)
        streamed = list(runtime.stream_generate(SMOKE_PROMPT, options=greedy_options))
        streamed_text = "".join(chunk.delta for chunk in streamed)
        streamed_token_ids = tuple(
            chunk.token_id for chunk in streamed if chunk.token_id is not None
        )

        sampled_first = runtime.generate(SMOKE_PROMPT, options=sample_options)
        sampled_second = runtime.generate(SMOKE_PROMPT, options=sample_options)

        cancellation = CancellationToken()
        cancellation.cancel()
        try:
            list(runtime.stream_generate(SMOKE_PROMPT, cancellation=cancellation))
            raise AssertionError("pre-cancelled generation unexpectedly completed")
        except GenerationCancelled as error:
            cancellation_tokens = error.generated_tokens

        try:
            list(
                runtime.stream_generate(
                    SMOKE_PROMPT,
                    deadline_monotonic=time.monotonic() - 1.0,
                )
            )
            raise AssertionError("expired-deadline generation unexpectedly completed")
        except GenerationDeadlineExceeded as error:
            deadline_tokens = error.generated_tokens

        runtime.unload()
        unloaded_health = runtime.health()
        restart_started = time.perf_counter()
        runtime.load(ARTIFACT_ROOT, allow_experimental=True)
        restart_load_ms = (time.perf_counter() - restart_started) * 1_000
        restarted = runtime.generate(SMOKE_PROMPT, options=greedy_options)
        restart_health = runtime.health()
        runtime.unload()

    checks = {
        "signedArtifactVerified": True,
        "networkDeniedDuringLifecycle": True,
        "loadedAsExperimentalDegraded": (
            loaded_health["state"] == "degraded"
            and loaded_health["runtimeOperational"] is True
            and loaded_health["ready"] is False
        ),
        "boundedGreedyGenerated": 0 <= first.completion_tokens <= 8,
        "streamMatchesGenerate": (
            streamed_text == first.text and streamed_token_ids == first.token_ids
        ),
        "boundedSamplingGenerated": 0 <= sampled_first.completion_tokens <= 8,
        "samplingSeedReproducible": (
            sampled_first.token_ids == sampled_second.token_ids
            and sampled_first.text == sampled_second.text
        ),
        "preCancellationStoppedAtZeroTokens": cancellation_tokens == 0,
        "expiredDeadlineStoppedAtZeroTokens": deadline_tokens == 0,
        "unloadReportedUnavailable": unloaded_health["state"] == "unavailable",
        "restartGreedyReproducible": (
            first.token_ids == restarted.token_ids and first.text == restarted.text
        ),
        "restartUsedSameArtifact": restart_health["artifactId"] == ARTIFACT_ID,
        "serviceServingRemainsDisabled": loaded_health["ready"] is False,
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise LocalRuntimeError(
            "MODEL_RUNTIME_SMOKE_FAILED", f"Runtime smoke checks failed: {failed}."
        )

    report = {
        "schemaVersion": 1,
        "reportId": "vfai017-gen1-runtime-smoke-v1",
        "generatedAtUtc": utc_now(),
        "artifactId": ARTIFACT_ID,
        "artifactManifestSha256": artifact.manifest_sha256,
        "artifactManifestFileSha256": artifact.manifest_file_sha256,
        "releaseStatus": artifact.release_status,
        "activationEligible": artifact.activation_eligible,
        "serviceActive": False,
        "device": loaded_health["device"],
        "dtype": loaded_health["dtype"],
        "parameterCount": loaded_health["parameterCount"],
        "contextLength": loaded_health["contextLength"],
        "prompt": {
            "stored": False,
            "characters": len(SMOKE_PROMPT),
            "sha256": _text_sha256(SMOKE_PROMPT),
            "tokens": first.prompt_tokens,
        },
        "lifecycle": {
            "firstLoadMs": first_load_ms,
            "restartLoadMs": restart_load_ms,
            "loadCountAfterRestart": restart_health["loadCount"],
            "statesImplemented": [
                "unavailable",
                "loading",
                "ready",
                "degraded",
                "failed",
            ],
            "actualExperimentalStates": [
                loaded_health["state"],
                unloaded_health["state"],
                restart_health["state"],
            ],
        },
        "greedy": {
            "maximumNewTokens": greedy_options.max_new_tokens,
            "completionTokens": first.completion_tokens,
            "finishReason": first.finish_reason,
            "elapsedMs": first.elapsed_ms,
            "firstTokenMs": first.first_token_ms,
            "outputStored": False,
            "outputSha256": _text_sha256(first.text),
            "tokenIdsSha256": sha256_bytes(canonical_json_bytes(first.token_ids)),
        },
        "sampling": {
            "maximumNewTokens": sample_options.max_new_tokens,
            "temperature": sample_options.temperature,
            "topK": sample_options.top_k,
            "topP": sample_options.top_p,
            "seed": sample_options.seed,
            "completionTokens": sampled_first.completion_tokens,
            "finishReason": sampled_first.finish_reason,
            "outputStored": False,
            "outputSha256": _text_sha256(sampled_first.text),
            "tokenIdsSha256": sha256_bytes(
                canonical_json_bytes(sampled_first.token_ids)
            ),
        },
        "checks": checks,
        "privacy": {
            "rawPromptStored": False,
            "generatedTextStored": False,
            "promptLoggingAllowed": False,
        },
        "boundary": (
            "This is offline experimental runtime evidence only. It grants no release, "
            "activation, service-serving, or output-quality approval."
        ),
    }
    report["reportSha256"] = sha256_bytes(canonical_json_bytes(report))
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    try:
        report = run_smoke(device=args.device, report_path=args.report)
    except LocalRuntimeError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": error.message}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "artifactId": report["artifactId"],
                "reportSha256": report["reportSha256"],
                "device": report["device"],
                "checks": report["checks"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

