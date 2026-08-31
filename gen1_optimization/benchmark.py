"""Offline, fresh-process optimization measurements for the Gen1 runtime."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import torch

from benchmarking.runtime_baseline import PeakMemorySampler, offline_runtime, process_memory_bytes
from gen1_training.data import load_approved_corpus
from model.gen1.model import VoltForgeGen1
from model.gen1.optimization import (
    InferenceOptimizationProfile,
    REFERENCE_PROFILE,
    VFAI017_SAFE_PROFILE,
    apply_model_optimizations,
    configure_torch_threads,
    model_tensor_bytes,
)
from model.registry_manager import (
    DEFAULT_TRUST_STORE_PATH,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_artifact_directory,
    write_json,
)
from model.tokenizer import VoltForgeTokenizer


AI_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARTIFACT_ID = "vfdlm-g1-edge-v0.1.1-runtime"
SOURCE_ARTIFACT_ROOT = AI_ROOT / "model" / "registry" / "artifacts" / SOURCE_ARTIFACT_ID
FROZEN_SUITE_PATH = AI_ROOT / "evaluation" / "fixtures" / "v1" / "suite.jsonl"
FROZEN_MANIFEST_PATH = AI_ROOT / "evaluation" / "frozen-manifest.v1.json"
REPORT_PATH = AI_ROOT / "benchmarks" / "reports" / "gen1-inference-optimization-v1.json"
POLICY_PATH = AI_ROOT / "model" / "gen1" / "optimization-policy.v1.json"
SCHEMA_VERSION = 1
GENERATION_STEPS = 8
GENERATION_PROBE_COUNT = 4
MEASURED_REPEATS = 3


def build_candidate_matrix() -> list[dict[str, Any]]:
    safe = VFAI017_SAFE_PROFILE
    return [
        _candidate("reference", REFERENCE_PROFILE, "reference", "reference", False),
        _candidate("kv-cache", safe, "reference", "kv-cache", True),
        _candidate(
            "sdpa",
            replace(safe, profile_id="vfai018-fp32-sdpa-kv", attention_backend="sdpa"),
            "kv-cache",
            "attention-kernel",
            True,
        ),
        _candidate(
            "memory-map",
            replace(safe, profile_id="vfai018-fp32-manual-kv-mmap", mmap_weights=True),
            "kv-cache",
            "memory-mapping",
            True,
        ),
        _candidate(
            "threads-1",
            replace(safe, profile_id="vfai018-fp32-manual-kv-t1", torch_threads=1),
            "kv-cache",
            "thread-setting",
            True,
        ),
        _candidate(
            "threads-2",
            replace(safe, profile_id="vfai018-fp32-manual-kv-t2", torch_threads=2),
            "kv-cache",
            "thread-setting",
            True,
        ),
        _candidate(
            "threads-8",
            replace(safe, profile_id="vfai018-fp32-manual-kv-t8", torch_threads=8),
            "kv-cache",
            "thread-setting",
            True,
        ),
        _candidate(
            "dynamic-batch-4",
            replace(
                safe,
                profile_id="vfai018-fp32-manual-kv-b4",
                dynamic_batch_max_size=4,
            ),
            "kv-cache",
            "dynamic-batching",
            False,
        ),
        _candidate(
            "int8",
            replace(
                safe,
                profile_id="vfai018-int8-weight-only-manual-kv",
                quantization="int8-weight-only",
            ),
            "kv-cache",
            "quantization-int8",
            False,
        ),
        _candidate(
            "int4",
            replace(
                safe,
                profile_id="vfai018-int4-weight-only-manual-kv",
                quantization="int4-weight-only",
            ),
            "kv-cache",
            "quantization-int4",
            False,
        ),
    ]


def _candidate(
    candidate_id: str,
    profile: InferenceOptimizationProfile,
    comparison_baseline: str,
    technique: str,
    runtime_selectable: bool,
) -> dict[str, Any]:
    profile.validate()
    return {
        "candidateId": candidate_id,
        "technique": technique,
        "comparisonBaselineId": comparison_baseline,
        "runtimeSelectable": runtime_selectable,
        "profile": profile.as_dict(),
    }


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _frozen_prompts(tokenizer: VoltForgeTokenizer) -> tuple[list[list[int]], dict[str, Any]]:
    records = []
    for line in FROZEN_SUITE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
        if len(records) == GENERATION_PROBE_COUNT:
            break
    prompts = []
    hashes = []
    for record in records:
        compiled = json.dumps(record["input"], sort_keys=True, separators=(",", ":"))
        hashes.append(hashlib.sha256(compiled.encode("utf-8")).hexdigest())
        token_ids = tokenizer.encode(compiled, add_bos=True, allowed_special=False)
        prompts.append(token_ids[:48])
    return prompts, {
        "suitePath": FROZEN_SUITE_PATH.relative_to(AI_ROOT).as_posix(),
        "suiteFileSha256": sha256_file(FROZEN_SUITE_PATH),
        "manifestPath": FROZEN_MANIFEST_PATH.relative_to(AI_ROOT).as_posix(),
        "manifestFileSha256": sha256_file(FROZEN_MANIFEST_PATH),
        "promptCount": len(prompts),
        "promptSha256": hashes,
        "rawPromptsStored": False,
        "rawOutputsStored": False,
    }


@torch.inference_mode()
def _evaluate_validation(model: VoltForgeGen1, tokenizer_directory: Path) -> dict[str, Any]:
    corpus = load_approved_corpus(
        model.config,
        tokenizer_directory=tokenizer_directory,
        packing_block_size=model.config.max_sequence_length,
    )
    total_loss = 0.0
    total_targets = 0
    total_correct = 0
    batch_size = 4
    started = time.perf_counter()
    for start in range(0, corpus.validation.block_count, batch_size):
        end = min(start + batch_size, corpus.validation.block_count)
        input_ids = corpus.validation.input_ids[start:end]
        labels = corpus.validation.labels[start:end]
        attention_mask = corpus.validation.attention_mask[start:end]
        output = model(input_ids, labels=labels, attention_mask=attention_mask)
        shifted_labels = labels[:, 1:]
        valid = shifted_labels != -100
        targets = int(valid.sum().item())
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("candidate produced non-finite validation loss")
        predictions = output.logits[:, :-1].argmax(dim=-1)
        total_loss += float(output.loss.item()) * targets
        total_correct += int((predictions[valid] == shifted_labels[valid]).sum().item())
        total_targets += targets
    elapsed = time.perf_counter() - started
    return {
        "splitId": corpus.manifest["splitId"],
        "validationStreamSha256": corpus.validation.stream_sha256,
        "predictedTokens": total_targets,
        "nextTokenLoss": _round(total_loss / total_targets, 9),
        "perplexity": _round(math.exp(min(total_loss / total_targets, 50.0)), 9),
        "top1Accuracy": _round(total_correct / total_targets, 9),
        "elapsedMs": _round(elapsed * 1_000),
        "tokensPerSecond": _round(total_targets / max(elapsed, 1e-12)),
    }


@torch.inference_mode()
def _decode_group(
    model: VoltForgeGen1,
    prompts: list[list[int]],
    profile: InferenceOptimizationProfile,
) -> tuple[list[list[int]], float, float]:
    pad_id = 0
    maximum = max(len(item) for item in prompts)
    padded = [[pad_id] * (maximum - len(item)) + item for item in prompts]
    masks = [[0] * (maximum - len(item)) + [1] * len(item) for item in prompts]
    input_ids = torch.tensor(padded, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.bool)
    complete_ids = input_ids
    complete_mask = attention_mask
    outputs = [[] for _ in prompts]
    started = time.perf_counter()
    result = model(
        input_ids,
        attention_mask=attention_mask,
        use_cache=profile.kv_cache_enabled,
    )
    first_token_ms = (time.perf_counter() - started) * 1_000
    cache = result.past_key_values
    logits = result.logits[:, -1]
    for index in range(GENERATION_STEPS):
        next_tokens = logits.argmax(dim=-1)
        for row, token in enumerate(next_tokens.tolist()):
            outputs[row].append(int(token))
        if index + 1 == GENERATION_STEPS:
            break
        next_column = next_tokens[:, None]
        complete_ids = torch.cat((complete_ids, next_column), dim=1)
        complete_mask = torch.cat(
            (complete_mask, torch.ones((len(prompts), 1), dtype=torch.bool)), dim=1
        )
        result = model(
            next_column if profile.kv_cache_enabled else complete_ids,
            attention_mask=complete_mask,
            past_key_values=cache if profile.kv_cache_enabled else None,
            use_cache=profile.kv_cache_enabled,
        )
        cache = result.past_key_values
        logits = result.logits[:, -1]
    return outputs, (time.perf_counter() - started) * 1_000, first_token_ms


def _measure_generation(
    model: VoltForgeGen1,
    prompts: list[list[int]],
    profile: InferenceOptimizationProfile,
) -> dict[str, Any]:
    groups = (
        [prompts]
        if profile.dynamic_batch_max_size >= len(prompts)
        else [[prompt] for prompt in prompts]
    )
    for group in groups:
        _decode_group(model, group, profile)
    latencies = []
    first_token_latencies = []
    stable_outputs: list[list[int]] | None = None
    for _ in range(MEASURED_REPEATS):
        outputs = []
        total_ms = 0.0
        first_ms = 0.0
        for group in groups:
            generated, elapsed_ms, first_token_ms = _decode_group(model, group, profile)
            outputs.extend(generated)
            total_ms += elapsed_ms
            first_ms += first_token_ms
        if stable_outputs is not None and stable_outputs != outputs:
            raise RuntimeError("greedy generation changed between candidate repetitions")
        stable_outputs = outputs
        latencies.append(total_ms)
        first_token_latencies.append(first_ms)
    generated_tokens = len(prompts) * GENERATION_STEPS
    median_ms = statistics.median(latencies)
    return {
        "requestCount": len(prompts),
        "batchSize": len(prompts) if len(groups) == 1 else 1,
        "generatedTokensPerRepeat": generated_tokens,
        "repeats": MEASURED_REPEATS,
        "latencyMs": {
            "median": _round(median_ms),
            "p95": _round(_percentile(latencies, 0.95)),
        },
        "firstTokenLatencyMs": {
            "median": _round(statistics.median(first_token_latencies)),
            "p95": _round(_percentile(first_token_latencies, 0.95)),
        },
        "throughputTokensPerSecond": _round(
            generated_tokens / max(median_ms / 1_000, 1e-12)
        ),
        "outputTokenIdsSha256": sha256_bytes(canonical_json_bytes(stable_outputs)),
        "rawOutputsStored": False,
    }


def run_worker(candidate: Mapping[str, Any]) -> dict[str, Any]:
    profile = InferenceOptimizationProfile.from_mapping(candidate["profile"])
    with offline_runtime():
        artifact = verify_artifact_directory(
            SOURCE_ARTIFACT_ROOT, trust_store_path=DEFAULT_TRUST_STORE_PATH
        )
        manifest = artifact.manifest
        model_root = SOURCE_ARTIFACT_ROOT / "model"
        tokenizer_root = SOURCE_ARTIFACT_ROOT / "tokenizer"
        configure_torch_threads(profile)
        before_rss = process_memory_bytes()[0]
        load_started = time.perf_counter()
        with PeakMemorySampler() as load_memory:
            model = VoltForgeGen1.from_checkpoint(
                model_root, device="cpu", mmap_weights=profile.mmap_weights
            )
            logical_parameters = model.parameter_count()
            optimization = apply_model_optimizations(model, profile, torch.device("cpu"))
        load_ms = (time.perf_counter() - load_started) * 1_000
        tokenizer = VoltForgeTokenizer(vocab_size=model.config.vocab_size)
        tokenizer.load(tokenizer_root)
        prompts, fixture = _frozen_prompts(tokenizer)
        with PeakMemorySampler() as execution_memory:
            validation = _evaluate_validation(model, tokenizer_root)
            generation = _measure_generation(model, prompts, profile)
        after_rss = process_memory_bytes()[0]
    return {
        "candidateId": candidate["candidateId"],
        "technique": candidate["technique"],
        "comparisonBaselineId": candidate["comparisonBaselineId"],
        "runtimeSelectable": candidate["runtimeSelectable"],
        "status": "measured",
        "profile": profile.as_dict(),
        "artifact": {
            "artifactId": artifact.artifact_id,
            "manifestSha256": artifact.manifest_sha256,
            "parameterCount": logical_parameters,
            "weightDtype": manifest["model"]["weightDtype"],
        },
        "quality": validation,
        "generation": generation,
        "memory": {
            "processBeforeLoadMiB": None
            if before_rss is None
            else _round(before_rss / (1024 * 1024)),
            "processAfterRunMiB": None
            if after_rss is None
            else _round(after_rss / (1024 * 1024)),
            "load": load_memory.report(),
            "execution": execution_memory.report(),
            "residentTensorBytes": model_tensor_bytes(model),
        },
        "loadLatencyMs": _round(load_ms),
        "optimization": optimization,
        "fixture": fixture,
        "networkAccessAllowed": False,
    }


def _comparison(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_loss = float(candidate["quality"]["nextTokenLoss"])
    baseline_loss = float(baseline["quality"]["nextTokenLoss"])
    candidate_accuracy = float(candidate["quality"]["top1Accuracy"])
    baseline_accuracy = float(baseline["quality"]["top1Accuracy"])
    candidate_tps = float(candidate["generation"]["throughputTokensPerSecond"])
    baseline_tps = float(baseline["generation"]["throughputTokensPerSecond"])
    candidate_bytes = int(candidate["memory"]["residentTensorBytes"])
    baseline_bytes = int(baseline["memory"]["residentTensorBytes"])
    exact_generation = (
        candidate["generation"]["outputTokenIdsSha256"]
        == baseline["generation"]["outputTokenIdsSha256"]
    )
    loss_delta_percent = ((candidate_loss / baseline_loss) - 1.0) * 100.0
    accuracy_delta_points = (candidate_accuracy - baseline_accuracy) * 100.0
    quantized = candidate["profile"]["quantization"] != "fp32"
    quality_preserved = (
        loss_delta_percent <= (1.0 if quantized else 0.01)
        and accuracy_delta_points >= (-0.5 if quantized else -0.01)
        and exact_generation
    )
    throughput_change = ((candidate_tps / baseline_tps) - 1.0) * 100.0
    storage_change = ((candidate_bytes / baseline_bytes) - 1.0) * 100.0
    materially_better = throughput_change >= 3.0 or storage_change <= -10.0
    return {
        "quality": {
            "nextTokenLossDeltaPercent": _round(loss_delta_percent),
            "top1AccuracyDeltaPoints": _round(accuracy_delta_points),
            "greedyOutputExactMatch": exact_generation,
            "preserved": quality_preserved,
        },
        "latency": {
            "medianGenerationChangePercent": _round(
                (
                    float(candidate["generation"]["latencyMs"]["median"])
                    / float(baseline["generation"]["latencyMs"]["median"])
                    - 1.0
                )
                * 100.0
            ),
            "medianFirstTokenChangePercent": _round(
                (
                    float(candidate["generation"]["firstTokenLatencyMs"]["median"])
                    / float(baseline["generation"]["firstTokenLatencyMs"]["median"])
                    - 1.0
                )
                * 100.0
            ),
        },
        "throughput": {"changePercent": _round(throughput_change)},
        "memory": {
            "residentTensorBytesChangePercent": _round(storage_change),
            "residentTensorBytesDelta": candidate_bytes - baseline_bytes,
        },
        "materiallyBetter": materially_better,
    }


def build_report(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = build_candidate_matrix()
    expected = [item["candidateId"] for item in matrix]
    if [item.get("candidateId") for item in measurements] != expected:
        raise ValueError("optimization measurements do not match the frozen candidate order")
    by_id = {item["candidateId"]: item for item in measurements}
    for item in measurements:
        baseline = by_id[item["comparisonBaselineId"]]
        item["comparison"] = _comparison(item, baseline)
        selectable = item["runtimeSelectable"] is True
        portable = item["profile"]["quantization"] == "fp32"
        item["decision"] = {
            "qualityGatePassed": item["comparison"]["quality"]["preserved"],
            "performanceGatePassed": item["comparison"]["materiallyBetter"],
            "portableAcrossDeclaredTiers": portable,
            "eligibleForSelection": (
                selectable
                and portable
                and item["comparison"]["quality"]["preserved"]
                and (
                    item["candidateId"] == "kv-cache"
                    or item["comparison"]["materiallyBetter"]
                )
            ),
        }
    eligible = [item for item in measurements if item["decision"]["eligibleForSelection"]]
    if not eligible:
        selected = by_id["kv-cache"]
        selection_reason = "No measured candidate cleared all gates; retain the VFAI-017 safe profile."
    else:
        selected = max(
            eligible,
            key=lambda item: (
                float(item["generation"]["throughputTokensPerSecond"]),
                -int(item["memory"]["residentTensorBytes"]),
                item["candidateId"],
            ),
        )
        selection_reason = (
            "Selected the fastest quality-preserving, runtime-selectable FP32 candidate on "
            "the measured CPU tier."
        )
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "reportId": "vfai018-gen1-inference-optimization-v1",
        "generatedAtUtc": generated,
        "networkAccessAllowed": False,
        "sourceArtifactId": SOURCE_ARTIFACT_ID,
        "sourceArtifactManifestSha256": measurements[0]["artifact"]["manifestSha256"],
        "environment": environment_identity(),
        "candidateMatrix": matrix,
        "measurements": measurements,
        "selection": {
            "selectedCandidateId": selected["candidateId"],
            "selectedProfile": selected["profile"],
            "fallbackProfile": VFAI017_SAFE_PROFILE.as_dict(),
            "reason": selection_reason,
            "modelReleaseApproved": False,
            "serviceActivationApproved": False,
        },
        "portability": {
            "releasedWeightFormat": "pytorch-fp32-weights-only-state-dict-v1",
            "declaredTiers": [
                {"tier": "cpu", "status": "measured", "required": True},
                {"tier": "cuda", "status": "conditional-unmeasured", "required": False},
                {"tier": "mps", "status": "conditional-unmeasured", "required": False},
            ],
            "quantizedFormatsReleased": [],
            "reason": "Only FP32 is selected; CPU-only int8/int4 probes cannot define a portable release.",
        },
        "limitations": [
            "The checkpoint remains undertrained and receives no release-quality credit.",
            "Timing is host-specific and does not claim CUDA or MPS performance.",
            "Dynamic batching is measurement-only until a cancellation-aware service scheduler exists.",
            "Int8/int4 are local runtime probes, not signed released weight formats.",
        ],
    }
    report["reportSha256"] = sha256_bytes(canonical_json_bytes(report))
    validate_report(report)
    return report


def build_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    selected = report["selection"]["selectedProfile"]
    fallback = report["selection"]["fallbackProfile"]
    profiles = [selected]
    if fallback["profileId"] != selected["profileId"]:
        profiles.append(fallback)
    policy = {
        "schemaVersion": 1,
        "policyId": "vfai018-gen1-optimization-policy-v1",
        "sourceArtifactId": SOURCE_ARTIFACT_ID,
        "benchmarkReportPath": "benchmarks/reports/gen1-inference-optimization-v1.json",
        "benchmarkReportSha256": report["reportSha256"],
        "selectedProfileId": selected["profileId"],
        "fallbackProfileId": fallback["profileId"],
        "profiles": profiles,
        "fallbackOnUnavailableOptimization": True,
        "qualityRegressionFallbackRequired": True,
        "artifactRollbackRequired": True,
        "quantizedReleaseApproved": False,
        "dynamicBatchServingApproved": False,
    }
    policy["policySha256"] = sha256_bytes(canonical_json_bytes(policy))
    return policy


def validate_report(report: Mapping[str, Any]) -> None:
    required = {
        "schemaVersion",
        "reportId",
        "generatedAtUtc",
        "networkAccessAllowed",
        "sourceArtifactId",
        "sourceArtifactManifestSha256",
        "environment",
        "candidateMatrix",
        "measurements",
        "selection",
        "portability",
        "limitations",
        "reportSha256",
    }
    if set(report) != required or report["schemaVersion"] != SCHEMA_VERSION:
        raise ValueError("optimization report contract is invalid")
    if report["networkAccessAllowed"] is not False:
        raise ValueError("optimization benchmark must be offline")
    if len(report["measurements"]) != len(build_candidate_matrix()):
        raise ValueError("optimization report has incomplete measurements")
    if report["selection"]["modelReleaseApproved"] is not False:
        raise ValueError("VFAI-018 cannot approve the undertrained model")
    unsigned = dict(report)
    digest = unsigned.pop("reportSha256")
    if digest != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("optimization report digest mismatch")
    json.dumps(report, allow_nan=False)


def run_all_workers(tool_path: Path) -> list[dict[str, Any]]:
    measurements = []
    for candidate in build_candidate_matrix():
        completed = subprocess.run(
            [sys.executable, str(tool_path), "worker", "--candidate-json", json.dumps(candidate)],
            cwd=AI_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        measurements.append(json.loads(completed.stdout))
    return measurements


def write_outputs(report: Mapping[str, Any]) -> None:
    write_json(REPORT_PATH, dict(report))
    write_json(POLICY_PATH, build_policy(report))


def environment_identity() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchThreadsAtHarnessStart": torch.get_num_threads(),
        "cudaAvailable": torch.cuda.is_available(),
        "mpsAvailable": bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ),
        "harnessSha256": sha256_file(Path(__file__)),
    }
