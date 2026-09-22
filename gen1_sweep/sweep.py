"""Fresh-process controlled training, benchmarking, and Pareto selection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Mapping, Sequence
import uuid

import torch

from benchmarking.runtime_baseline import PeakMemorySampler
from gen1_training import Gen1Trainer, Gen1TrainingConfig, PackedCorpus, load_approved_corpus
from model.gen1 import Gen1Config, VoltForgeGen1


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = AI_ROOT / "gen1_sweep" / "plan.v1.json"
DEFAULT_REPORT_PATH = AI_ROOT / "benchmarks" / "reports" / "gen1-architecture-sweep-v1.json"
DEFAULT_SELECTED_CONFIG_PATH = AI_ROOT / "model" / "gen1" / "configs" / "gen1-selected-v1.json"
EDGE_RANGE = (8_000_000, 25_000_000)
CORE_RANGE = (25_000_000, 60_000_000)


class SweepContractError(RuntimeError):
    """Raised when a sweep plan, candidate result, or scorecard is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SweepContractError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SweepContractError(f"{label} must be a JSON object")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fingerprint(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def load_sweep_plan(path: str | Path = DEFAULT_PLAN_PATH) -> dict[str, Any]:
    plan = _read_json(Path(path), "Gen1 sweep plan")
    expected = {
        "schemaVersion",
        "sweepId",
        "purpose",
        "referenceTier",
        "controls",
        "selectionPolicy",
        "candidates",
        "nonExecutableControls",
    }
    if set(plan) != expected or plan.get("schemaVersion") != 1:
        raise SweepContractError("sweep plan keys/schema are invalid")
    controls = plan["controls"]
    candidates = plan["candidates"]
    if not isinstance(controls, dict) or not isinstance(candidates, list) or len(candidates) < 2:
        raise SweepContractError("sweep controls/candidates are invalid")
    ids: set[str] = set()
    dimensions: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {
            "id",
            "deploymentProfile",
            "dimensions",
            "weightDecay",
            "model",
        }:
            raise SweepContractError("candidate keys are invalid")
        candidate_id = candidate["id"]
        if not isinstance(candidate_id, str) or candidate_id in ids:
            raise SweepContractError("candidate IDs must be unique strings")
        ids.add(candidate_id)
        dimensions.update(candidate["dimensions"])
        try:
            config = Gen1Config.from_dict(candidate["model"])
        except ValueError as exc:
            raise SweepContractError(f"invalid candidate {candidate_id}: {exc}") from exc
        if config.vocab_size != controls.get("vocabSize"):
            raise SweepContractError("all executable candidates must use the approved vocabulary")
        if config.max_sequence_length < controls.get("packingBlockSize", 0):
            raise SweepContractError("candidate context is shorter than the controlled packing size")
        profile = candidate["deploymentProfile"]
        limits = EDGE_RANGE if profile == "edge" else CORE_RANGE if profile == "core" else None
        if limits is None or not limits[0] <= config.parameter_count() <= limits[1]:
            raise SweepContractError(
                f"candidate {candidate_id} parameter count is outside its declared profile range"
            )
    required_dimensions = {
        "depth",
        "width",
        "query-heads",
        "kv-heads",
        "ffn-ratio",
        "context",
        "dropout",
        "weight-decay",
    }
    if not required_dimensions.issubset(dimensions):
        raise SweepContractError(
            f"sweep omits architecture dimensions: {sorted(required_dimensions - dimensions)}"
        )
    controls_by_dimension = {
        item.get("dimension"): item for item in plan["nonExecutableControls"]
    }
    vocabulary_control = controls_by_dimension.get("vocabulary")
    if not isinstance(vocabulary_control, dict) or vocabulary_control.get("status") != (
        "rejected-before-training"
    ):
        raise SweepContractError("sweep must explicitly resolve the vocabulary dimension")
    plan["planFingerprint"] = _fingerprint(plan)
    return plan


def _candidate(plan: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [item for item in plan["candidates"] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise SweepContractError(f"unknown sweep candidate: {candidate_id}")
    return matches[0]


@torch.no_grad()
def _quality(model: VoltForgeGen1, corpus: PackedCorpus, batch_size: int) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_targets = 0
    correct = 0
    for start in range(0, corpus.validation.block_count, batch_size):
        end = min(start + batch_size, corpus.validation.block_count)
        input_ids = corpus.validation.input_ids[start:end].to(device)
        labels = corpus.validation.labels[start:end].to(device)
        attention_mask = corpus.validation.attention_mask[start:end].to(device)
        output = model(input_ids, labels=labels, attention_mask=attention_mask)
        if output.loss is None or not torch.isfinite(output.loss):
            raise SweepContractError("candidate validation produced a non-finite loss")
        targets = labels[:, 1:]
        valid = targets != -100
        predictions = output.logits[:, :-1].argmax(dim=-1)
        target_count = int(valid.sum().item())
        total_loss += float(output.loss.item()) * target_count
        total_targets += target_count
        correct += int((predictions[valid] == targets[valid]).sum().item())
    loss = total_loss / total_targets
    return {
        "nextTokenLoss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "top1TokenAccuracy": correct / total_targets,
        "correctTokens": correct,
        "predictedTokens": total_targets,
        "releaseCredit": 0.0,
        "classification": "architecture-proxy-only-not-release-evaluation",
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _one_generation(
    model: VoltForgeGen1, prompt: torch.Tensor, generated_tokens: int
) -> tuple[float, float]:
    device = prompt.device
    _synchronize(device)
    first_started = time.perf_counter()
    output = model(prompt, use_cache=True)
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    cache = output.past_key_values
    _synchronize(device)
    first_ms = (time.perf_counter() - first_started) * 1_000
    decode_count = generated_tokens - 1
    decode_started = time.perf_counter()
    for _ in range(decode_count):
        output = model(next_token, past_key_values=cache, use_cache=True)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
    _synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    return first_ms, decode_count / max(decode_seconds, 1e-12)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _inference_benchmark(
    model: VoltForgeGen1, corpus: PackedCorpus, controls: Mapping[str, Any]
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prompt_size = int(controls["inferencePromptTokens"])
    generated = int(controls["generatedTokens"])
    valid = corpus.validation.input_ids[0][corpus.validation.attention_mask[0]]
    if len(valid) < prompt_size or prompt_size + generated - 1 > model.config.max_sequence_length:
        raise SweepContractError("inference probe exceeds candidate context")
    prompt = valid[:prompt_size].unsqueeze(0).to(device)
    model.eval()
    for _ in range(int(controls["inferenceWarmupRuns"])):
        _one_generation(model, prompt, generated)
    first_samples: list[float] = []
    decode_samples: list[float] = []
    with PeakMemorySampler() as memory:
        for _ in range(int(controls["inferenceMeasuredRuns"])):
            first_ms, decode_tps = _one_generation(model, prompt, generated)
            first_samples.append(first_ms)
            decode_samples.append(decode_tps)
    return {
        "promptTokens": prompt_size,
        "generatedTokensPerRun": generated,
        "measuredRuns": len(first_samples),
        "firstTokenLatencyMs": {
            "median": statistics.median(first_samples),
            "p95": _percentile(first_samples, 0.95),
            "samples": first_samples,
        },
        "cachedDecodeTokensPerSecond": {
            "median": statistics.median(decode_samples),
            "samples": decode_samples,
        },
        "memory": memory.report(),
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _set_threads(controls: Mapping[str, Any]) -> None:
    torch.set_num_threads(int(controls["torchNumThreads"]))
    requested_interop = int(controls["torchNumInteropThreads"])
    if torch.get_num_interop_threads() != requested_interop:
        try:
            torch.set_num_interop_threads(requested_interop)
        except RuntimeError as exc:
            raise SweepContractError("Torch interop threads were initialized before sweep control") from exc


def run_candidate(
    candidate_id: str,
    work_root: str | Path,
    *,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
) -> dict[str, Any]:
    plan = load_sweep_plan(plan_path)
    candidate = _candidate(plan, candidate_id)
    controls = plan["controls"]
    _set_threads(controls)
    config = Gen1Config.from_dict(candidate["model"])
    corpus = load_approved_corpus(
        config, packing_block_size=int(controls["packingBlockSize"])
    )
    candidate_root = Path(work_root).resolve() / candidate_id
    if candidate_root.exists() and any(candidate_root.iterdir()):
        raise SweepContractError(f"candidate work directory is not empty: {candidate_root}")
    candidate_root.mkdir(parents=True, exist_ok=True)
    training = Gen1TrainingConfig(
        model=config,
        seed=int(controls["seed"]),
        max_steps=int(controls["optimizerSteps"]),
        micro_batch_size=int(controls["microBatchSize"]),
        gradient_accumulation_steps=int(controls["gradientAccumulationSteps"]),
        peak_learning_rate=float(controls["peakLearningRate"]),
        minimum_learning_rate=float(controls["minimumLearningRate"]),
        warmup_steps=int(controls["warmupSteps"]),
        weight_decay=float(candidate["weightDecay"]),
        max_gradient_norm=float(controls["maxGradientNorm"]),
        validation_interval=int(controls["optimizerSteps"]),
        checkpoint_interval=int(controls["optimizerSteps"]),
        precision=str(controls["precision"]),
        device=str(controls["device"]),
    )
    process_cpu_started = time.process_time()
    with PeakMemorySampler() as training_memory:
        trainer = Gen1Trainer.create(
            training,
            corpus,
            candidate_root / "run",
            run_id=f"{plan['sweepId']}-{candidate_id}",
        )
        initial_quality = _quality(trainer.model, corpus, training.micro_batch_size)
        training_started = time.perf_counter()
        training_result = trainer.train()
        training_wall_seconds = time.perf_counter() - training_started
        final_quality = _quality(trainer.model, corpus, training.micro_batch_size)
    process_cpu_seconds = time.process_time() - process_cpu_started
    if trainer.tokens_seen != controls["expectedPredictedTokens"]:
        raise SweepContractError(
            f"candidate consumed {trainer.tokens_seen} tokens, expected "
            f"{controls['expectedPredictedTokens']}"
        )
    inference = _inference_benchmark(trainer.model, corpus, controls)
    checkpoint = Path(training_result["lastCheckpoint"])
    if not checkpoint.is_absolute():
        checkpoint = AI_ROOT / checkpoint
    run_manifest = _read_json(candidate_root / "run" / "run-manifest.json", "run manifest")
    result = {
        "schemaVersion": 1,
        "sweepId": plan["sweepId"],
        "planFingerprint": plan["planFingerprint"],
        "candidateId": candidate_id,
        "deploymentProfile": candidate["deploymentProfile"],
        "dimensions": candidate["dimensions"],
        "status": "completed",
        "modelConfig": config.to_dict(),
        "parameterCount": config.parameter_count(),
        "weightDecay": candidate["weightDecay"],
        "controls": {
            "packingBlockSize": controls["packingBlockSize"],
            "optimizerSteps": controls["optimizerSteps"],
            "predictedTokens": trainer.tokens_seen,
            "seed": controls["seed"],
            "precision": trainer.precision,
            "device": str(trainer.device),
            "torchNumThreads": torch.get_num_threads(),
            "torchNumInteropThreads": torch.get_num_interop_threads(),
            "datasetFingerprint": corpus.fingerprint,
            "trainingConfigFingerprint": training.fingerprint(),
        },
        "quality": {
            "initial": initial_quality,
            "final": final_quality,
            "lossImprovement": initial_quality["nextTokenLoss"]
            - final_quality["nextTokenLoss"],
        },
        "training": {
            "wallSeconds": training_wall_seconds,
            "processCpuSeconds": process_cpu_seconds,
            "predictedTokens": trainer.tokens_seen,
            "tokensPerSecond": trainer.tokens_seen / training_wall_seconds,
            "peakMemory": training_memory.report(),
            "finalGradientNormBeforeClip": trainer.metrics[-1]["gradientNormBeforeClip"],
        },
        "inference": inference,
        "artifact": {
            "checkpointBytes": _directory_size(checkpoint),
            "modelWeightsSha256": run_manifest["checkpoints"][-1]["modelWeightsSha256"],
            "trainingStateSha256": run_manifest["checkpoints"][-1]["trainingStateSha256"],
        },
        "energyRuntimeCost": {
            "energyMeasurementAvailable": False,
            "energyReason": "Reference host exposes no calibrated package/wall power sensor; no watt-hour value is invented.",
            "trainingWallSeconds": training_wall_seconds,
            "processCpuSeconds": process_cpu_seconds,
            "cpuSecondsPerMillionTrainingTokens": process_cpu_seconds
            * 1_000_000
            / trainer.tokens_seen,
        },
        "environment": {
            "generatedAtUtc": _utc_now(),
            "referenceTier": plan["referenceTier"],
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "hardware": run_manifest["hardware"],
            "codeRevision": run_manifest["codeRevision"],
        },
    }
    result["resultSha256"] = _fingerprint(result)
    _atomic_json(candidate_root / "candidate-result.json", result)
    return result


def _objective_vector(result: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(result["quality"]["final"]["nextTokenLoss"]),
        -float(result["quality"]["final"]["top1TokenAccuracy"]),
        float(result["training"]["wallSeconds"]),
        float(result["inference"]["firstTokenLatencyMs"]["median"]),
        -float(result["inference"]["cachedDecodeTokensPerSecond"]["median"]),
        float(result["training"]["peakMemory"]["peakWorkingSetMiB"]),
        float(result["artifact"]["checkpointBytes"]),
        float(result["training"]["processCpuSeconds"]),
    )


def pareto_frontier(results: Sequence[Mapping[str, Any]]) -> list[str]:
    completed = [result for result in results if result.get("status") == "completed"]
    frontier: list[str] = []
    for candidate in completed:
        candidate_vector = _objective_vector(candidate)
        dominated = False
        for other in completed:
            if other["candidateId"] == candidate["candidateId"]:
                continue
            other_vector = _objective_vector(other)
            if all(left <= right for left, right in zip(other_vector, candidate_vector)) and any(
                left < right for left, right in zip(other_vector, candidate_vector)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["candidateId"])
    return sorted(frontier)


def select_candidate(
    results: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    completed = [result for result in results if result.get("status") == "completed"]
    frontier_ids = set(pareto_frontier(completed))
    if not completed or not frontier_ids:
        raise SweepContractError("no completed Pareto candidate is available")
    edge = [item for item in completed if item["deploymentProfile"] == "edge"]
    core = [item for item in completed if item["deploymentProfile"] == "core"]
    if not edge:
        raise SweepContractError("selection requires at least one edge candidate")
    best_edge_loss = min(item["quality"]["final"]["nextTokenLoss"] for item in edge)
    best_core_loss = (
        min(item["quality"]["final"]["nextTokenLoss"] for item in core) if core else None
    )
    core_improvement = (
        0.0
        if best_core_loss is None
        else (best_edge_loss - best_core_loss) * 100.0 / best_edge_loss
    )
    core_threshold = float(policy["coreMinimumMaterialLossImprovementPercent"])
    allowed_profiles = {"edge", "core"} if core_improvement >= core_threshold else {"edge"}
    eligible = [
        item
        for item in completed
        if item["candidateId"] in frontier_ids
        and item["deploymentProfile"] in allowed_profiles
    ]
    if not eligible:
        raise SweepContractError("profile policy removed every Pareto candidate")
    best_eligible_loss = min(item["quality"]["final"]["nextTokenLoss"] for item in eligible)
    tolerance = float(policy["edgeQualityTolerancePercent"]) / 100.0
    quality_eligible = [
        item
        for item in eligible
        if item["quality"]["final"]["nextTokenLoss"] <= best_eligible_loss * (1 + tolerance)
    ]
    minima = {
        "training": min(item["training"]["wallSeconds"] for item in quality_eligible),
        "first": min(
            item["inference"]["firstTokenLatencyMs"]["median"] for item in quality_eligible
        ),
        "memory": min(
            item["training"]["peakMemory"]["peakWorkingSetMiB"] for item in quality_eligible
        ),
        "artifact": min(item["artifact"]["checkpointBytes"] for item in quality_eligible),
        "cpu": min(item["training"]["processCpuSeconds"] for item in quality_eligible),
    }
    max_decode = max(
        item["inference"]["cachedDecodeTokensPerSecond"]["median"]
        for item in quality_eligible
    )
    scored = []
    for item in quality_eligible:
        quality_gap_percent = (
            item["quality"]["final"]["nextTokenLoss"] - best_eligible_loss
        ) * 100.0 / best_eligible_loss
        efficiency_score = 5.0 * quality_gap_percent + sum(
            (
                item["training"]["wallSeconds"] / minima["training"] - 1.0,
                item["inference"]["firstTokenLatencyMs"]["median"] / minima["first"]
                - 1.0,
                item["training"]["peakMemory"]["peakWorkingSetMiB"] / minima["memory"]
                - 1.0,
                item["artifact"]["checkpointBytes"] / minima["artifact"] - 1.0,
                item["training"]["processCpuSeconds"] / minima["cpu"] - 1.0,
                max_decode
                / item["inference"]["cachedDecodeTokensPerSecond"]["median"]
                - 1.0,
            )
        )
        scored.append(
            {
                "candidateId": item["candidateId"],
                "qualityGapPercent": quality_gap_percent,
                "efficiencyScore": efficiency_score,
            }
        )
    selected_score = min(scored, key=lambda item: (item["efficiencyScore"], item["candidateId"]))
    selected = next(
        item for item in completed if item["candidateId"] == selected_score["candidateId"]
    )
    return {
        "selectedCandidateId": selected["candidateId"],
        "deploymentProfile": selected["deploymentProfile"],
        "parameterCount": selected["parameterCount"],
        "paretoFrontier": sorted(frontier_ids),
        "bestEdgeValidationLoss": best_edge_loss,
        "bestCoreValidationLoss": best_core_loss,
        "coreLossImprovementPercent": core_improvement,
        "coreMaterialThresholdMet": core_improvement >= core_threshold,
        "qualityTolerancePercent": float(policy["edgeQualityTolerancePercent"]),
        "qualityEligibleCandidates": sorted(item["candidateId"] for item in quality_eligible),
        "efficiencyScores": sorted(scored, key=lambda item: item["candidateId"]),
        "selectionRule": policy["rule"],
        "parameterCountRewarded": False,
        "status": "selected-for-vfai014-bootstrap-training-not-release",
        "rationale": (
            "Core retained eligibility because its loss improvement met the material threshold."
            if core_improvement >= core_threshold
            else "Core was not selected because its proxy-loss improvement did not meet the material cost threshold."
        ),
    }


def _validate_result(result: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    declared = result.get("resultSha256")
    actual = _fingerprint({key: value for key, value in result.items() if key != "resultSha256"})
    if declared != actual:
        raise SweepContractError(f"candidate result checksum mismatch: {result.get('candidateId')}")
    if result.get("sweepId") != plan["sweepId"] or result.get("planFingerprint") != plan[
        "planFingerprint"
    ]:
        raise SweepContractError("candidate result belongs to a different sweep plan")
    if result.get("controls", {}).get("predictedTokens") != plan["controls"][
        "expectedPredictedTokens"
    ]:
        raise SweepContractError("candidate result changed the controlled token budget")


def assemble_scorecard(
    work_root: str | Path,
    *,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    selected_config_path: str | Path = DEFAULT_SELECTED_CONFIG_PATH,
) -> dict[str, Any]:
    plan = load_sweep_plan(plan_path)
    results = []
    for candidate in plan["candidates"]:
        result = _read_json(
            Path(work_root) / candidate["id"] / "candidate-result.json",
            f"candidate result {candidate['id']}",
        )
        _validate_result(result, plan)
        results.append(result)
    tokens = {result["controls"]["predictedTokens"] for result in results}
    datasets = {result["controls"]["datasetFingerprint"] for result in results}
    if tokens != {plan["controls"]["expectedPredictedTokens"]} or len(datasets) != 1:
        raise SweepContractError("candidate controls are not comparable")
    selection = select_candidate(results, plan["selectionPolicy"])
    selected_result = next(
        result for result in results if result["candidateId"] == selection["selectedCandidateId"]
    )
    report = {
        "schemaVersion": 1,
        "reportId": "vfai013-gen1-architecture-scorecard-v1",
        "generatedAtUtc": _utc_now(),
        "sweepId": plan["sweepId"],
        "planPath": Path(plan_path).resolve().relative_to(AI_ROOT).as_posix(),
        "planFileSha256": _sha256_file(Path(plan_path)),
        "planFingerprint": plan["planFingerprint"],
        "referenceTier": plan["referenceTier"],
        "controls": plan["controls"],
        "dimensionCoverage": sorted(
            {dimension for item in plan["candidates"] for dimension in item["dimensions"]}
        ),
        "nonExecutableControls": plan["nonExecutableControls"],
        "qualityBoundary": {
            "proxy": "Frozen VFAI-010 model-validation next-token loss/perplexity/top-1 accuracy",
            "releaseSuite": "not executed against raw decoder; VFAI-014 and runtime integration must produce independent VFAI-005 release scorecards",
            "releaseCredit": 0.0,
        },
        "energyBoundary": {
            "energyMeasurementAvailable": False,
            "reason": "No calibrated package or wall power sensor is exposed by the CPU reference host; process CPU and wall time are reported instead of invented energy.",
        },
        "scorecard": results,
        "selectionPolicy": plan["selectionPolicy"],
        "selection": selection,
        "selectedModelConfig": selected_result["modelConfig"],
        "trainingDataReadiness": {
            "approvedTrainingTokens": 71411,
            "edgeMinimumTargetTokens": 100000000,
            "targetCoveragePercent": 0.071411,
            "releaseScaleReady": False,
            "note": "Selection authorizes only VFAI-014 bootstrap training/evaluation, not a production model claim."
        },
    }
    report["reportSha256"] = _fingerprint(report)
    destination = Path(report_path)
    _atomic_json(destination, report)
    selected_config = {
        "schemaVersion": 1,
        "artifactKind": "vfdlm-gen1-selected-bootstrap-config",
        "selectionStatus": selection["status"],
        "candidateId": selection["selectedCandidateId"],
        "deploymentProfile": selection["deploymentProfile"],
        "parameterCount": selection["parameterCount"],
        "modelConfig": selected_result["modelConfig"],
        "recommendedTrainingControls": {
            "packingBlockSize": plan["controls"]["packingBlockSize"],
            "weightDecay": selected_result["weightDecay"],
            "precision": plan["controls"]["precision"],
            "device": plan["controls"]["device"],
        },
        "evidence": {
            "sweepReportPath": destination.resolve().relative_to(AI_ROOT).as_posix(),
            "sweepReportSha256": report["reportSha256"],
            "sweepReportFileSha256": _sha256_file(destination),
            "planFingerprint": plan["planFingerprint"],
        },
        "releaseApproved": False,
        "nextGate": "VFAI-014",
    }
    selected_config["manifestSha256"] = _fingerprint(selected_config)
    _atomic_json(Path(selected_config_path), selected_config)
    return report


def check_scorecard(
    *,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    selected_config_path: str | Path = DEFAULT_SELECTED_CONFIG_PATH,
) -> dict[str, Any]:
    plan = load_sweep_plan(plan_path)
    report_file = Path(report_path)
    report = _read_json(report_file, "architecture scorecard")
    if report.get("reportSha256") != _fingerprint(
        {key: value for key, value in report.items() if key != "reportSha256"}
    ):
        raise SweepContractError("architecture scorecard checksum mismatch")
    if (
        report.get("planFingerprint") != plan["planFingerprint"]
        or report.get("planFileSha256") != _sha256_file(Path(plan_path))
    ):
        raise SweepContractError("architecture scorecard plan binding changed")
    for result in report.get("scorecard", []):
        _validate_result(result, plan)
    expected_selection = select_candidate(report["scorecard"], plan["selectionPolicy"])
    if report.get("selection") != expected_selection:
        raise SweepContractError("architecture selection no longer follows policy")
    selected = _read_json(Path(selected_config_path), "selected Gen1 config")
    if selected.get("manifestSha256") != _fingerprint(
        {key: value for key, value in selected.items() if key != "manifestSha256"}
    ):
        raise SweepContractError("selected Gen1 config checksum mismatch")
    if (
        selected.get("candidateId") != expected_selection["selectedCandidateId"]
        or selected.get("modelConfig") != report.get("selectedModelConfig")
        or selected.get("releaseApproved") is not False
        or selected.get("evidence", {}).get("sweepReportSha256") != report["reportSha256"]
        or selected.get("evidence", {}).get("sweepReportFileSha256")
        != _sha256_file(report_file)
    ):
        raise SweepContractError("selected Gen1 config evidence binding changed")
    return {
        "decision": "pass",
        "reportId": report["reportId"],
        "selectedCandidateId": expected_selection["selectedCandidateId"],
        "paretoFrontier": expected_selection["paretoFrontier"],
        "candidateCount": len(report["scorecard"]),
    }
