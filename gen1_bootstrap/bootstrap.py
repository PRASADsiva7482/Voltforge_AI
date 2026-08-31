"""Governed bootstrap training, held-out scoring, and best-checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence
import uuid

import torch

from benchmarking.runtime_baseline import PeakMemorySampler
from gen1_sweep import check_scorecard
from gen1_training import (
    CURRENT_TRAINING_TOKENIZER_RELEASE_PATH,
    Gen1Trainer,
    Gen1TrainingConfig,
    PackedCorpus,
    load_approved_corpus,
)
from model.gen1 import Gen1Config, VoltForgeGen1
from model.tokenizer import VoltForgeTokenizer
from task_schema.compiler import compile_task_record
from task_schema.io import read_task_shard


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = AI_ROOT / "gen1_bootstrap" / "plan.v1.json"
DEFAULT_RUN_DIRECTORY = AI_ROOT / "training_runs" / "vfai014-gen1-bootstrap-v1"
DEFAULT_REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "gen1-bootstrap-heldout-v1.json"
DEFAULT_GLOBAL_REPORT_PATH = (
    AI_ROOT / "evaluation" / "reports" / "gen1-bootstrap-global-gates-v1.json"
)
DEFAULT_BEST_CONFIG_PATH = (
    AI_ROOT / "model" / "gen1" / "configs" / "gen1-bootstrap-best-v1.json"
)


class BootstrapContractError(RuntimeError):
    """Raised when bootstrap controls or evidence are incomplete or inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapContractError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BootstrapContractError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BootstrapContractError(f"{label} path is missing")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise BootstrapContractError(f"{label} path escapes the AI workspace") from exc
    return path


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(AI_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_bootstrap_plan(path: str | Path = DEFAULT_PLAN_PATH) -> dict[str, Any]:
    plan_path = Path(path)
    plan = _read_json(plan_path, "Gen1 bootstrap plan")
    expected_keys = {
        "schemaVersion",
        "planId",
        "runId",
        "purpose",
        "selectedConfig",
        "sweepEvidence",
        "controls",
        "heldoutPolicy",
        "stoppingPolicy",
        "globalReleasePolicy",
        "releaseBoundary",
    }
    if set(plan) != expected_keys or plan.get("schemaVersion") != 1:
        raise BootstrapContractError("bootstrap plan keys/schema are invalid")

    selected_contract = plan["selectedConfig"]
    selected_path = _resolve_repo_path(selected_contract.get("path"), "selected config")
    if _sha256_file(selected_path) != selected_contract.get("fileSha256"):
        raise BootstrapContractError("selected VFAI-013 config file checksum changed")
    selected = _read_json(selected_path, "selected VFAI-013 config")
    if selected.get("manifestSha256") != _fingerprint(
        {key: value for key, value in selected.items() if key != "manifestSha256"}
    ):
        raise BootstrapContractError("selected VFAI-013 config manifest checksum changed")
    if (
        selected.get("candidateId") != selected_contract.get("candidateId")
        or selected.get("parameterCount") != selected_contract.get("parameterCount")
        or selected.get("selectionStatus")
        != selected_contract.get("requiredSelectionStatus")
        or selected.get("releaseApproved") is not False
    ):
        raise BootstrapContractError("selected VFAI-013 config contract changed")
    config = Gen1Config.from_dict(selected["modelConfig"])
    if config.parameter_count() != selected_contract["parameterCount"]:
        raise BootstrapContractError("selected model parameter count changed")

    sweep = plan["sweepEvidence"]
    sweep_path = _resolve_repo_path(sweep.get("path"), "sweep evidence")
    if _sha256_file(sweep_path) != sweep.get("fileSha256"):
        raise BootstrapContractError("VFAI-013 sweep report file checksum changed")
    checked_sweep = check_scorecard()
    sweep_report = _read_json(sweep_path, "VFAI-013 sweep report")
    if (
        checked_sweep["selectedCandidateId"] != selected["candidateId"]
        or sweep_report.get("reportSha256") != sweep.get("reportSha256")
    ):
        raise BootstrapContractError("VFAI-013 selection evidence changed")

    controls = plan["controls"]
    expected_control_keys = {
        "approvedDatasetId",
        "tokenizerId",
        "packingBlockSize",
        "seed",
        "maxSteps",
        "microBatchSize",
        "gradientAccumulationSteps",
        "peakLearningRate",
        "minimumLearningRate",
        "warmupSteps",
        "beta1",
        "beta2",
        "adamEpsilon",
        "weightDecay",
        "maxGradientNorm",
        "evaluationInterval",
        "checkpointInterval",
        "precision",
        "device",
        "deterministicAlgorithms",
        "torchNumThreads",
        "torchNumInteropThreads",
        "expectedTrainingPredictedTokens",
        "approvedUniqueTrainingPredictedTokens",
        "maximumExposureMultiple",
    }
    if not isinstance(controls, dict) or set(controls) != expected_control_keys:
        raise BootstrapContractError("bootstrap controls are invalid")
    training = _training_config(config, controls)
    if (
        controls["packingBlockSize"] != config.max_sequence_length
        or controls["maxSteps"] % controls["evaluationInterval"] != 0
        or controls["checkpointInterval"] != controls["evaluationInterval"]
        or controls["expectedTrainingPredictedTokens"]
        > controls["approvedUniqueTrainingPredictedTokens"]
        * controls["maximumExposureMultiple"]
        or training.model.fingerprint() != config.fingerprint()
    ):
        raise BootstrapContractError("bootstrap control relationships are invalid")

    heldout = plan["heldoutPolicy"]
    expected_heldout_keys = {
        "splitId",
        "recordCount",
        "packedPredictedTokens",
        "expectedTasks",
        "outputSectionTypes",
        "selectionMethod",
        "objectives",
        "trainingLossUsedForSelection",
        "tieBreaker",
        "releaseCredit",
    }
    if not isinstance(heldout, dict) or set(heldout) != expected_heldout_keys:
        raise BootstrapContractError("held-out policy is invalid")
    objective_paths = [item.get("path") for item in heldout["objectives"]]
    expected_objectives = {
        "fullValidation.nextTokenLoss",
        "fullValidation.top1TokenAccuracy",
        "outputHeldout.nextTokenLoss",
        "outputHeldout.top1TokenAccuracy",
        "outputHeldout.taskMacroLoss",
        "outputHeldout.worstTaskLoss",
    }
    if (
        set(objective_paths) != expected_objectives
        or len(objective_paths) != len(expected_objectives)
        or any(item.get("direction") not in {"min", "max"} for item in heldout["objectives"])
        or heldout["trainingLossUsedForSelection"] is not False
        or heldout["releaseCredit"] != 0.0
        or len(set(heldout["expectedTasks"])) != len(heldout["expectedTasks"])
    ):
        raise BootstrapContractError("held-out objective contract is invalid")

    global_policy = plan["globalReleasePolicy"]
    metrics_path = _resolve_repo_path(global_policy.get("metricsPath"), "release metrics")
    frozen_path = _resolve_repo_path(
        global_policy.get("frozenManifestPath"), "frozen evaluation manifest"
    )
    if (
        _sha256_file(metrics_path) != global_policy.get("metricsFileSha256")
        or _sha256_file(frozen_path)
        != global_policy.get("frozenManifestFileSha256")
    ):
        raise BootstrapContractError("frozen global release policy checksum changed")
    metrics = _read_json(metrics_path, "release metrics").get("metrics")
    if (
        not isinstance(metrics, list)
        or len(metrics) != global_policy.get("expectedMetricCount")
        or len({item.get("id") for item in metrics}) != len(metrics)
        or global_policy.get("unsupportedCheckpointScore") != 0.0
    ):
        raise BootstrapContractError("global release metric contract is invalid")
    if plan["releaseBoundary"].get("releaseApproved") is not False:
        raise BootstrapContractError("bootstrap plan cannot approve release")
    plan["planFingerprint"] = _fingerprint(plan)
    plan["selectedModelConfig"] = selected["modelConfig"]
    return plan


def _training_config(
    model_config: Gen1Config, controls: Mapping[str, Any]
) -> Gen1TrainingConfig:
    return Gen1TrainingConfig(
        model=model_config,
        seed=int(controls["seed"]),
        max_steps=int(controls["maxSteps"]),
        micro_batch_size=int(controls["microBatchSize"]),
        gradient_accumulation_steps=int(controls["gradientAccumulationSteps"]),
        peak_learning_rate=float(controls["peakLearningRate"]),
        minimum_learning_rate=float(controls["minimumLearningRate"]),
        warmup_steps=int(controls["warmupSteps"]),
        beta1=float(controls["beta1"]),
        beta2=float(controls["beta2"]),
        adam_epsilon=float(controls["adamEpsilon"]),
        weight_decay=float(controls["weightDecay"]),
        max_gradient_norm=float(controls["maxGradientNorm"]),
        validation_interval=int(controls["evaluationInterval"]),
        checkpoint_interval=int(controls["checkpointInterval"]),
        precision=str(controls["precision"]),
        device=str(controls["device"]),
        deterministic_algorithms=bool(controls["deterministicAlgorithms"]),
    )


@dataclass(frozen=True, slots=True)
class HeldoutChunk:
    record_id: str
    task: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    predicted_tokens: int


@dataclass(frozen=True, slots=True)
class PreparedHeldout:
    chunks: tuple[HeldoutChunk, ...]
    record_count_by_task: Mapping[str, int]
    predicted_tokens: int
    fingerprint: str


def _first_output_token_index(
    tokenizer: VoltForgeTokenizer,
    compiled: str,
    output_section_types: Sequence[str],
) -> tuple[list[int], int, int]:
    raw = compiled.encode("utf-8", errors="surrogatepass")
    positions = [
        raw.find(f"<|vf:{section}:v1:".encode("ascii"))
        for section in output_section_types
    ]
    positions = [position for position in positions if position >= 0]
    if not positions:
        raise BootstrapContractError("held-out record has no supervised output section")
    output_byte_offset = min(positions)
    encoded = tokenizer.encode_bytes(raw)
    if tokenizer.decode_bytes(encoded) != raw:
        raise BootstrapContractError("tokenizer did not preserve held-out compiled bytes")
    cumulative = 0
    first_output_encoded_index: int | None = None
    for index, token_id in enumerate(encoded):
        token = tokenizer.token_bytes[token_id]
        if token is None:
            raise BootstrapContractError("held-out text encoded to a special token")
        cumulative += len(token)
        if cumulative > output_byte_offset:
            first_output_encoded_index = index
            break
    if first_output_encoded_index is None:
        raise BootstrapContractError("held-out output boundary is outside encoded text")
    bos = tokenizer.special_token_to_id["[BOS]"]
    eos = tokenizer.special_token_to_id["[EOS]"]
    token_ids = [bos, *encoded, eos]
    return token_ids, first_output_encoded_index + 1, output_byte_offset


def _record_chunks(
    *,
    record_id: str,
    task: str,
    token_ids: Sequence[int],
    first_output_token: int,
    block_size: int,
    pad_token_id: int,
) -> list[HeldoutChunk]:
    chunks: list[HeldoutChunk] = []
    start = 0
    while start < len(token_ids) - 1:
        values = list(token_ids[start : start + block_size])
        if len(values) < 2:
            break
        valid_length = len(values)
        labels = list(values)
        for local_index in range(valid_length):
            global_index = start + local_index
            if global_index < first_output_token:
                labels[local_index] = -100
        padding = block_size - valid_length
        values.extend([pad_token_id] * padding)
        labels.extend([-100] * padding)
        mask = [True] * valid_length + [False] * padding
        target_count = sum(value != -100 for value in labels[1:])
        if target_count:
            chunks.append(
                HeldoutChunk(
                    record_id=record_id,
                    task=task,
                    input_ids=tuple(values),
                    labels=tuple(labels),
                    attention_mask=tuple(mask),
                    predicted_tokens=target_count,
                )
            )
        start += block_size - 1
    return chunks


def prepare_output_heldout(
    corpus: PackedCorpus,
    model_config: Gen1Config,
    heldout_policy: Mapping[str, Any],
    *,
    tokenizer_directory: str | Path = CURRENT_TRAINING_TOKENIZER_RELEASE_PATH,
) -> PreparedHeldout:
    tokenizer = VoltForgeTokenizer(model_config.vocab_size)
    tokenizer.load(Path(tokenizer_directory))
    validation_ids = corpus.manifest.get("validationRecordIds")
    if not isinstance(validation_ids, list) or len(validation_ids) != heldout_policy[
        "recordCount"
    ]:
        raise BootstrapContractError("held-out validation IDs changed")
    wanted = set(validation_ids)
    records: dict[str, dict[str, Any]] = {}
    for descriptor in corpus.manifest["shards"]:
        shard_path = _resolve_repo_path(descriptor["path"], "approved shard")
        for record in read_task_shard(shard_path):
            if record["recordId"] in wanted:
                records[record["recordId"]] = record
    if set(records) != wanted:
        raise BootstrapContractError("held-out records do not match the frozen split")

    all_chunks: list[HeldoutChunk] = []
    task_counts: dict[str, int] = {}
    descriptors = []
    for record_id in sorted(records):
        record = records[record_id]
        task = str(record["task"])
        task_counts[task] = task_counts.get(task, 0) + 1
        compiled = compile_task_record(record)
        token_ids, first_output_token, output_byte_offset = _first_output_token_index(
            tokenizer, compiled, heldout_policy["outputSectionTypes"]
        )
        chunks = _record_chunks(
            record_id=record_id,
            task=task,
            token_ids=token_ids,
            first_output_token=first_output_token,
            block_size=model_config.max_sequence_length,
            pad_token_id=tokenizer.special_token_to_id["[PAD]"],
        )
        if not chunks:
            raise BootstrapContractError("held-out record has no output targets")
        all_chunks.extend(chunks)
        descriptors.append(
            {
                "recordId": record_id,
                "task": task,
                "tokenCount": len(token_ids),
                "firstOutputToken": first_output_token,
                "outputByteOffset": output_byte_offset,
                "outputPredictedTokens": sum(item.predicted_tokens for item in chunks),
            }
        )
    if set(task_counts) != set(heldout_policy["expectedTasks"]):
        raise BootstrapContractError("held-out task coverage changed")
    predicted_tokens = sum(item.predicted_tokens for item in all_chunks)
    prepared_manifest = {
        "schemaVersion": 1,
        "splitId": heldout_policy["splitId"],
        "recordDescriptors": descriptors,
        "recordCountByTask": dict(sorted(task_counts.items())),
        "predictedTokens": predicted_tokens,
        "blockSize": model_config.max_sequence_length,
        "tokenizer": corpus.manifest["tokenizer"],
    }
    return PreparedHeldout(
        chunks=tuple(all_chunks),
        record_count_by_task=dict(sorted(task_counts.items())),
        predicted_tokens=predicted_tokens,
        fingerprint=_fingerprint(prepared_manifest),
    )


@torch.no_grad()
def evaluate_heldout(
    model: VoltForgeGen1,
    corpus: PackedCorpus,
    prepared: PreparedHeldout,
    *,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    full_loss_total = 0.0
    full_targets = 0
    full_correct = 0
    started = time.perf_counter()
    for start in range(0, corpus.validation.block_count, batch_size):
        end = min(start + batch_size, corpus.validation.block_count)
        input_ids = corpus.validation.input_ids[start:end].to(device)
        labels = corpus.validation.labels[start:end].to(device)
        attention_mask = corpus.validation.attention_mask[start:end].to(device)
        output = model(input_ids, labels=labels, attention_mask=attention_mask)
        if output.loss is None or not torch.isfinite(output.loss):
            raise BootstrapContractError("full held-out evaluation produced non-finite loss")
        targets = labels[:, 1:]
        valid = targets != -100
        count = int(valid.sum().item())
        predictions = output.logits[:, :-1].argmax(dim=-1)
        full_loss_total += float(output.loss.item()) * count
        full_targets += count
        full_correct += int((predictions[valid] == targets[valid]).sum().item())

    task_totals: dict[str, dict[str, float | int]] = {}
    output_loss_total = 0.0
    output_targets = 0
    output_correct = 0
    chunks = prepared.chunks
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        input_ids = torch.tensor([item.input_ids for item in batch], dtype=torch.long).to(device)
        labels = torch.tensor([item.labels for item in batch], dtype=torch.long).to(device)
        attention_mask = torch.tensor(
            [item.attention_mask for item in batch], dtype=torch.bool
        ).to(device)
        output = model(input_ids, labels=labels, attention_mask=attention_mask)
        if output.loss is None or not torch.isfinite(output.loss):
            raise BootstrapContractError("output held-out evaluation produced non-finite loss")
        predictions = output.logits[:, :-1].argmax(dim=-1)
        for index, item in enumerate(batch):
            targets = labels[index, 1:]
            valid = targets != -100
            count = int(valid.sum().item())
            token_losses = torch.nn.functional.cross_entropy(
                output.logits[index, :-1][valid],
                targets[valid],
                reduction="sum",
            )
            loss_sum = float(token_losses.item())
            correct = int((predictions[index][valid] == targets[valid]).sum().item())
            totals = task_totals.setdefault(
                item.task, {"lossSum": 0.0, "predictedTokens": 0, "correctTokens": 0}
            )
            totals["lossSum"] = float(totals["lossSum"]) + loss_sum
            totals["predictedTokens"] = int(totals["predictedTokens"]) + count
            totals["correctTokens"] = int(totals["correctTokens"]) + correct
            output_loss_total += loss_sum
            output_targets += count
            output_correct += correct
    if full_targets != corpus.validation.predicted_token_count:
        raise BootstrapContractError("full held-out target accounting changed")
    if output_targets != prepared.predicted_tokens:
        raise BootstrapContractError("output held-out target accounting changed")
    task_metrics = {}
    for task, totals in sorted(task_totals.items()):
        count = int(totals["predictedTokens"])
        loss = float(totals["lossSum"]) / count
        task_metrics[task] = {
            "recordCount": prepared.record_count_by_task[task],
            "predictedTokens": count,
            "nextTokenLoss": loss,
            "perplexity": math.exp(min(loss, 20.0)),
            "top1TokenAccuracy": int(totals["correctTokens"]) / count,
            "correctTokens": int(totals["correctTokens"]),
        }
    task_losses = [value["nextTokenLoss"] for value in task_metrics.values()]
    full_loss = full_loss_total / full_targets
    output_loss = output_loss_total / output_targets
    return {
        "status": "completed",
        "fullValidation": {
            "nextTokenLoss": full_loss,
            "perplexity": math.exp(min(full_loss, 20.0)),
            "top1TokenAccuracy": full_correct / full_targets,
            "correctTokens": full_correct,
            "predictedTokens": full_targets,
        },
        "outputHeldout": {
            "nextTokenLoss": output_loss,
            "perplexity": math.exp(min(output_loss, 20.0)),
            "top1TokenAccuracy": output_correct / output_targets,
            "correctTokens": output_correct,
            "predictedTokens": output_targets,
            "taskMacroLoss": sum(task_losses) / len(task_losses),
            "worstTaskLoss": max(task_losses),
            "taskMetrics": task_metrics,
            "preparedHeldoutFingerprint": prepared.fingerprint,
        },
        "evaluationSeconds": time.perf_counter() - started,
        "releaseCredit": 0.0,
        "classification": "teacher-forced-bootstrap-heldout-not-generation-release-evaluation",
    }


def _path_value(record: Mapping[str, Any], path: str) -> float:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise BootstrapContractError(f"checkpoint score omits objective {path}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise BootstrapContractError(f"checkpoint objective {path} is not numeric")
    value = float(current)
    if not math.isfinite(value):
        raise BootstrapContractError(f"checkpoint objective {path} is not finite")
    return value


def _objective_vector(
    record: Mapping[str, Any], objectives: Sequence[Mapping[str, Any]]
) -> tuple[float, ...]:
    return tuple(
        _path_value(record, item["path"])
        * (1.0 if item["direction"] == "min" else -1.0)
        for item in objectives
    )


def pareto_checkpoint_steps(
    records: Sequence[Mapping[str, Any]], objectives: Sequence[Mapping[str, Any]]
) -> list[int]:
    completed = [item for item in records if item.get("status") == "completed"]
    frontier: list[int] = []
    for candidate in completed:
        vector = _objective_vector(candidate, objectives)
        dominated = False
        for other in completed:
            if other["step"] == candidate["step"]:
                continue
            other_vector = _objective_vector(other, objectives)
            if all(left <= right for left, right in zip(other_vector, vector)) and any(
                left < right for left, right in zip(other_vector, vector)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(int(candidate["step"]))
    return sorted(frontier)


def select_best_checkpoint(
    records: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    completed = [item for item in records if item.get("status") == "completed"]
    objectives = policy["objectives"]
    frontier = set(pareto_checkpoint_steps(completed, objectives))
    if not completed or not frontier:
        raise BootstrapContractError("no completed checkpoint evaluation is selectable")
    objective_ranks: dict[int, dict[str, int]] = {int(item["step"]): {} for item in completed}
    for objective in objectives:
        path = objective["path"]
        direction = objective["direction"]
        values = sorted(
            {_path_value(item, path) for item in completed}, reverse=direction == "max"
        )
        ranks = {value: index + 1 for index, value in enumerate(values)}
        for item in completed:
            objective_ranks[int(item["step"])][path] = ranks[_path_value(item, path)]
    scores = []
    for item in completed:
        step = int(item["step"])
        ranks = objective_ranks[step]
        scores.append(
            {
                "step": step,
                "onParetoFrontier": step in frontier,
                "objectiveRanks": ranks,
                "meanObjectiveRank": sum(ranks.values()) / len(ranks),
            }
        )
    selected_score = min(
        (item for item in scores if item["onParetoFrontier"]),
        key=lambda item: (item["meanObjectiveRank"], item["step"]),
    )
    return {
        "selectedStep": selected_score["step"],
        "paretoFrontierSteps": sorted(frontier),
        "selectionMethod": policy["selectionMethod"],
        "objectives": objectives,
        "trainingLossUsedForSelection": False,
        "checkpointRanks": sorted(scores, key=lambda item: item["step"]),
        "tieBreaker": policy["tieBreaker"],
        "releaseCredit": 0.0,
        "status": "best-experimental-bootstrap-checkpoint-not-release",
    }


def _deterioration_stop(
    records: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> bool:
    patience = int(policy["deteriorationPatienceIntervals"])
    if (
        len(records) <= patience
        or int(records[-1]["step"]) < int(policy["minimumStepForDeteriorationStop"])
    ):
        return False
    prior = records[:-patience]
    recent = records[-patience:]
    factor = 1.0 + float(policy["relativeLossRegressionPercent"]) / 100.0
    best_full = min(item["fullValidation"]["nextTokenLoss"] for item in prior)
    best_output = min(item["outputHeldout"]["nextTokenLoss"] for item in prior)
    return all(
        item["fullValidation"]["nextTokenLoss"] > best_full * factor
        and item["outputHeldout"]["nextTokenLoss"] > best_output * factor
        for item in recent
    )


def _set_threads(controls: Mapping[str, Any]) -> None:
    torch.set_num_threads(int(controls["torchNumThreads"]))
    requested = int(controls["torchNumInteropThreads"])
    if torch.get_num_interop_threads() != requested:
        try:
            torch.set_num_interop_threads(requested)
        except RuntimeError as exc:
            raise BootstrapContractError(
                "Torch interop threads were initialized before bootstrap controls"
            ) from exc


def _checkpoint_descriptor(
    trainer: Gen1Trainer, checkpoint: Path, quality: Mapping[str, Any]
) -> dict[str, Any]:
    entry = trainer.checkpoints[-1]
    metric = trainer.metrics[-1]
    manifest_path = checkpoint / "checkpoint-manifest.json"
    return {
        "step": trainer.global_step,
        "status": "completed",
        "checkpoint": {
            "path": _portable_path(checkpoint),
            "bytes": _directory_size(checkpoint),
            "checkpointManifestFileSha256": _sha256_file(manifest_path),
            "checkpointManifestSha256": _read_json(
                manifest_path, "checkpoint manifest"
            )["manifestSha256"],
            "modelWeightsSha256": entry["modelWeightsSha256"],
            "trainingStateSha256": entry["trainingStateSha256"],
        },
        "trainingSnapshot": {
            "trainingLoss": metric["trainingLoss"],
            "learningRate": metric["learningRate"],
            "gradientNormBeforeClip": metric["gradientNormBeforeClip"],
            "stepSeconds": metric["stepSeconds"],
            "stepTokensPerSecond": metric["tokensPerSecond"],
            "cumulativePredictedTokens": trainer.tokens_seen,
            "dataEpoch": trainer.stream.epoch,
            "dataPosition": trainer.stream.position,
        },
        **quality,
    }


def _pointer_payload(
    *,
    kind: str,
    plan: Mapping[str, Any],
    run_directory: Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schemaVersion": 1,
        "artifactKind": kind,
        "planId": plan["planId"],
        "planFingerprint": plan["planFingerprint"],
        "runId": plan["runId"],
        "runDirectory": _portable_path(run_directory),
        "step": record["step"],
        "checkpoint": record["checkpoint"],
        "releaseApproved": False,
    }
    value["manifestSha256"] = _fingerprint(value)
    return value


def _global_gate_evidence(
    plan: Mapping[str, Any], global_report_path: Path
) -> dict[str, Any]:
    from evaluation.release_gate import build_evaluation_report

    release_report = build_evaluation_report()
    _atomic_json(global_report_path, release_report)
    policy_path = _resolve_repo_path(
        plan["globalReleasePolicy"]["metricsPath"], "release metrics"
    )
    metric_policy = _read_json(policy_path, "release metrics")
    expected = {item["id"]: item for item in metric_policy["metrics"]}
    actual = {item["id"]: item for item in release_report["metrics"]}
    if set(actual) != set(expected):
        raise BootstrapContractError("global held-out metric coverage changed")
    status = plan["globalReleasePolicy"]["checkpointGenerationStatus"]
    score = plan["globalReleasePolicy"]["unsupportedCheckpointScore"]
    gates = []
    for metric_id in expected:
        policy = expected[metric_id]
        observed = actual[metric_id]
        gates.append(
            {
                "id": metric_id,
                "task": policy["task"],
                "minimumScore": policy["minimumScore"],
                "criticalFailureRule": policy["criticalFailureRule"],
                "currentDeterministicSystem": {
                    "score": observed["score"],
                    "thresholdMet": observed["thresholdMet"],
                    "passedCases": observed["passedCases"],
                    "failedCases": observed["failedCases"],
                    "fallbackCases": observed["fallbackCases"],
                    "unsupportedCases": observed["unsupportedCases"],
                },
                "bestNeuralCheckpoint": {
                    "status": status,
                    "score": score,
                    "thresholdMet": False,
                    "releaseCredit": 0.0,
                    "reason": "VFAI-014 has no approved generation runtime, prompt compiler, output gate, or artifact activation path; deterministic-system credit cannot be assigned to raw decoder weights.",
                },
            }
        )
    return {
        "reportPath": _portable_path(global_report_path),
        "reportFileSha256": _sha256_file(global_report_path),
        "suite": release_report["suite"],
        "currentDeterministicSystemSummary": release_report["summary"],
        "bestCheckpointMetricCount": len(gates),
        "bestCheckpointPassedMetrics": 0,
        "bestCheckpointReleaseDecision": "blocked",
        "gates": gates,
    }


def run_bootstrap(
    *,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    run_directory: str | Path = DEFAULT_RUN_DIRECTORY,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    global_report_path: str | Path = DEFAULT_GLOBAL_REPORT_PATH,
    best_config_path: str | Path = DEFAULT_BEST_CONFIG_PATH,
) -> dict[str, Any]:
    plan = load_bootstrap_plan(plan_path)
    controls = plan["controls"]
    _set_threads(controls)
    model_config = Gen1Config.from_dict(plan["selectedModelConfig"])
    corpus = load_approved_corpus(
        model_config, packing_block_size=int(controls["packingBlockSize"])
    )
    if (
        corpus.manifest["datasetId"] != controls["approvedDatasetId"]
        or corpus.manifest["tokenizer"]["tokenizerId"] != controls["tokenizerId"]
        or corpus.train.predicted_token_count
        != controls["approvedUniqueTrainingPredictedTokens"]
        or corpus.validation.predicted_token_count
        != plan["heldoutPolicy"]["packedPredictedTokens"]
    ):
        raise BootstrapContractError("approved bootstrap corpus controls changed")
    prepared = prepare_output_heldout(corpus, model_config, plan["heldoutPolicy"])
    target = Path(run_directory).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise BootstrapContractError(f"bootstrap run directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    training_config = _training_config(model_config, controls)
    trainer = Gen1Trainer.create(
        training_config,
        corpus,
        target,
        run_id=plan["runId"],
    )
    initial_heldout = evaluate_heldout(
        trainer.model,
        corpus,
        prepared,
        batch_size=training_config.micro_batch_size,
    )
    checkpoint_records: list[dict[str, Any]] = []
    stop_reason: str | None = None
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    with PeakMemorySampler() as memory:
        for step in range(
            controls["evaluationInterval"],
            controls["maxSteps"] + 1,
            controls["evaluationInterval"],
        ):
            result = trainer.train(until_step=step)
            checkpoint = Path(result["lastCheckpoint"])
            quality = evaluate_heldout(
                trainer.model,
                corpus,
                prepared,
                batch_size=training_config.micro_batch_size,
            )
            checkpoint_records.append(
                _checkpoint_descriptor(trainer, checkpoint, quality)
            )
            if _deterioration_stop(checkpoint_records, plan["stoppingPolicy"]):
                stop_reason = "heldout-loss-deterioration"
                break
    training_wall_seconds = time.perf_counter() - wall_started
    process_cpu_seconds = time.process_time() - cpu_started
    if trainer.tokens_seen > controls["expectedTrainingPredictedTokens"]:
        raise BootstrapContractError("bootstrap exceeded its controlled token ceiling")
    if trainer.global_step == controls["maxSteps"]:
        if trainer.tokens_seen != controls["expectedTrainingPredictedTokens"]:
            raise BootstrapContractError("bootstrap target-token accounting changed")
        stop_reason = "approved-repeated-data-exposure-ceiling"
    if stop_reason is None:
        raise BootstrapContractError("bootstrap ended without a governed stopping reason")

    selection = select_best_checkpoint(checkpoint_records, plan["heldoutPolicy"])
    best = next(
        item for item in checkpoint_records if item["step"] == selection["selectedStep"]
    )
    last = checkpoint_records[-1]
    best_pointer = _pointer_payload(
        kind="vfdlm-gen1-best-bootstrap-checkpoint-pointer",
        plan=plan,
        run_directory=target,
        record=best,
    )
    last_pointer = _pointer_payload(
        kind="vfdlm-gen1-last-bootstrap-checkpoint-pointer",
        plan=plan,
        run_directory=target,
        record=last,
    )
    best_pointer_path = target / "best-checkpoint.json"
    last_pointer_path = target / "last-checkpoint.json"
    _atomic_json(best_pointer_path, best_pointer)
    _atomic_json(last_pointer_path, last_pointer)
    global_evidence = _global_gate_evidence(plan, Path(global_report_path))
    run_manifest_path = target / "run-manifest.json"
    run_manifest = _read_json(run_manifest_path, "bootstrap run manifest")
    report = {
        "schemaVersion": 1,
        "reportId": "vfai014-gen1-bootstrap-heldout-v1",
        "generatedAtUtc": _utc_now(),
        "planId": plan["planId"],
        "planPath": _portable_path(Path(plan_path)),
        "planFileSha256": _sha256_file(Path(plan_path)),
        "planFingerprint": plan["planFingerprint"],
        "run": {
            "runId": plan["runId"],
            "runDirectory": _portable_path(target),
            "runManifestPath": _portable_path(run_manifest_path),
            "runManifestFileSha256": _sha256_file(run_manifest_path),
            "runManifestSha256": run_manifest["manifestSha256"],
            "status": run_manifest["status"],
            "globalStep": trainer.global_step,
            "predictedTokens": trainer.tokens_seen,
            "uniqueTrainingPredictedTokens": corpus.train.predicted_token_count,
            "exposureMultiple": trainer.tokens_seen / corpus.train.predicted_token_count,
            "trainingWallSeconds": training_wall_seconds,
            "processCpuSeconds": process_cpu_seconds,
            "predictedTokensPerWallSecond": trainer.tokens_seen
            / max(training_wall_seconds, 1e-12),
            "peakMemory": memory.report(),
            "trainingConfig": training_config.to_dict(),
            "trainingConfigFingerprint": training_config.fingerprint(),
            "datasetFingerprint": corpus.fingerprint,
            "codeRevision": run_manifest["codeRevision"],
            "hardware": run_manifest["hardware"],
        },
        "heldoutBoundary": {
            "splitId": plan["heldoutPolicy"]["splitId"],
            "recordCount": plan["heldoutPolicy"]["recordCount"],
            "packedPredictedTokens": corpus.validation.predicted_token_count,
            "outputPredictedTokens": prepared.predicted_tokens,
            "taskCoverage": sorted(prepared.record_count_by_task),
            "recordCountByTask": prepared.record_count_by_task,
            "preparedHeldoutFingerprint": prepared.fingerprint,
            "classification": "frozen-VFAI-010-validation-proxy-not-VFAI-005-generation-credit",
        },
        "initialHeldout": initial_heldout,
        "checkpointScorecard": checkpoint_records,
        "selection": selection,
        "bestCheckpoint": best["checkpoint"],
        "lastCheckpoint": last["checkpoint"],
        "checkpointPointers": {
            "best": {
                "path": _portable_path(best_pointer_path),
                "fileSha256": _sha256_file(best_pointer_path),
                "manifestSha256": best_pointer["manifestSha256"],
            },
            "last": {
                "path": _portable_path(last_pointer_path),
                "fileSha256": _sha256_file(last_pointer_path),
                "manifestSha256": last_pointer["manifestSha256"],
            },
        },
        "globalHeldoutScorecard": global_evidence,
        "stoppingDecision": {
            "reason": stop_reason,
            "continueTrainingApproved": False,
            "dataComputeAssumption": "invalid-for-release-scaling",
            "explanation": plan["stoppingPolicy"]["scalingInterpretation"],
            "nextGate": plan["releaseBoundary"]["nextDecisionGate"],
        },
        "limitations": [
            "Only 138 approved training records and 71,377 unique next-token transitions are available; repeated exposure is not new data.",
            "The frozen model-validation split has only 16 records across nine synthetic task strata, so checkpoint differences are bootstrap evidence rather than release-quality estimates.",
            "The decoder has a 128-token context and has not yet been connected to the VFAI-017 local inference runtime, VFAI-019 output gates, or VFAI-020 project-context compiler.",
            "All 13 VFAI-005 generation correctness, grounding, privacy, refusal, malformed-input, and adversarial gates assign the raw checkpoint zero neural credit until an approved runtime can execute them.",
            "Simulation interpretation, search grounding, memory isolation, malformed-input rejection, and adversarial request handling are not learned capabilities demonstrated by this checkpoint.",
            "Training was measured only in deterministic CPU float32; no calibrated energy sensor or supported accelerator was available.",
            "The best checkpoint is experimental, is not active in the service, and cannot be packaged or promoted before VFAI-015 and VFAI-016.",
        ],
        "releaseBoundary": {
            "artifactStatus": plan["releaseBoundary"]["artifactStatus"],
            "releaseApproved": False,
            "releaseCredit": 0.0,
            "globalMetricThresholdsMet": False,
            "activeRuntimeChanged": False,
            "nextDecisionGate": plan["releaseBoundary"]["nextDecisionGate"],
        },
        "runtime": {
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
        },
    }
    report["reportSha256"] = _fingerprint(report)
    destination = Path(report_path)
    _atomic_json(destination, report)
    best_config = {
        "schemaVersion": 1,
        "artifactKind": "vfdlm-gen1-best-bootstrap-selection",
        "selectionStatus": selection["status"],
        "runId": plan["runId"],
        "step": best["step"],
        "candidateId": plan["selectedConfig"]["candidateId"],
        "parameterCount": plan["selectedConfig"]["parameterCount"],
        "modelConfig": plan["selectedModelConfig"],
        "checkpoint": best["checkpoint"],
        "evidence": {
            "reportPath": _portable_path(destination),
            "reportSha256": report["reportSha256"],
            "reportFileSha256": _sha256_file(destination),
            "planFingerprint": plan["planFingerprint"],
            "selectionMethod": selection["selectionMethod"],
        },
        "releaseApproved": False,
        "nextGate": plan["releaseBoundary"]["nextDecisionGate"],
    }
    best_config["manifestSha256"] = _fingerprint(best_config)
    _atomic_json(Path(best_config_path), best_config)
    return report


def check_bootstrap_scorecard(
    *,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    global_report_path: str | Path = DEFAULT_GLOBAL_REPORT_PATH,
    best_config_path: str | Path = DEFAULT_BEST_CONFIG_PATH,
    require_artifacts: bool = False,
) -> dict[str, Any]:
    plan = load_bootstrap_plan(plan_path)
    report_file = Path(report_path)
    report = _read_json(report_file, "Gen1 bootstrap held-out scorecard")
    if report.get("reportSha256") != _fingerprint(
        {key: value for key, value in report.items() if key != "reportSha256"}
    ):
        raise BootstrapContractError("bootstrap scorecard checksum mismatch")
    if (
        report.get("planFingerprint") != plan["planFingerprint"]
        or report.get("planFileSha256") != _sha256_file(Path(plan_path))
    ):
        raise BootstrapContractError("bootstrap scorecard plan binding changed")
    scorecard = report.get("checkpointScorecard")
    expected_steps = list(
        range(
            plan["controls"]["evaluationInterval"],
            report["run"]["globalStep"] + 1,
            plan["controls"]["evaluationInterval"],
        )
    )
    if (
        not isinstance(scorecard, list)
        or [item.get("step") for item in scorecard] != expected_steps
        or report["run"]["predictedTokens"]
        > plan["controls"]["expectedTrainingPredictedTokens"]
    ):
        raise BootstrapContractError("bootstrap checkpoint scorecard intervals changed")
    expected_selection = select_best_checkpoint(scorecard, plan["heldoutPolicy"])
    if report.get("selection") != expected_selection:
        raise BootstrapContractError("best checkpoint no longer follows frozen policy")
    metric_policy = _read_json(
        _resolve_repo_path(
            plan["globalReleasePolicy"]["metricsPath"], "release metrics"
        ),
        "release metrics",
    )
    expected_metric_ids = [item["id"] for item in metric_policy["metrics"]]
    gates = report.get("globalHeldoutScorecard", {}).get("gates")
    if (
        not isinstance(gates, list)
        or [item.get("id") for item in gates] != expected_metric_ids
        or any(
            item.get("bestNeuralCheckpoint", {}).get("score") != 0.0
            or item.get("bestNeuralCheckpoint", {}).get("releaseCredit") != 0.0
            or item.get("bestNeuralCheckpoint", {}).get("thresholdMet") is not False
            for item in gates
        )
    ):
        raise BootstrapContractError("global neural checkpoint gate coverage changed")
    global_file = Path(global_report_path)
    if (
        _sha256_file(global_file)
        != report["globalHeldoutScorecard"]["reportFileSha256"]
    ):
        raise BootstrapContractError("global held-out report checksum changed")
    selected = _read_json(Path(best_config_path), "best Gen1 bootstrap config")
    if selected.get("manifestSha256") != _fingerprint(
        {key: value for key, value in selected.items() if key != "manifestSha256"}
    ):
        raise BootstrapContractError("best Gen1 bootstrap config checksum mismatch")
    if (
        selected.get("step") != expected_selection["selectedStep"]
        or selected.get("checkpoint") != report.get("bestCheckpoint")
        or selected.get("modelConfig") != plan["selectedModelConfig"]
        or selected.get("evidence", {}).get("reportSha256") != report["reportSha256"]
        or selected.get("evidence", {}).get("reportFileSha256")
        != _sha256_file(report_file)
        or selected.get("releaseApproved") is not False
        or report.get("releaseBoundary", {}).get("releaseApproved") is not False
    ):
        raise BootstrapContractError("best Gen1 bootstrap evidence binding changed")
    if require_artifacts:
        for label, checkpoint in (
            ("best", report["bestCheckpoint"]),
            ("last", report["lastCheckpoint"]),
        ):
            path = _resolve_repo_path(checkpoint["path"], f"{label} checkpoint")
            if (
                not path.is_dir()
                or _directory_size(path) != checkpoint["bytes"]
                or _sha256_file(path / "checkpoint-manifest.json")
                != checkpoint["checkpointManifestFileSha256"]
                or _sha256_file(path / "model" / "weights.pt")
                != checkpoint["modelWeightsSha256"]
                or _sha256_file(path / "training-state.pt")
                != checkpoint["trainingStateSha256"]
            ):
                raise BootstrapContractError(f"{label} checkpoint artifact changed")
    return {
        "decision": "pass",
        "reportId": report["reportId"],
        "selectedStep": expected_selection["selectedStep"],
        "lastStep": scorecard[-1]["step"],
        "paretoFrontierSteps": expected_selection["paretoFrontierSteps"],
        "globalMetricCount": len(gates),
        "releaseApproved": False,
        "artifactsChecked": require_artifacts,
    }
