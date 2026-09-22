"""Real AdamW optimization, validation, and exact resumable checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import time
from typing import Any, Mapping
import uuid

import numpy as np
import torch
from torch import Tensor

from model.gen1 import DEFAULT_ALLOCATION_LIMIT, VoltForgeGen1

from .config import Gen1TrainingConfig, TRAINING_PIPELINE_ID
from .data import PackedBatchStream, PackedCorpus, TrainingDataContractError


AI_ROOT = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
MAXIMUM_OVERFLOW_RETRIES_PER_STEP = 8
SOURCE_PATHS = (
    "evaluation/frozen-manifest.v1.json",
    "evaluation/metrics.v1.json",
    "gen1_bootstrap/plan.v1.json",
    "gen1_bootstrap/bootstrap.py",
    "gen1_sweep/plan.v1.json",
    "gen1_sweep/sweep.py",
    "gen1_training/config.py",
    "gen1_training/data.py",
    "gen1_training/trainer.py",
    "model/gen1/config.py",
    "model/gen1/model.py",
    "model/tokenizer.py",
    "task_schema/compiler.py",
    "tools/train_gen1_bootstrap.py",
    "tools/run_gen1_sweep.py",
    "tools/train_gen1.py",
)


class TrainingRunError(RuntimeError):
    """Raised when a run cannot be created or continued safely."""


class TrainingCheckpointError(TrainingRunError):
    """Raised when a resumable checkpoint fails integrity/compatibility checks."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_json_bytes(value))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingCheckpointError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingCheckpointError(f"{label} must be a JSON object")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_revision() -> dict[str, Any]:
    files = []
    for relative in SOURCE_PATHS:
        path = AI_ROOT / relative
        if not path.is_file():
            raise TrainingRunError(f"training source file is missing: {relative}")
        files.append({"path": relative, **_file_descriptor(path)})
    source_fingerprint = _sha256_bytes(_canonical_json(files).encode("utf-8"))
    commit = None
    dirty = None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=AI_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=AI_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = commit_result.stdout.strip()
        dirty = bool(status_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "gitCommit": commit,
        "workingTreeDirty": dirty,
        "sourceFingerprint": source_fingerprint,
        "files": files,
    }


def _hardware_manifest(device: torch.device, precision: str) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logicalCpuCount": os.cpu_count(),
        "pythonVersion": platform.python_version(),
        "torchVersion": torch.__version__,
        "device": str(device),
        "precision": precision,
        "cudaAvailable": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        descriptor.update(
            {
                "cudaVersion": torch.version.cuda,
                "deviceName": torch.cuda.get_device_name(device),
                "deviceCapability": list(torch.cuda.get_device_capability(device)),
            }
        )
    elif device.type == "cpu":
        descriptor["cpuCapability"] = torch.backends.cpu.get_cpu_capability()
    return descriptor


def _resolve_device(policy: str) -> torch.device:
    if policy == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if policy == "cuda" and not torch.cuda.is_available():
        raise TrainingRunError("CUDA was requested but is unavailable")
    return torch.device(policy)


def _resolve_precision(policy: str, device: torch.device) -> str:
    if policy == "auto":
        if device.type == "cuda":
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "float32"
    if policy == "float16" and device.type != "cuda":
        raise TrainingRunError("float16 training is supported only on CUDA")
    if policy == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise TrainingRunError("bfloat16 was requested but the CUDA device does not support it")
        if device.type == "cpu" and not torch.cpu._is_avx512_bf16_supported():
            raise TrainingRunError(
                "bfloat16 CPU training was requested without native AVX-512 BF16 capability"
            )
    return policy


def _seed_all(seed: int, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms


def _capture_rng_state() -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        "python": {
            "version": python_state[0],
            "state": list(python_state[1]),
            "gaussian": python_state[2],
        },
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64), dtype=torch.int64),
            "position": numpy_state[2],
            "hasGaussian": numpy_state[3],
            "cachedGaussian": numpy_state[4],
        },
        "torchCpu": torch.random.get_rng_state(),
        "torchCuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    try:
        python_state = state["python"]
        numpy_state = state["numpy"]
        random.setstate(
            (
                int(python_state["version"]),
                tuple(int(value) for value in python_state["state"]),
                python_state["gaussian"],
            )
        )
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["hasGaussian"]),
                float(numpy_state["cachedGaussian"]),
            )
        )
        torch.random.set_rng_state(state["torchCpu"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torchCuda"]])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise TrainingCheckpointError("checkpoint RNG state is invalid") from exc


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, config: Gen1TrainingConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.step_index = 0
        self._set_lr(config.learning_rate_for_step(0))

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> None:
        self.step_index += 1
        if self.step_index < self.config.max_steps:
            self._set_lr(self.config.learning_rate_for_step(self.step_index))

    def state_dict(self) -> dict[str, Any]:
        return {"schemaVersion": 1, "stepIndex": self.step_index, "currentLr": self.current_lr}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"schemaVersion", "stepIndex", "currentLr"}:
            raise TrainingCheckpointError("scheduler state keys are invalid")
        step_index = state["stepIndex"]
        if state["schemaVersion"] != 1 or not isinstance(step_index, int) or not (
            0 <= step_index <= self.config.max_steps
        ):
            raise TrainingCheckpointError("scheduler state is invalid")
        expected_lr = self.config.learning_rate_for_step(
            min(step_index, self.config.max_steps - 1)
        )
        if abs(float(state["currentLr"]) - expected_lr) > 1e-15:
            raise TrainingCheckpointError("scheduler learning rate does not match its step")
        self.step_index = step_index
        self._set_lr(float(state["currentLr"]))


def _build_optimizer(
    model: VoltForgeGen1, config: Gen1TrainingConfig
) -> torch.optim.AdamW:
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.peak_learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,
    )


class Gen1Trainer:
    """One deterministic Gen1 training run with exact safe-point resume."""

    def __init__(self) -> None:
        raise TypeError("Use Gen1Trainer.create() or Gen1Trainer.resume()")

    @classmethod
    def create(
        cls,
        config: Gen1TrainingConfig,
        corpus: PackedCorpus,
        run_directory: str | Path,
        *,
        run_id: str,
        allocation_limit: int = DEFAULT_ALLOCATION_LIMIT,
        allow_test_corpus: bool = False,
    ) -> "Gen1Trainer":
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise TrainingRunError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}")
        if corpus.corpus_kind != "approved-vfai009" and not allow_test_corpus:
            raise TrainingRunError("production training requires the approved VFAI-009 corpus")
        expected_corpus_fingerprint = _sha256_bytes(
            _canonical_json(corpus.manifest).encode("utf-8")
        )
        if corpus.fingerprint != expected_corpus_fingerprint:
            raise TrainingRunError("packed corpus manifest fingerprint is invalid")
        corpus.train.validate_integrity()
        corpus.validation.validate_integrity()
        if (
            corpus.train.block_size > config.model.max_sequence_length
            or corpus.validation.block_size != corpus.train.block_size
        ):
            raise TrainingRunError("packed block size is incompatible with the model context")
        for split in (corpus.train, corpus.validation):
            valid_tokens = split.input_ids[split.attention_mask]
            if torch.any(valid_tokens >= config.model.vocab_size):
                raise TrainingRunError("packed corpus contains a token outside the model vocabulary")
        if corpus.corpus_kind == "approved-vfai009" and (
            corpus.manifest.get("approvalStatus") != "approved"
            or corpus.manifest.get("datasetId") != "vfai009-approved-packed-v1"
            or not isinstance(corpus.manifest.get("tokenizer"), Mapping)
        ):
            raise TrainingRunError("approved corpus metadata is incomplete")
        target = Path(run_directory)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise TrainingRunError(f"new run directory must be absent or empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
        device = _resolve_device(config.device)
        precision = _resolve_precision(config.precision, device)
        _seed_all(config.seed, config.deterministic_algorithms)
        model = VoltForgeGen1.for_training(
            config.model, device=device, allocation_limit=allocation_limit
        )
        instance = object.__new__(cls)
        instance._initialize_common(
            config=config,
            corpus=corpus,
            run_directory=target,
            run_id=run_id,
            device=device,
            precision=precision,
            model=model,
            allocation_limit=allocation_limit,
        )
        instance.started_at_utc = _utc_now()
        instance.accumulated_elapsed_seconds = 0.0
        instance.global_step = 0
        instance.tokens_seen = 0
        instance.metrics: list[dict[str, Any]] = []
        instance.checkpoints: list[dict[str, Any]] = []
        instance.code_revision = _source_revision()
        instance.hardware = _hardware_manifest(device, precision)
        instance._write_run_manifest(status="running")
        return instance

    def _initialize_common(
        self,
        *,
        config: Gen1TrainingConfig,
        corpus: PackedCorpus,
        run_directory: Path,
        run_id: str,
        device: torch.device,
        precision: str,
        model: VoltForgeGen1,
        allocation_limit: int,
    ) -> None:
        self.config = config
        self.corpus = corpus
        self.run_directory = run_directory
        self.run_id = run_id
        self.device = device
        self.precision = precision
        self.model = model
        self.allocation_limit = allocation_limit
        self.optimizer = _build_optimizer(model, config)
        self.scheduler = WarmupCosineScheduler(self.optimizer, config)
        self.scaler = torch.amp.GradScaler(
            device=device.type,
            enabled=precision == "float16",
        )
        self.stream = PackedBatchStream(
            corpus.train, config.micro_batch_size, config.seed + 1
        )
        self._session_started = time.perf_counter()

    def _autocast(self):
        if self.precision == "float32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bfloat16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _elapsed_seconds(self) -> float:
        return self.accumulated_elapsed_seconds + (time.perf_counter() - self._session_started)

    def _run_manifest_payload(self, status: str) -> dict[str, Any]:
        payload = {
            "schemaVersion": 1,
            "pipelineId": TRAINING_PIPELINE_ID,
            "runId": self.run_id,
            "status": status,
            "startedAtUtc": self.started_at_utc,
            "updatedAtUtc": _utc_now(),
            "completedAtUtc": _utc_now() if status == "completed" else None,
            "elapsedSeconds": round(self._elapsed_seconds(), 6),
            "globalStep": self.global_step,
            "tokensSeen": self.tokens_seen,
            "trainingConfig": self.config.to_dict(),
            "trainingConfigFingerprint": self.config.fingerprint(),
            "modelParameterCount": self.config.model.parameter_count(),
            "dataset": dict(self.corpus.manifest),
            "datasetFingerprint": self.corpus.fingerprint,
            "codeRevision": self.code_revision,
            "hardware": self.hardware,
            "metrics": list(self.metrics),
            "checkpoints": list(self.checkpoints),
        }
        payload["manifestSha256"] = _sha256_bytes(
            _canonical_json(payload).encode("utf-8")
        )
        return payload

    def _write_run_manifest(self, status: str) -> None:
        _atomic_write_json(
            self.run_directory / "run-manifest.json",
            self._run_manifest_payload(status),
        )

    def _next_accumulated_batches(self) -> list[tuple[Tensor, Tensor, Tensor, int]]:
        batches = []
        for _ in range(self.config.gradient_accumulation_steps):
            input_ids, labels, attention_mask = self.stream.next_batch(self.device)
            target_count = int((labels[:, 1:] != -100).sum().item())
            if target_count < 1:
                raise TrainingRunError("training batch contains no next-token targets")
            batches.append((input_ids, labels, attention_mask, target_count))
        return batches

    def _train_one_step(self) -> dict[str, Any]:
        started = time.perf_counter()
        overflow_retries = 0
        while True:
            stream_state = self.stream.state_dict()
            rng_state = _capture_rng_state()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            batches = self._next_accumulated_batches()
            total_targets = sum(batch[3] for batch in batches)
            weighted_loss = 0.0
            for input_ids, labels, attention_mask, target_count in batches:
                with self._autocast():
                    output = self.model(
                        input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise TrainingRunError(
                            "training produced a missing or non-finite real loss"
                        )
                    contribution = output.loss * (target_count / total_targets)
                weighted_loss += float(output.loss.detach().item()) * target_count
                self.scaler.scale(contribution).backward()

            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_gradient_norm,
                error_if_nonfinite=not self.scaler.is_enabled(),
            )
            learning_rate = self.scheduler.current_lr
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            current_scale = self.scaler.get_scale()
            optimizer_updated = not self.scaler.is_enabled() or current_scale >= previous_scale
            if optimizer_updated:
                break

            overflow_retries += 1
            self.optimizer.zero_grad(set_to_none=True)
            try:
                self.stream.load_state_dict(stream_state)
                _restore_rng_state(rng_state)
            except (TrainingDataContractError, TrainingCheckpointError) as exc:
                raise TrainingRunError(
                    "mixed-precision overflow could not restore the safe-point state"
                ) from exc
            if overflow_retries >= MAXIMUM_OVERFLOW_RETRIES_PER_STEP:
                raise TrainingRunError(
                    "mixed-precision overflow retry limit reached before an optimizer update"
                )

        self.global_step += 1
        self.tokens_seen += total_targets
        self.scheduler.step()
        duration = time.perf_counter() - started
        return {
            "step": self.global_step,
            "trainingLoss": weighted_loss / total_targets,
            "learningRate": learning_rate,
            "gradientNormBeforeClip": float(gradient_norm.detach().cpu().item()),
            "predictedTokens": total_targets,
            "stepSeconds": duration,
            "tokensPerSecond": total_targets / max(duration, 1e-12),
            "mixedPrecisionOverflowRetries": overflow_retries,
            "gradientScaleBefore": previous_scale if self.scaler.is_enabled() else None,
            "gradientScaleAfter": current_scale if self.scaler.is_enabled() else None,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_targets = 0
        batch_size = self.config.micro_batch_size
        started = time.perf_counter()
        for start in range(0, self.corpus.validation.block_count, batch_size):
            end = min(start + batch_size, self.corpus.validation.block_count)
            input_ids = self.corpus.validation.input_ids[start:end].to(self.device)
            labels = self.corpus.validation.labels[start:end].to(self.device)
            attention_mask = self.corpus.validation.attention_mask[start:end].to(self.device)
            target_count = int((labels[:, 1:] != -100).sum().item())
            with self._autocast():
                output = self.model(
                    input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
            if output.loss is None or not torch.isfinite(output.loss):
                raise TrainingRunError("validation produced a missing or non-finite real loss")
            total_loss += float(output.loss.item()) * target_count
            total_targets += target_count
        duration = time.perf_counter() - started
        return {
            "validationLoss": total_loss / total_targets,
            "validationPredictedTokens": total_targets,
            "validationSeconds": duration,
        }

    def train(self, *, until_step: int | None = None) -> dict[str, Any]:
        target = self.config.max_steps if until_step is None else until_step
        if not isinstance(target, int) or not self.global_step < target <= self.config.max_steps:
            raise TrainingRunError(
                f"until_step must be greater than current step {self.global_step} "
                f"and <= {self.config.max_steps}"
            )
        last_checkpoint: Path | None = None
        while self.global_step < target:
            metric = self._train_one_step()
            if (
                self.global_step % self.config.validation_interval == 0
                or self.global_step == target
                or self.global_step == self.config.max_steps
            ):
                metric.update(self.validate())
            self.metrics.append(metric)
            if (
                self.global_step % self.config.checkpoint_interval == 0
                or self.global_step == target
                or self.global_step == self.config.max_steps
            ):
                last_checkpoint = self.save_checkpoint()
            else:
                self._write_run_manifest(status="running")
        status = "completed" if self.global_step == self.config.max_steps else "interrupted"
        self._write_run_manifest(status=status)
        return {
            "runId": self.run_id,
            "status": status,
            "globalStep": self.global_step,
            "tokensSeen": self.tokens_seen,
            "lastCheckpoint": str(last_checkpoint) if last_checkpoint else None,
            "latestMetrics": dict(self.metrics[-1]),
        }

    def _training_state(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "trainingConfigFingerprint": self.config.fingerprint(),
            "datasetFingerprint": self.corpus.fingerprint,
            "globalStep": self.global_step,
            "tokensSeen": self.tokens_seen,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "accumulationMicroStep": 0,
            "dataCursor": self.stream.state_dict(),
            "rng": _capture_rng_state(),
            "metrics": list(self.metrics),
            "accumulatedElapsedSeconds": self._elapsed_seconds(),
        }

    def save_checkpoint(self) -> Path:
        checkpoint_root = self.run_directory / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        name = f"step-{self.global_step:08d}"
        final = checkpoint_root / name
        if final.exists():
            raise TrainingCheckpointError(f"checkpoint already exists: {final}")
        temporary = checkpoint_root / f".{name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            self.model.save_checkpoint(temporary / "model")
            state_path = temporary / "training-state.pt"
            torch.save(self._training_state(), state_path)
            file_paths = [
                temporary / "model" / "config.json",
                temporary / "model" / "weights.pt",
                temporary / "model" / "manifest.json",
                state_path,
            ]
            files = {
                path.relative_to(temporary).as_posix(): _file_descriptor(path)
                for path in file_paths
            }
            checkpoint_manifest = {
                "schemaVersion": 1,
                "pipelineId": TRAINING_PIPELINE_ID,
                "runId": self.run_id,
                "globalStep": self.global_step,
                "trainingConfig": self.config.to_dict(),
                "trainingConfigFingerprint": self.config.fingerprint(),
                "datasetFingerprint": self.corpus.fingerprint,
                "modelConfigFingerprint": self.config.model.fingerprint(),
                "codeSourceFingerprint": self.code_revision["sourceFingerprint"],
                "runtimeContract": {
                    "pythonVersion": platform.python_version(),
                    "torchVersion": torch.__version__,
                    "deviceType": self.device.type,
                    "precision": self.precision,
                },
                "files": files,
            }
            checkpoint_manifest["manifestSha256"] = _sha256_bytes(
                _canonical_json(checkpoint_manifest).encode("utf-8")
            )
            _atomic_write_json(temporary / "checkpoint-manifest.json", checkpoint_manifest)
            temporary.replace(final)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        manifest_path = final / "checkpoint-manifest.json"
        entry = {
            "globalStep": self.global_step,
            "path": final.relative_to(self.run_directory).as_posix(),
            "manifestSha256": _sha256_file(manifest_path),
            "trainingStateSha256": _sha256_file(final / "training-state.pt"),
            "modelWeightsSha256": _sha256_file(final / "model" / "weights.pt"),
        }
        self.checkpoints.append(entry)
        _atomic_write_json(
            self.run_directory / "latest-checkpoint.json",
            {
                "schemaVersion": 1,
                "runId": self.run_id,
                **entry,
            },
        )
        self._write_run_manifest(status="running")
        return final

    @classmethod
    def resume(
        cls,
        checkpoint_directory: str | Path,
        corpus: PackedCorpus,
        *,
        allocation_limit: int = DEFAULT_ALLOCATION_LIMIT,
        allow_test_corpus: bool = False,
    ) -> "Gen1Trainer":
        checkpoint = Path(checkpoint_directory)
        if checkpoint.parent.name != "checkpoints":
            raise TrainingCheckpointError("checkpoint must be inside a run checkpoints directory")
        run_directory = checkpoint.parent.parent
        checkpoint_manifest_path = checkpoint / "checkpoint-manifest.json"
        checkpoint_manifest = _read_json_object(
            checkpoint_manifest_path, "training checkpoint manifest"
        )
        declared_manifest_hash = checkpoint_manifest.get("manifestSha256")
        actual_manifest_hash = _sha256_bytes(
            _canonical_json(
                {key: value for key, value in checkpoint_manifest.items() if key != "manifestSha256"}
            ).encode("utf-8")
        )
        if declared_manifest_hash != actual_manifest_hash:
            raise TrainingCheckpointError("training checkpoint manifest checksum mismatch")
        if (
            checkpoint_manifest.get("schemaVersion") != 1
            or checkpoint_manifest.get("pipelineId") != TRAINING_PIPELINE_ID
        ):
            raise TrainingCheckpointError("training checkpoint manifest is incompatible")
        files = checkpoint_manifest.get("files")
        expected_files = {
            "model/config.json",
            "model/weights.pt",
            "model/manifest.json",
            "training-state.pt",
        }
        if not isinstance(files, Mapping) or set(files) != expected_files:
            raise TrainingCheckpointError("training checkpoint file set is invalid")
        for relative, descriptor in files.items():
            path = checkpoint / relative
            if (
                not path.is_file()
                or not isinstance(descriptor, Mapping)
                or set(descriptor) != {"bytes", "sha256"}
                or descriptor["bytes"] != path.stat().st_size
                or descriptor["sha256"] != _sha256_file(path)
            ):
                raise TrainingCheckpointError(
                    f"training checkpoint checksum/size mismatch: {relative}"
                )

        run_manifest = _read_json_object(run_directory / "run-manifest.json", "run manifest")
        run_hash = run_manifest.get("manifestSha256")
        if run_hash != _sha256_bytes(
            _canonical_json(
                {key: value for key, value in run_manifest.items() if key != "manifestSha256"}
            ).encode("utf-8")
        ):
            raise TrainingCheckpointError("run manifest checksum mismatch")
        try:
            config = Gen1TrainingConfig.from_dict(checkpoint_manifest["trainingConfig"])
        except (KeyError, ValueError) as exc:
            raise TrainingCheckpointError(f"training config is invalid: {exc}") from exc
        if checkpoint_manifest.get("trainingConfigFingerprint") != config.fingerprint():
            raise TrainingCheckpointError("training config fingerprint mismatch")
        if checkpoint_manifest.get("modelConfigFingerprint") != config.model.fingerprint():
            raise TrainingCheckpointError("model config fingerprint mismatch")
        if checkpoint_manifest.get("datasetFingerprint") != corpus.fingerprint:
            raise TrainingCheckpointError("training dataset fingerprint mismatch")
        if corpus.corpus_kind != "approved-vfai009" and not allow_test_corpus:
            raise TrainingCheckpointError("production resume requires the approved VFAI-009 corpus")
        if corpus.fingerprint != _sha256_bytes(
            _canonical_json(corpus.manifest).encode("utf-8")
        ):
            raise TrainingCheckpointError("packed corpus manifest fingerprint is invalid")
        try:
            corpus.train.validate_integrity()
            corpus.validation.validate_integrity()
        except TrainingDataContractError as exc:
            raise TrainingCheckpointError(f"packed corpus integrity failed: {exc}") from exc
        for split in (corpus.train, corpus.validation):
            if split.block_size > config.model.max_sequence_length or torch.any(
                split.input_ids[split.attention_mask] >= config.model.vocab_size
            ):
                raise TrainingCheckpointError("packed corpus is incompatible with the model")
        current_revision = _source_revision()
        if checkpoint_manifest.get("codeSourceFingerprint") != current_revision["sourceFingerprint"]:
            raise TrainingCheckpointError("training source fingerprint changed since checkpoint")

        device = _resolve_device(config.device)
        precision = _resolve_precision(config.precision, device)
        expected_runtime = {
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "deviceType": device.type,
            "precision": precision,
        }
        if checkpoint_manifest.get("runtimeContract") != expected_runtime:
            raise TrainingCheckpointError("training runtime contract changed since checkpoint")
        if (
            run_manifest.get("runId") != checkpoint_manifest.get("runId")
            or run_manifest.get("trainingConfigFingerprint") != config.fingerprint()
            or run_manifest.get("datasetFingerprint") != corpus.fingerprint
            or (run_manifest.get("codeRevision") or {}).get("sourceFingerprint")
            != current_revision["sourceFingerprint"]
        ):
            raise TrainingCheckpointError("run manifest disagrees with the selected checkpoint")

        try:
            state = torch.load(
                checkpoint / "training-state.pt", map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise TrainingCheckpointError(f"unable to load training state: {exc}") from exc
        if not isinstance(state, Mapping):
            raise TrainingCheckpointError("training state must be a mapping")
        expected_state = {
            "schemaVersion",
            "trainingConfigFingerprint",
            "datasetFingerprint",
            "globalStep",
            "tokensSeen",
            "optimizer",
            "scheduler",
            "scaler",
            "accumulationMicroStep",
            "dataCursor",
            "rng",
            "metrics",
            "accumulatedElapsedSeconds",
        }
        if set(state) != expected_state or state.get("schemaVersion") != 1:
            raise TrainingCheckpointError("training state keys/schema are invalid")
        if (
            state["trainingConfigFingerprint"] != config.fingerprint()
            or state["datasetFingerprint"] != corpus.fingerprint
            or state["globalStep"] != checkpoint_manifest.get("globalStep")
            or state["accumulationMicroStep"] != 0
        ):
            raise TrainingCheckpointError("training state compatibility fields disagree")
        model = VoltForgeGen1.from_checkpoint(
            checkpoint / "model", device=device, allocation_limit=allocation_limit
        )
        instance = object.__new__(cls)
        instance._initialize_common(
            config=config,
            corpus=corpus,
            run_directory=run_directory,
            run_id=checkpoint_manifest["runId"],
            device=device,
            precision=precision,
            model=model,
            allocation_limit=allocation_limit,
        )
        try:
            instance.optimizer.load_state_dict(state["optimizer"])
            instance.scheduler.load_state_dict(state["scheduler"])
            instance.scaler.load_state_dict(state["scaler"])
            instance.stream.load_state_dict(state["dataCursor"])
        except (ValueError, KeyError, RuntimeError, TrainingDataContractError) as exc:
            raise TrainingCheckpointError(f"resumable training state is incompatible: {exc}") from exc
        instance.started_at_utc = run_manifest["startedAtUtc"]
        instance.accumulated_elapsed_seconds = float(state["accumulatedElapsedSeconds"])
        instance._session_started = time.perf_counter()
        instance.global_step = int(state["globalStep"])
        instance.tokens_seen = int(state["tokensSeen"])
        instance.metrics = list(state["metrics"])
        instance.checkpoints = list(run_manifest["checkpoints"])
        instance.code_revision = current_revision
        instance.hardware = _hardware_manifest(device, precision)
        _restore_rng_state(state["rng"])
        instance._write_run_manifest(status="running")
        return instance

    @classmethod
    def resume_latest(
        cls,
        run_directory: str | Path,
        corpus: PackedCorpus,
        *,
        allocation_limit: int = DEFAULT_ALLOCATION_LIMIT,
        allow_test_corpus: bool = False,
    ) -> "Gen1Trainer":
        run_root = Path(run_directory)
        pointer = _read_json_object(
            run_root / "latest-checkpoint.json", "latest checkpoint pointer"
        )
        path = pointer.get("path")
        if not isinstance(path, str):
            raise TrainingCheckpointError("latest checkpoint pointer path is invalid")
        checkpoint = (run_root / path).resolve()
        try:
            checkpoint.relative_to(run_root.resolve())
        except ValueError as exc:
            raise TrainingCheckpointError("latest checkpoint path escapes the run") from exc
        manifest_path = checkpoint / "checkpoint-manifest.json"
        if pointer.get("manifestSha256") != _sha256_file(manifest_path):
            raise TrainingCheckpointError("latest checkpoint pointer checksum mismatch")
        return cls.resume(
            checkpoint,
            corpus,
            allocation_limit=allocation_limit,
            allow_test_corpus=allow_test_corpus,
        )
